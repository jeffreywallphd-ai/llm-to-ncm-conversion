"""Bounded per-example tensor shards and streamed content-bound indices."""

from pathlib import Path
from dataclasses import asdict
import json
import torch
from ncm_lab.runtime.io import write_json, read, digest, resolve_reference
from ncm_lab.data.records import Example

KINDS = {"native_data_v1", "native_features_v1", "native_targets_v1"}


class ArtifactWriter:
    def __init__(self, path, kind, identity, meter=None):
        if kind not in KINDS:
            raise ValueError("native data/cache schema required")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.kind = kind
        self.identity = identity
        self.meter = meter
        self.count = 0
        self.index = (self.path / "records.jsonl").open("x", encoding="utf-8", newline="\n")
        self.stats = {
            "examples": 0,
            "unique_source_tokens": 0,
            "unique_answer_positions": 0,
            "padding_positions": 0,
            "source_types": {},
            "answer_length_histogram": {},
        }
        self.groups = {"train": set(), "dev": set()}
        self.ids = set()

    def append(self, example, tensor=None):
        if (example.source_type == "software_fixture") != bool(self.identity.get("model", {}).get("fixture")):
            raise ValueError("fixture lineage cannot be relabeled as real source data")
        if example.provenance.get("template_sha256") != self.identity.get("template_sha256"):
            raise ValueError("example and artifact prompt template differ")
        if example.original_id in self.ids:
            raise ValueError("duplicate original source example")
        if example.group_id in self.groups["dev" if example.split == "train" else "train"]:
            raise ValueError("conversation/document group crosses train/dev")
        self.ids.add(example.original_id)
        self.groups[example.split].add(example.group_id)
        row = {"example": asdict(example), "example_sha256": example.identity}
        if tensor is not None:
            if tensor.shape[0] != len(example.answer_ids):
                raise ValueError("cached answer positions differ from reference")
            if self.meter:
                self.meter.check_disk(tensor.numel() * tensor.element_size() + 4096)
            name = f"shard-{self.count:08d}.pt"
            path = self.path / name
            torch.save(tensor.detach().to("cpu").contiguous().clone(), path)
            row["tensor"] = {
                "file": name,
                "sha256": digest(path, self.meter.check if self.meter else None),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": path.stat().st_size,
            }
        self.index.write(json.dumps(row, allow_nan=False) + "\n")
        self.index.flush()
        self.count += 1
        self.stats["examples"] += 1
        self.stats["unique_source_tokens"] += len(example.prompt_ids) + len(example.answer_ids)
        self.stats["unique_answer_positions"] += len(example.answer_ids)
        canvas = self.identity.get(
            "canvas", self.identity.get("architecture", {}).get("canvas", len(example.answer_ids))
        )
        self.stats["padding_positions"] += canvas - len(example.answer_ids)
        origin = example.source_type
        self.stats["source_types"][origin] = self.stats["source_types"].get(origin, 0) + 1
        bucket = str(min(1024, 2 ** (len(example.answer_ids) - 1).bit_length()))
        self.stats["answer_length_histogram"][bucket] = (
            self.stats["answer_length_histogram"].get(bucket, 0) + 1
        )

    def finish(self, extra=None):
        self.index.close()
        if not self.count:
            raise ValueError("empty native artifact")
        manifest = {
            "schema": self.kind,
            "identity": self.identity,
            "index_sha256": digest(self.path / "records.jsonl"),
            "records": self.count,
            "stats": self.stats,
            "groups": {k: sorted(v) for k, v in self.groups.items()},
            "unique_units": "distinct prepared-example token positions; repeated epochs excluded; shared prompt prefixes across different examples counted separately",
            "extra": extra or {},
        }
        write_json(self.path / "manifest.json", manifest, exclusive=True)
        return self.path / "manifest.json"


class Dataset:
    def __init__(self, path, kind=None, identity=None):
        path = resolve_reference(path)
        self.path = path.parent
        self.manifest = read(path)
        if self.manifest.get("schema") not in KINDS or kind and self.manifest["schema"] != kind:
            raise ValueError("native artifact schema mismatch; legacy caches are incompatible")
        if identity is not None and self.manifest["identity"] != identity:
            raise ValueError("native artifact dependency mismatch")
        if digest(self.path / "records.jsonl") != self.manifest["index_sha256"]:
            raise ValueError("native artifact index changed")
        self.sha256 = digest(path)
        if not 0 < self.manifest["records"] <= 1000000:
            raise ValueError("native initial dataset metadata cap is one million examples")
        self.offsets = []
        with (self.path / "records.jsonl").open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                self.offsets.append(offset)
        if len(self.offsets) != self.manifest["records"]:
            raise ValueError("native artifact record count mismatch")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        row = self.metadata(index)
        example = Example(**row["example"])
        if example.identity != row["example_sha256"]:
            raise ValueError("native example changed")
        tensor = None
        if "tensor" in row:
            info = row["tensor"]
            path = (self.path / info["file"]).resolve()
            if not path.is_relative_to(self.path.resolve()) or digest(path) != info["sha256"]:
                raise ValueError("corrupt native data shard")
            tensor = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
            if list(tensor.shape) != info["shape"] or str(tensor.dtype) != info["dtype"]:
                raise ValueError("native cached tensor shape/dtype mismatch")
        return example, tensor

    def metadata(self, index):
        with (self.path / "records.jsonl").open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        return row

    def indices(self, split):
        return [i for i, row in self.iter_metadata() if row["example"]["split"] == split]

    def iter_metadata(self):
        with (self.path / "records.jsonl").open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                yield index, json.loads(line)


class Cursor:
    def __init__(self, indices, sampler, state=None):
        if not indices:
            raise ValueError("nonempty training subset required")
        self.sampler = sampler
        self.order = list(indices)
        self.position = 0
        self.epoch = 0
        if state:
            self.order = state["order"]
            self.position = state["position"]
            self.epoch = state["epoch"]
        else:
            sampler.shuffle(self.order)

    def take(self, count, epochs=None):
        values = []
        for _ in range(count):
            if self.position == len(self.order):
                if epochs is not None and self.epoch + 1 >= epochs:
                    break
                self.position = 0
                self.epoch += 1
                self.sampler.shuffle(self.order)
            values.append(self.order[self.position])
            self.position += 1
        return values

    def state(self):
        return {"order": self.order, "position": self.position, "epoch": self.epoch}
