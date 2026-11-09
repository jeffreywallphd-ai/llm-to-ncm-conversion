"""Frozen parent-aligned teacher features and raw fine codes, outside generator execution."""

from contextlib import closing
from pathlib import Path
import sqlite3
import torch
from ncm_lab.block8.data import BlockDataset
from ncm_lab.block8.runtime import admission, checked, charge_write
from ncm_lab.runtime.io import canonical_hash, digest, read, reference, write_json
from ncm_lab.stage1.io import account
from ncm_lab.teacher.model import load_original, model_identity, prediction_states

FEATURES = "hierarchical_block8_teacher_features_v1"
TARGETS = "hierarchical_block8_fine_targets_v1"


def binding(settings, data, kind):
    return dict(
        schema=kind,
        data_sha256=data.sha256,
        architecture=settings["architecture"],
        block_contract=settings["block_contract"],
        model=model_identity(settings["model"]),
        template_sha256=data.manifest["template_sha256"],
        teacher_dtype=settings["features"]["teacher_dtype"],
        alignment="post_final_norm_at_prompt_length_minus_one_plus_answer_index",
    )


class TensorWriter:
    def __init__(self, run, settings, identity):
        self.run, self.settings, self.identity = run, settings, identity
        self.path = run.path / "cache"
        self.path.mkdir()
        self.db = sqlite3.connect(self.path / "index.sqlite")
        page = self.db.execute("PRAGMA page_size").fetchone()[0]
        self.db.execute(f"PRAGMA max_page_count={settings['data']['max_index_bytes'] // page}")
        self.db.execute(
            "CREATE TABLE tensors(parent INTEGER PRIMARY KEY, split TEXT, length INTEGER, file TEXT, sha TEXT, dtype TEXT, width INTEGER, parent_sha TEXT)"
        )
        self.count, self.positions = {}, {}

    def append(self, key, parent, tensor):
        if (
            tensor.ndim != 2
            or tensor.shape[0] != len(parent.answer_ids)
            or tensor.requires_grad
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("finite detached full-parent tensor required")
        self.run.meter.check_disk(tensor.numel() * tensor.element_size() + 65536)
        file = self.path / f"parent-{key:08d}.pt"
        with account(self.run.meter, file, tensor.numel() * tensor.element_size() + 65536):
            torch.save(tensor.detach().to("cpu").contiguous().clone(), file)
        charge_write(self.run, 16384)
        self.db.execute(
            "INSERT INTO tensors VALUES(?,?,?,?,?,?,?,?)",
            (
                key,
                parent.split,
                tensor.shape[0],
                file.name,
                digest(file, self.run.meter.check),
                str(tensor.dtype).removeprefix("torch."),
                tensor.shape[1],
                parent.identity,
            ),
        )
        self.db.commit()
        self.count[parent.split] = self.count.get(parent.split, 0) + 1
        self.positions[parent.split] = self.positions.get(parent.split, 0) + tensor.shape[0]
        self.run.meter.check_disk()

    def finish(self, extra=None):
        self.db.close()
        value = dict(
            **self.identity,
            index_sha256=digest(self.path / "index.sqlite", self.run.meter.check),
            parents=self.count,
            valid_positions=self.positions,
            extra=extra or {},
            tensor_scope="one valid-position tensor per parent; view pooling is block-local",
        )
        write_json(self.path / "manifest.json", value, exclusive=True)
        return reference(self.path / "manifest.json")

    def close(self):
        self.db.close()


class TensorStore:
    def __init__(self, path, settings, data, kind, check=None):
        self.path = Path(path)
        self.manifest = read(path)
        self.check = check
        expected = binding(settings, data, kind)
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("incompatible hierarchical parent tensor identity")
        if digest(self.path.parent / "index.sqlite", check) != self.manifest["index_sha256"]:
            raise ValueError("changed hierarchical tensor index")
        self.sha256 = digest(path, check)
        self.db = sqlite3.connect(
            (self.path.parent / "index.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
        )
        self.ids = {
            s: [
                r[0]
                for r in self.db.execute("SELECT parent FROM tensors WHERE split=? ORDER BY parent", (s,))
            ]
            for s in ("train", "dev")
        }
        if {s: len(v) for s, v in self.ids.items()} != self.manifest["parents"]:
            self.close()
            raise ValueError("hierarchical cache parent counts differ")

    def close(self):
        self.db.close()

    def get(self, key, parent):
        row = self.db.execute(
            "SELECT split,length,file,sha,dtype,width,parent_sha FROM tensors WHERE parent=?", (key,)
        ).fetchone()
        if (
            row is None
            or row[0] != parent.split
            or row[1] != len(parent.answer_ids)
            or row[6] != parent.identity
        ):
            raise ValueError("missing/mismatched hierarchical cached parent")
        file = (self.path.parent / row[2]).resolve()
        if file.parent != self.path.parent.resolve() or digest(file, self.check) != row[3]:
            raise ValueError("corrupt hierarchical parent tensor")
        value = torch.load(file, weights_only=True, mmap=True, map_location="cpu")
        if (
            list(value.shape) != [row[1], row[5]]
            or str(value.dtype).removeprefix("torch.") != row[4]
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("invalid hierarchical parent tensor shape/precision")
        return value


def fixed_views(data, ids, limit, seed):
    """Parent-balanced fixed offsets, independent of student/teacher output quality."""
    ordered = sorted(ids, key=lambda k: canonical_hash([seed, data.parent(k).original_id]))
    rows = []
    for turn in range((data.manifest["policy"]["max_answer_tokens"] + 7) // 8):
        for key in ordered:
            count = (len(data.parent(key).answer_ids) + 7) // 8
            if turn < count:
                shift = int(canonical_hash([seed, key])[:8], 16) % count
                rows.append((key, 8 * ((shift + turn) % count)))
                if len(rows) >= limit:
                    return rows
    return rows


def teacher_features(run, settings, data_ref):
    with closing(BlockDataset(checked(data_ref), settings, run.meter.check)) as data:
        options = settings["features"]
        train = sorted(
            data.ids["train"], key=lambda k: canonical_hash([settings["seed"], data.parent(k).original_id])
        )
        selected, positions = [], 0
        for key in train:
            if positions >= options["target_train_positions"]:
                break
            selected.append(key)
            positions += len(data.parent(key).answer_ids)
        if positions < options["minimum_train_positions"]:
            raise ValueError("insufficient unique train answer positions for hierarchical codec subset")
        dev = data.ids["dev"][: options["max_dev_parents"]]
        if not dev:
            raise ValueError("empty hierarchical teacher dev subset")
        admission(run, settings, training=False, kind="teacher", data=data)
        teacher = load_original(
            settings["model"], settings["device"], getattr(torch, options["teacher_dtype"]), frozen=True
        )
        with (
            closing(TensorWriter(run, settings, binding(settings, data, FEATURES))) as writer,
            torch.no_grad(),
        ):
            for key in selected + dev:
                parent = data.parent(key)
                run.meter.charge(
                    source=len(parent.prompt_ids) + len(parent.answer_ids) - 1,
                    answer=len(parent.answer_ids),
                    steps=1,
                )
                states = prediction_states(teacher, parent.prompt_ids, parent.answer_ids)
                writer.append(key, parent, states)
            artifact = writer.finish(
                dict(
                    train_positions=positions,
                    whole_parent_overshoot_bound=settings["data"]["max_answer_tokens"] - 1,
                    target_train_positions=options["target_train_positions"],
                    teacher_frozen=not any(p.requires_grad for p in teacher.parameters()),
                    test_parents_used=False,
                )
            )
        return dict(features=artifact, train_positions=positions, teacher_frozen=True)


def targets(run, settings, data_ref, features_ref, codec_ref):
    from ncm_lab.block8.training import load_codec

    with (
        closing(BlockDataset(checked(data_ref), settings, run.meter.check)) as data,
        closing(TensorStore(checked(features_ref), settings, data, FEATURES, run.meter.check)) as features,
    ):
        admission(run, settings, training=False, kind="targets", data=data)
        codec, receipt = load_codec(checked(codec_ref), settings, data, settings["device"], run.meter.check)
        teacher = load_original(
            settings["model"],
            settings["device"],
            getattr(torch, settings["features"]["teacher_dtype"]),
            frozen=True,
        )
        identity = binding(settings, data, TARGETS)
        identity.update(
            codec=codec_ref,
            encoder_dtype="float32",
            storage_dtype="float32",
            generator_boundary_cast="explicit_target_to_generator_parameter_dtype_v1",
        )
        cached = set(features.ids["train"] + features.ids["dev"])
        with closing(TensorWriter(run, settings, identity)) as writer, torch.no_grad():
            for key in data.ids["train"] + data.ids["dev"]:
                parent = data.parent(key)
                run.meter.charge(
                    source=len(parent.prompt_ids) + len(parent.answer_ids) - 1,
                    answer=len(parent.answer_ids),
                    canvas=len(parent.answer_ids),
                    steps=1,
                )
                states = (
                    features.get(key, parent).to(settings["device"])
                    if key in cached
                    else prediction_states(teacher, parent.prompt_ids, parent.answer_ids)
                )
                fine = codec.fine_codes(states.float()).detach()
                writer.append(key, parent, fine)
            artifact = writer.finish(
                dict(
                    teacher_frozen=not any(p.requires_grad for p in teacher.parameters()),
                    codec_frozen=not any(p.requires_grad for p in codec.parameters()),
                    test_parents_used=False,
                    scales=receipt["scales"],
                    normalization=receipt["normalization"],
                )
            )
        return dict(targets=artifact, codec=codec_ref, teacher_frozen=True, codec_frozen=True)
