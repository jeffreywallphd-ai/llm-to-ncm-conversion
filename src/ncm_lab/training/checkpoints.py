"""Native sharded state: one bounded CPU tensor chunk at a time, atomic manifest pointer."""

import os
from pathlib import Path
import random
import uuid
import torch
from ncm_lab.runtime.io import read, write_json, atomic_json, digest, canonical_hash

RECOVERY = "native_recovery_v1"
INFERENCE = "native_inference_v1"


def boundary(stage):
    """Fault-injection seam, never an artifact-provided callback."""


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def rng_state(sampler):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.getstate(),
    }


def restore_rng(state, sampler):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    sampler.setstate(state["sampler"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    folder, payload, *, kind=RECOVERY, meter=None, shard_bytes=64 * 1024 * 1024, pointer="current.json"
):
    if kind not in {RECOVERY, INFERENCE} or shard_bytes < 1024:
        raise ValueError("native checkpoint kind/shard budget invalid")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if meter:
        meter.check_disk(tensor_bytes(payload) + 1024 * 1024)
    version = uuid.uuid4().hex
    pending = folder / (".pending-" + version)
    pending.mkdir()
    files = {}
    number = 0
    part_number = 0
    buffer = {}
    buffer_bytes = 0

    def flush():
        nonlocal number, buffer_bytes, buffer
        if not buffer:
            return
        if meter:
            meter.check()
        name = f"shard-{number:06d}.pt"
        number += 1
        path = pending / name
        with path.open("xb") as stream:
            torch.save(buffer, stream)
            stream.flush()
            os.fsync(stream.fileno())
        files[name] = {"sha256": digest(path, meter.check if meter else None), "bytes": path.stat().st_size}
        buffer = {}
        buffer_bytes = 0
        boundary("shard_finished")

    def encode(value):
        nonlocal part_number, buffer_bytes
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or not value.is_contiguous():
                raise ValueError("checkpoint tensor must be dense/contiguous; no implicit whole-state copy")
            pieces = []
            flat = value.detach().view(-1)
            count = max(1, shard_bytes // value.element_size())
            for start in range(0, value.numel(), count):
                if meter:
                    meter.check()
                elements = min(count, value.numel() - start)
                needed = elements * value.element_size()
                if buffer_bytes + needed > shard_bytes:
                    flush()
                name = f"shard-{number:06d}.pt"
                key = f"part-{part_number:08d}"
                part_number += 1
                # clone prevents torch.save serializing the complete backing storage.
                part = flat[start : start + count].to("cpu").clone()
                pieces.append({"file": name, "key": key, "start": start, "elements": part.numel()})
                buffer[key] = part
                buffer_bytes += needed
                if buffer_bytes >= shard_bytes:
                    flush()
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "pieces": pieces,
            }
        if isinstance(value, dict):
            return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {
                "type": "tuple" if isinstance(value, tuple) else "list",
                "items": [encode(v) for v in value],
            }
        if value is None or isinstance(value, (str, int, float, bool)):
            return {"type": "value", "value": value}
        raise ValueError("unsupported checkpoint metadata: " + str(type(value)))

    tree = encode(payload)
    flush()
    boundary("before_manifest")
    manifest = {
        "schema": kind,
        "identity": payload["identity"],
        "files": files,
        "tree": tree,
        "shard_payload_limit": shard_bytes,
    }
    write_json(pending / "manifest.json", manifest, exclusive=True)
    boundary("manifest_written")
    if meter:
        meter.check_disk()
    boundary("before_commit")
    committed = folder / ("version-" + version)
    os.replace(pending, committed)
    boundary("after_commit")
    result = {
        "schema": "native_checkpoint_pointer_v1",
        "manifest": str(committed / "manifest.json"),
        "sha256": digest(committed / "manifest.json"),
    }
    boundary("before_pointer")
    atomic_json(folder / pointer, result)
    boundary("after_pointer")
    return result


class CheckpointReader:
    def __init__(self, path, *, expected_kind=RECOVERY, identity=None, check=None):
        path = Path(path)
        if path.is_dir():
            path = path / "manifest.json"
        value = read(path)
        if value.get("schema") == "native_checkpoint_pointer_v1":
            target = Path(value["manifest"])
            if digest(target, check) != value["sha256"]:
                raise ValueError("native checkpoint pointer digest mismatch")
            path = target
            value = read(path)
        if value.get("schema") != expected_kind:
            raise ValueError(
                "native checkpoint schema mismatch; legacy and inference artifacts cannot resume training"
            )
        if identity is not None and value["identity"] != identity:
            raise ValueError("native checkpoint dependency identity mismatch")
        self.path = path
        self.manifest = value
        self.check = check
        self._cached_file = None
        self._cached_shard = None
        # Verify ALL shards before modifying any live parameter or optimizer state.
        for name, receipt in value["files"].items():
            source = (path.parent / name).resolve()
            if (
                not source.is_relative_to(path.parent.resolve())
                or source.stat().st_size != receipt["bytes"]
                or digest(source, check) != receipt["sha256"]
            ):
                raise ValueError("corrupt/missing native checkpoint shard: " + name)

    def field(self, name):
        for key, value in self.manifest["tree"]["items"]:
            if key.get("value") == name:
                return value
        raise ValueError("missing native checkpoint field " + name)

    def copy_tensor(self, node, destination):
        if node["shape"] != list(destination.shape) or node["dtype"] != str(destination.dtype).removeprefix(
            "torch."
        ):
            raise ValueError("native checkpoint tensor shape/dtype mismatch")
        flat = destination.view(-1)
        expected = 0
        with torch.no_grad():
            for piece in node["pieces"]:
                if self.check:
                    self.check()
                if piece["start"] != expected or piece["file"] not in self.manifest["files"]:
                    raise ValueError("invalid native shard extent")
                if self._cached_file != piece["file"]:
                    self._cached_shard = None
                    self._cached_shard = torch.load(
                        self.path.parent / piece["file"], weights_only=True, mmap=True, map_location="cpu"
                    )
                    self._cached_file = piece["file"]
                tensor = self._cached_shard[piece["key"]]
                if (
                    tensor.numel() != piece["elements"]
                    or str(tensor.dtype).removeprefix("torch.") != node["dtype"]
                ):
                    raise ValueError("native tensor shard metadata mismatch")
                flat[expected : expected + tensor.numel()].copy_(tensor)
                expected += tensor.numel()
                del tensor
        if expected != destination.numel():
            raise ValueError("incomplete native tensor extent")

    def materialize(self, node, max_tensor_bytes=1024 * 1024):
        kind = node["type"]
        if kind == "value":
            return node["value"]
        if kind == "tensor":
            dtype = getattr(torch, node["dtype"])
            elements = 1
            for size in node["shape"]:
                elements *= size
            if elements * torch.empty((), dtype=dtype).element_size() > max_tensor_bytes:
                raise ValueError("large checkpoint tensor must stream directly into destination")
            value = torch.empty(node["shape"], dtype=dtype)
            self.copy_tensor(node, value)
            return value
        if kind == "dict":
            return {
                self.materialize(k, max_tensor_bytes): self.materialize(v, max_tensor_bytes)
                for k, v in node["items"]
            }
        values = [self.materialize(v, max_tensor_bytes) for v in node["items"]]
        return tuple(values) if kind == "tuple" else values

    def load_model(self, model):
        nodes = {self.materialize(k): v for k, v in self.field("model")["items"]}
        state = model.state_dict(keep_vars=True)
        if set(nodes) != set(state):
            raise ValueError("incomplete native model state (including embeddings/head/norms)")
        for name, destination in state.items():
            if nodes[name]["shape"] != list(destination.shape):
                raise ValueError("native model shape mismatch")
        for name, destination in state.items():
            self.copy_tensor(nodes[name], destination)

    def load_optimizer(self, optimizer):
        nodes = {self.materialize(k): v for k, v in self.field("optimizer")["items"]}
        groups = self.materialize(nodes["groups"])
        if groups != optimizer.groups or self.materialize(nodes["schema"]) != "fp32_moment_adamw_v1":
            raise ValueError("native optimizer parameter groups/precision mismatch")
        if self.materialize(nodes["weight_decay"]) != optimizer.weight_decay:
            raise ValueError("native optimizer weight-decay contract mismatch")
        optimizer.state = {}
        for key, node in nodes["state"]["items"]:
            name = self.materialize(key)
            if name not in optimizer.parameters:
                raise ValueError("unknown optimizer parameter")
            values = {self.materialize(k): v for k, v in node["items"]}
            state = {"step": self.materialize(values["step"])}
            for field in ("exp_avg", "exp_avg_sq"):
                tensor = torch.empty_like(optimizer.parameters[name], dtype=torch.float32)
                self.copy_tensor(values[field], tensor)
                state[field] = tensor
            optimizer.state[name] = state


def recovery_payload(model, optimizer, identity, metadata, state, sampler):
    return {
        "identity": identity,
        "metadata": metadata,
        "model": model.state_dict(keep_vars=True),
        "optimizer": optimizer.state_dict(),
        "training": state,
        "rng": rng_state(sampler),
        "scheduler": {"schema": "constant_separate_rates_v1"},
        "scaler": None,
    }


def native_identity(metadata):
    return canonical_hash(metadata)
