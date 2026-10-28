"""Restricted tensor checkpoints and bounded atomic recovery writes."""
import os
from pathlib import Path
import random
import torch
from ncm_lab.runtime.io import digest, canonical_hash, write_json, read
from ncm_lab.runtime.budget import BudgetExceeded


def student_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith(("embeddings.", "head."))}


def selected_payload(payload):
    return {**payload, "optimizer": None, "rng": None, "role": "selected_deployment_weights"}


def load_student(model, state):
    if any(k.startswith(("embeddings.", "head.")) for k in state):
        raise ValueError("student checkpoint may not overwrite frozen target matrices")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("embeddings.", "head.")) for k in missing):
        raise ValueError("incomplete or incompatible student state")


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def storage_size(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


def preflight(meter, estimate):
    if storage_size(meter.accounting_dir) + estimate > meter.budget.max_storage_bytes:
        raise BudgetExceeded(f"checkpoint peak write requires {estimate} additional bytes")


def atomic_checkpoint(path, payload, meter=None):
    path = Path(path)
    estimate = tensor_bytes(payload) + 1024 * 1024
    if meter:
        preflight(meter, estimate)
    temporary = path.with_suffix(".partial")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if meter and storage_size(meter.accounting_dir) > meter.budget.max_storage_bytes:
            raise BudgetExceeded("actual serialized checkpoint exceeded reserved storage")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"schema_version": 2, "sha256": digest(path), "bytes": path.stat().st_size,
               "identity": payload["identity"], "format": "torch_weights_only"}
    write_json(path.with_suffix(".receipt.json"), receipt)
    return "sha256:" + receipt["sha256"]


def read_checkpoint(path, identity=None):
    path = Path(path)
    receipt = read(path.with_suffix(".receipt.json"))
    if receipt["sha256"] != digest(path) or identity is not None and receipt["identity"] != identity:
        raise ValueError("checkpoint digest/dependency mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["identity"] != receipt["identity"]:
        raise ValueError("checkpoint identity mismatch")
    return payload


def rng_state(sampler):
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "sampler": sampler.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, sampler):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    sampler.setstate(state["sampler"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def identity(model, data_hash):
    return canonical_hash({"schema": 2, "architecture": getattr(model, "compatibility", {
        "schedule": list(model.schedule), "skip": model.skip_enabled}), "data": data_hash,
        "training_contract": getattr(model, "training_contract", None)})
