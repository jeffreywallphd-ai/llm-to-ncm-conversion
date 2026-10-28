"""Restricted tensor checkpoints and bounded atomic recovery writes."""

import os
from pathlib import Path
import random
import json
import io
import zipfile
import torch
from ncm_lab.runtime.io import digest, canonical_hash, read
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


def publication_boundary(stage):
    """Fault-injection seam; production publication never runs callbacks from artifacts."""


def sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_checkpoint(path, payload, meter=None):
    path = Path(path)
    # Previous committed bundle + raw serialization + new complete bundle coexist.
    estimate = 2 * tensor_bytes(payload) + 2 * 1024 * 1024
    if meter:
        preflight(meter, estimate)
    temporary = path.with_suffix(".partial")
    serialized = path.with_suffix(".payload.partial")
    try:
        with serialized.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        publication_boundary("weights_finished")
        receipt = {
            "schema_version": 3,
            "payload_sha256": digest(serialized),
            "bytes": serialized.stat().st_size,
            "identity": payload["identity"],
            "format": "atomic_bundle_torch_weights_only",
        }
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
            archive.write(serialized, "payload.pt")
            publication_boundary("before_receipt")
            archive.writestr("receipt.json", json.dumps(receipt, sort_keys=True))
            publication_boundary("receipt_written")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if meter and storage_size(meter.accounting_dir) > meter.budget.max_storage_bytes:
            raise BudgetExceeded("actual serialized checkpoint exceeded reserved storage")
        publication_boundary("before_commit")
        os.replace(temporary, path)
        sync_directory(path.parent)
        publication_boundary("after_commit")
    finally:
        temporary.unlink(missing_ok=True)
        serialized.unlink(missing_ok=True)
    return "sha256:" + digest(path)


def read_checkpoint(path, identity=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError("no committed checkpoint available")
    with zipfile.ZipFile(path) as archive:
        if "receipt.json" in archive.namelist():
            import hashlib

            receipt = json.loads(archive.read("receipt.json"))
            body = archive.read("payload.pt")
            if (
                hashlib.sha256(body).hexdigest() != receipt["payload_sha256"]
                or identity is not None
                and receipt["identity"] != identity
            ):
                raise ValueError("checkpoint digest/dependency mismatch")
            payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
            if payload["identity"] != receipt["identity"]:
                raise ValueError("checkpoint identity mismatch")
            return payload
    # Historical two-file artifacts remain read-only and require their original receipt.
    receipt = read(path.with_suffix(".receipt.json"))
    if receipt["sha256"] != digest(path) or identity is not None and receipt["identity"] != identity:
        raise ValueError("checkpoint digest/dependency mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["identity"] != receipt["identity"]:
        raise ValueError("checkpoint identity mismatch")
    return payload


def rng_state(sampler):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "sampler": sampler.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state, sampler):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    sampler.setstate(state["sampler"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def identity(model, data_hash):
    return canonical_hash(
        {
            "schema": 2,
            "architecture": getattr(
                model, "compatibility", {"schedule": list(model.schedule), "skip": model.skip_enabled}
            ),
            "data": data_hash,
            "training_contract": getattr(model, "training_contract", None),
        }
    )
