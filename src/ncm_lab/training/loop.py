import json
from pathlib import Path
import random
import time
import torch
from ncm_lab.models.upstream import FeatureExtractor, checkpoint_id
from ncm_lab.runtime.budget import BudgetExceeded
from ncm_lab.training.losses import final_token_loss


def resolve_training_data(config):
    from ncm_lab.runtime.io import root, read, digest
    if config.get("training_data_reference"):
        reference = read(root()/config["training_data_reference"])
        path = Path(reference["path"])
        if not path.is_file() or digest(path) != reference["sha256"]:
            raise ValueError("prepared teacher-data artifact is missing or changed")
        return path
    return root()/config["training_data"]


def load_training(path, origin="target_distillation"):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("split") != "train" or row.get("origin") != origin:
            raise ValueError("training requires declared train-only target distillation (fixtures only in smoke mode)")
        if not row.get("group_id") or not isinstance(row.get("input_ids"), list):
            raise ValueError("training record lacks token IDs or split-group identity")
    if not rows:
        raise ValueError("empty training corpus")
    return rows


def save_checkpoint(path, drafter, optimizer, counters, data_hash, best_dev_loss=None):
    torch.save({"model": drafter.state_dict(), "optimizer": optimizer.state_dict(), "counters": counters,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "data_hash": data_hash,
                "schedule": drafter.schedule, "best_dev_loss": best_dev_loss}, path)
    return checkpoint_id(path)


def restore_checkpoint(path, drafter, optimizer, data_hash):
    # Only locally produced, integrity-checked checkpoints are accepted by the CLI.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["data_hash"] != data_hash or tuple(checkpoint["schedule"]) != drafter.schedule:
        raise ValueError("resume corpus or architecture mismatch")
    drafter.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    random.setstate(checkpoint["python_rng"])
    return checkpoint["counters"]


def train(target, drafter, records, meter, data_hash, steps=2, seed=0, resume=None, learning_rate=1e-3):
    random.seed(seed)
    torch.manual_seed(seed)
    extractor = FeatureExtractor(target, drafter.backbone.target_layer_ids)
    optimizer = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad], lr=learning_rate)
    counters = {"optimizer_steps": 0, "source_token_passes": 0, "supervised_future_positions": 0,
                "sampled_anchors": 0, "unique_teacher_tokens": sum(len(r["input_ids"]) for r in records),
                "generated_teacher_tokens": sum(r.get("generated_tokens", 0) for r in records),
                "losses": [], "label_positions": []}
    if resume:
        counters = restore_checkpoint(resume, drafter, optimizer, data_hash)
    device = next(drafter.parameters()).device
    started, status, reason = time.monotonic(), "completed", None
    drafter.train()
    # Frozen modules stay frozen and eval even after drafter.train().
    drafter.embeddings.eval()
    drafter.head.eval()
    try:
        while counters["optimizer_steps"] < steps:
            row = records[counters["optimizer_steps"] % len(records)]
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            horizon = drafter.horizon
            if ids.shape[1] <= horizon + 1:
                raise ValueError("training sequence is too short for the declared anchor and horizon")
            anchor = random.randrange(1, ids.shape[1] - horizon)
            meter.check(ids.numel(), horizon)
            # Frozen target pass is inside measured training time and source-token accounting.
            features = extractor(ids)
            positions = torch.arange(ids.shape[1], device=device)[None]
            logits = drafter(features, positions, torch.zeros_like(positions),
                             torch.tensor([anchor], device=device), ids[:, anchor],
                             torch.zeros(1, dtype=torch.long, device=device))
            labels = ids[:, anchor + 1:anchor + 1 + horizon]
            loss = final_token_loss(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            meter.record(ids.numel(), horizon)
            counters["optimizer_steps"] += 1
            counters["source_token_passes"] += ids.numel()
            counters["supervised_future_positions"] += horizon
            counters["sampled_anchors"] += 1
            counters["losses"].append(float(loss.detach()))
            counters["label_positions"].append([anchor + 1, anchor + horizon])
            checkpoint_path = meter.artifact_dir / f"step-{counters['optimizer_steps']:06d}.pt"
            save_checkpoint(checkpoint_path, drafter, optimizer, counters, data_hash)
    except BudgetExceeded as exc:
        status, reason = "cancelled", str(exc)
    except KeyboardInterrupt:
        status, reason = "cancelled", "interrupted; recover from last complete checkpoint"
    final = meter.artifact_dir / "final.pt"
    artifact_id = save_checkpoint(final, drafter, optimizer, counters, data_hash)
    return {"status": status, "reason": reason, **counters, "elapsed_seconds": time.monotonic() - started,
            "parameter_counts": drafter.parameter_counts(), "final_checkpoint_artifact_id": artifact_id,
            "final_checkpoint": final.name,
            "selection": "last budgeted checkpoint; not selected for publication without a dev selection record"}


def select_checkpoint(development_records):
    valid = [r for r in development_records if r.get("split") == "dev" and r.get("dev_loss") is not None]
    if not valid or len(valid) != len(development_records):
        raise ValueError("checkpoint selection requires development-only finite losses")
    if any(not torch.isfinite(torch.tensor(r["dev_loss"])) for r in valid):
        raise ValueError("non-finite checkpoint selection loss")
    return min(valid, key=lambda r: (r["dev_loss"], r["step"], r["artifact_id"]))
