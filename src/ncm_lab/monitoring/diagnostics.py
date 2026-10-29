"""Bounded detached reductions; no extra backward, parameter copies or gold masks."""

import math
import torch
from ncm_lab.monitoring.records import Measure


@torch.no_grad()
def rms(tensor):
    flat = tensor.detach().reshape(-1)
    total = sum(float(flat[i : i + 1048576].float().square().sum()) for i in range(0, flat.numel(), 1048576))
    return math.sqrt(total / flat.numel()) if flat.numel() else None


@torch.no_grad()
def tensor_statistics(tensor):
    flat = tensor.detach().reshape(-1)
    values = flat[:: max(1, (flat.numel() + 4095) // 4096)][:4096].float()
    n = values.numel()
    return {
        "count": n,
        "mean": float(values.mean()) if n else None,
        "rms": float(values.square().mean().sqrt()) if n else None,
        "sampled_pooled_scalar_variance": float(values.var(unbiased=True)) if n > 1 else None,
        "sampling": "at most 4096 evenly strided scalar elements; not across-example semantic diversity",
    }


class Activity:
    def __init__(self):
        self.rows = []
        self.logical_count = 0

    def __call__(self, level, slots, hidden, compact, residual, bridge, gate):
        z, dz, h, dh = (rms(x) for x in (compact, residual, hidden, bridge))
        row = {
            "update": level + 1,
            "resolution": slots,
            "gate_effective": float(gate.detach()),
            "latent_base_rms": z,
            "latent_refinement_rms": dz,
            "hidden_base_rms": h,
            "hidden_injection_rms": dh,
            "latent_refinement_base_ratio": dz / z if z is not None and z > 1e-12 else None,
            "hidden_injection_base_ratio": dh / h if h is not None and h > 1e-12 else None,
        }
        if level < len(self.rows):
            self.rows[level] = row
        else:
            self.rows.append(row)
        self.logical_count += 1


@torch.no_grad()
def gradients(model):
    """Each parameter identity once; combined objective, pre-clipping gradient."""
    result = {}
    owners = {}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith("refinements."):
            group = int(name.split(".")[1]) + 1
            owner = f"shared_group_{group:02d}_updates_{2 * group - 1:02d}_{2 * group:02d}"
        elif name.startswith("backbone.model.layers."):
            block = int(name.split(".")[3]) + 1
            if block not in (1, 9, 24, 36):
                continue
            owner = f"backbone_block_{block:02d}"
        else:
            owner = name.replace(".", "/")
        values = owners.setdefault(owner, [])
        status = "frozen" if not p.requires_grad else "absent" if p.grad is None else "valid"
        norm = rms(p.grad) * math.sqrt(p.numel()) if p.grad is not None else None
        if norm is not None:
            status = (
                "nonfinite" if not math.isfinite(norm) else "finite_zero" if norm == 0 else "finite_nonzero"
            )
        values.append((norm, status))
    for owner, values in owners.items():
        statuses = {status for _, status in values}
        status = (
            "nonfinite"
            if "nonfinite" in statuses
            else "finite_nonzero"
            if "finite_nonzero" in statuses
            else "finite_zero"
            if "finite_zero" in statuses
            else "absent"
            if "absent" in statuses
            else "frozen"
        )
        total = (
            math.sqrt(sum(v * v for v, _ in values if v is not None))
            if any(v is not None for v, _ in values)
            else None
        )
        result["gradients/" + owner + "/combined_l2"] = Measure(
            total,
            len(values),
            reduction="unique parameters; accumulated unscaled gradient before clipping",
            status=status,
        )
    return result
