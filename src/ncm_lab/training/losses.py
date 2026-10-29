"""Exact chunked text CE and compact auxiliary targets, outside student forward."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ncm_lab.embedded.geometry import masked_pool


def chunked_ce(head, states, labels, valid, chunk_positions=8):
    if chunk_positions <= 0 or labels.shape != valid.shape or states.shape[:2] != labels.shape:
        raise ValueError("invalid parallel CE shapes/chunk budget")
    selected = states[valid]
    targets = labels[valid]
    if not targets.numel():
        raise ValueError("no valid answer supervision")
    total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_positions):
        x = selected[start : start + chunk_positions]
        y = targets[start : start + chunk_positions]

        def loss(values, weight, truth):
            return F.cross_entropy(F.linear(values, weight).float(), truth, reduction="sum")

        if torch.is_grad_enabled():
            total = total + checkpoint(loss, x, head.weight, y, use_reentrant=False)
        else:
            total = total + loss(x, head.weight, y)
    return total / targets.numel()


def auxiliary_loss(predicted, fine_targets, valid, resolutions, scales, diagnostics=None):
    levels = list(range(1, len(resolutions), 2))
    if len(predicted) != len(levels):
        raise ValueError("one cumulative prediction required after every update pair")
    losses = []
    counts = []
    for prediction, level in zip(predicted, levels):
        target, mask = masked_pool(fine_targets.detach(), valid, resolutions[level])
        count = int(mask.sum())
        if count:
            scale = torch.as_tensor(
                scales[str(level)], device=prediction.device, dtype=torch.float32
            ).clamp_min(1e-3)
            losses.append(((prediction.float() - target.float()) / scale)[mask].square().mean())
        else:
            losses.append(prediction.sum() * 0)
        counts.append(count)
        if diagnostics is not None:
            raw_sum = (
                float((prediction.detach().float() - target.float())[mask].square().sum()) if count else 0.0
            )
            elements = count * prediction.shape[-1]
            diagnostics.append(
                {
                    "level": level,
                    "resolution": resolutions[level],
                    "vectors": count,
                    "elements": elements,
                    "raw_sum": raw_sum,
                    "normalized_sum": float(losses[-1].detach()) * elements,
                }
            )
    active = [value for value, count in zip(losses, counts) if count]
    if not active:
        raise ValueError("all auxiliary groups have empty target supervision")
    return torch.stack(active).mean(), losses, counts


def normalized_mse(prediction, target, valid, normalization):
    if not bool(valid.any()) or normalization <= 0:
        raise ValueError("positive training-derived normalization and valid states required")
    return (prediction.float() - target.float())[valid].square().mean() / normalization


def codec_loss(
    codec,
    head,
    states,
    valid,
    normalization,
    sampler,
    beta=0.1,
    max_head_positions=32,
    chunk_positions=8,
    diagnostics=None,
    cutoff_index=None,
):
    if beta < 0 or not 1 <= max_head_positions <= 32 or not 1 <= chunk_positions <= 8:
        raise ValueError("bounded codec head-alignment configuration required")
    fine = codec.fine_codes(states)
    cumulative, _, _ = codec.targets(fine, valid)
    cutoff = sampler.randrange(len(cumulative) - 1) if cutoff_index is None else cutoff_index
    if type(cutoff) is not int or not 0 <= cutoff < len(cumulative) - 1:
        raise ValueError("invalid planned coarse cutoff index")
    full = codec.reconstruct(cumulative[-1], len(cumulative) - 1)
    coarse = codec.reconstruct(cumulative[cutoff], cutoff)
    mse_full = normalized_mse(full, states, valid, normalization)
    mse_cutoff = normalized_mse(coarse, states, valid, normalization)
    available = valid.flatten().nonzero().flatten().tolist()
    indices = sampler.sample(available, min(max_head_positions, len(available)))
    if not indices:
        raise ValueError("empty codec batch")
    teacher = states.flatten(0, 1)[indices].detach()
    reconstruction = coarse.flatten(0, 1)[indices]
    kl = states.new_zeros((), dtype=torch.float32)
    for start in range(0, len(indices), chunk_positions):
        original = teacher[start : start + chunk_positions]
        recovered = reconstruction[start : start + chunk_positions]

        def consistency(x, y, weight):
            with torch.no_grad():
                reference = F.log_softmax(F.linear(x, weight).float(), -1)
            predicted = F.log_softmax(F.linear(y, weight).float(), -1)
            return F.kl_div(predicted, reference, log_target=True, reduction="sum")

        if torch.is_grad_enabled():
            kl = kl + checkpoint(consistency, original, recovered, head.weight, use_reentrant=False)
        else:
            kl = kl + consistency(original, recovered, head.weight)
    kl = kl / len(indices)
    total = 0.5 * mse_full + 0.5 * mse_cutoff + beta * kl
    if diagnostics is not None:
        from ncm_lab.monitoring.diagnostics import tensor_statistics

        diagnostics.update(
            elements=int(valid.sum()) * states.shape[-1],
            cutoff_resolution=codec.architecture.resolutions[cutoff],
            normalization=normalization,
            beta=beta,
            head_reconstruction="sampled_cutoff",
            kl_direction="KL(teacher || reconstruction)",
            temperature=1.0,
            statistics={
                "teacher_states": tensor_statistics(states[valid]),
                "encoder_codes": tensor_statistics(fine[valid]),
                "reconstructed_states": tensor_statistics(coarse[valid]),
            }
            if diagnostics.get("statistics_enabled", True)
            else {},
        )
    return total, {
        "mse_full": float(mse_full.detach()),
        "mse_cutoff": float(mse_cutoff.detach()),
        "head_kl": float(kl.detach()),
        "loss": float(total.detach()),
        "cutoff": cutoff,
        "head_positions": len(indices),
    }
