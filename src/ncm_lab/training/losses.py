import torch
from torch.nn import functional as F


def final_token_loss(logits, labels, valid=None, decay_gamma=7.0):
    """Final-token CE, excluding anchor; optional inspected DCore position decay.

    L = sum_valid exp(-j/gamma) CE(q_j, y_j) / N_valid, j=0..H-1.
    Unlike the upstream batch reduction this explicitly normalizes by supervised
    positions. Apply the identical objective to flat and hierarchical controls.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("labels must align exactly with unknown future logits")
    valid = labels.ne(-100) if valid is None else valid.bool() & labels.ne(-100)
    if not valid.any():
        raise ValueError("no valid supervised future positions")
    per_token = F.cross_entropy(logits.flatten(0,-2), labels.flatten(), reduction="none", ignore_index=-100).view_as(labels)
    weights = torch.ones(labels.shape[-1], device=logits.device)
    if decay_gamma is not None:
        if decay_gamma <= 0:
            raise ValueError("decay_gamma must be positive or None")
        weights = torch.exp(-torch.arange(labels.shape[-1],device=logits.device) / decay_gamma)
    return (per_token * weights * valid).sum() / valid.sum()
