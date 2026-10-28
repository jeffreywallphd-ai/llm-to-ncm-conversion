"""Continuous independent-level codecs. No target or vocabulary weights are owned here."""

import torch
from torch import nn
from ncm_lab.models.intervals import partition


def pool(states, count, valid=None):
    """Ordered unequal intervals; label masks are optional, never used by students."""
    groups = partition(count).groups
    if states.shape[1] != 15:
        raise ValueError("pool expects fifteen ordered future positions")
    if valid is None:
        valid = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
    values, masks = [], []
    for group in groups:
        ix = torch.tensor([i - 1 for i in group], device=states.device)
        mask = valid.index_select(1, ix)
        value = states.index_select(1, ix).masked_fill(~mask[..., None], 0)
        values.append(value.sum(1) / mask.sum(1).clamp_min(1)[:, None])
        masks.append(mask.any(1))
    return torch.stack(values, 1), torch.stack(masks, 1)


def lift(states, source, destination=15):
    indices = torch.tensor(partition(source).parents(partition(destination)), device=states.device)
    return states.index_select(1, indices)


class Upsampler(nn.Module):
    def __init__(self, source, destination, width):
        super().__init__()
        self.source, self.destination = source, destination
        self.offset = nn.Parameter(torch.zeros(destination, width))

    def forward(self, code):
        return lift(code, self.source, self.destination) + self.offset


class MultiscaleCodec(nn.Module):
    def __init__(self, hidden_size, width=256, kind="nested", levels=(4, 8, 15), architecture_version=None):
        super().__init__()
        if kind not in {"nested", "linear", "independent", "pooling", "channel"}:
            raise ValueError("unknown codec kind")
        if not set(levels) <= {4, 8, 15} or not levels:
            raise ValueError("invalid codec levels")
        architecture_version = architecture_version or (3 if kind == "independent" else 2)
        if architecture_version not in {1, 2, 3}:
            raise ValueError("unsupported codec architecture version")
        self.spec = {
            "hidden_size": hidden_size,
            "width": width,
            "kind": kind,
            "levels": list(levels),
            "architecture_version": architecture_version,
        }
        self.levels = tuple(levels)

        def projection(a, b):
            return (
                nn.Linear(a, b)
                if kind in {"linear", "pooling"}
                else nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Linear(b, b))
            )

        nested = kind == "nested" or kind == "linear" and architecture_version >= 2
        self.encoder = (
            projection(hidden_size, width) if nested or architecture_version == 1 else nn.Identity()
        )
        self.coarsen8 = projection(width, width) if nested or architecture_version == 1 else nn.Identity()
        self.coarsen4 = projection(width, width) if nested or architecture_version == 1 else nn.Identity()
        self.independent = nn.ModuleDict(
            {str(n): projection(hidden_size, width) for n in levels}
            if not nested or architecture_version == 1
            else {}
        )
        self.up4 = Upsampler(4, 8, width) if 4 in levels or architecture_version == 1 else nn.Identity()
        self.up8 = (
            Upsampler(8, 15, width) if set(levels) & {4, 8} or architecture_version == 1 else nn.Identity()
        )
        self.output = projection(width, hidden_size)
        if kind == "pooling":
            self.requires_grad_(False)

    def encode(self, states, valid):
        if (
            self.spec["kind"] == "nested"
            or self.spec["kind"] == "linear"
            and self.spec["architecture_version"] >= 2
        ):
            z15 = self.encoder(states)
            z8, m8 = pool(z15, 8, valid)
            z8 = self.coarsen8(z8)
            # Pool children with valid-token weights, including unequal terminal groups.
            z4, _ = pool(lift(z8, 8), 4, valid)
            z4 = self.coarsen4(z4)
            codes = {4: z4, 8: z8, 15: z15}
        else:
            if self.spec["kind"] == "independent" and self.spec["architecture_version"] >= 3:
                # Independent F_l transforms every original position before P_l;
                # no branch consumes another branch's representation. G_l is identity.
                codes = {n: pool(self.independent[str(n)](states), n, valid)[0] for n in self.levels}
            else:
                codes = {n: self.independent[str(n)](pool(states, n, valid)[0]) for n in self.levels}
        return {n: codes[n] for n in self.levels}

    def reconstruct(self, code, level):
        if level not in self.levels or code.shape[1] != level:
            raise ValueError("one standalone code at its declared level is required")
        if level == 4:
            code = self.up4(code)
        if level in {4, 8}:
            code = self.up8(code)
        return self.output(code)

    def rates(self):
        element_size = next(self.parameters()).element_size()
        counts = {str(n): n * self.spec["width"] for n in self.levels}
        return {
            "standalone_scalars": counts,
            "all_levels_scalars": sum(counts.values()),
            "standalone_bytes": {n: r * element_size for n, r in counts.items()},
            "all_levels_bytes": sum(counts.values()) * element_size,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
