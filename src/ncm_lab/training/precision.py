"""Adjacent-value BF16 stochastic rounding and bounded device reductions.

Independent implementation of the adjacent-probability rule, not a port of an
external optimizer. Signed zero is preserved for exact inputs; subnormal values
between zero and a BF16 neighbor may round to signed zero of the input's sign.
"""

from dataclasses import dataclass
import torch

POLICY = "fp32_moments_bf16_stochastic_v1"
OPTIMIZER_SCHEMA = "fp32_adamw_bf16_sr_v2"


def bf16_supported(values):
    return torch.isfinite(values).all() & (values.abs() <= torch.finfo(torch.bfloat16).max).all()


@torch.no_grad()
def stochastic_bf16(values, generator, *, validated=False):
    if values.dtype != torch.float32:
        raise ValueError("stochastic write-back requires FP32 completed arithmetic")
    if not validated and not bool(bf16_supported(values)):
        raise ValueError("nonfinite/overflow value outside finite BF16 write-back range")
    nearest = values.to(torch.bfloat16)
    rounded = nearest.float()
    exact = rounded == values
    below = torch.nextafter(nearest, torch.full_like(nearest, -torch.inf)).float()
    above = torch.nextafter(nearest, torch.full_like(nearest, torch.inf)).float()
    lower = torch.where(rounded > values, below, rounded)
    upper = torch.where(rounded < values, above, rounded)
    gap = torch.where(exact, torch.ones_like(values), upper - lower)
    probability = (values - lower) / gap
    draws = torch.rand(values.shape, device=values.device, dtype=torch.float32, generator=generator)
    result = torch.where(exact, rounded, torch.where(draws < probability, upper, lower))
    return result.to(torch.bfloat16)


def eligible(parameters):
    return [p for p in parameters if p.requires_grad and p.grad is not None]


@dataclass
class GradientReport:
    parameters: list
    norm: float
    versions: tuple

    def validate(self, parameters):
        active = eligible(parameters)
        if tuple((id(p), id(p.grad), p.grad._version) for p in active) != self.versions:
            raise ValueError("gradient validation is stale or belongs to a different optimizer")

    def clip_(self, limit):
        if limit <= 0:
            raise ValueError("positive gradient clipping limit required")
        if self.norm > limit:
            for p in self.parameters:
                p.grad.mul_(limit / (self.norm + 1e-12))
        self.versions = tuple((id(p), id(p.grad), p.grad._version) for p in self.parameters)


@torch.no_grad()
def check_gradients(parameters, tile_elements=1048576):
    active = eligible(dict.fromkeys(parameters))
    if not active:
        return GradientReport([], 0.0, ())
    device = active[0].device
    squares = torch.zeros((), device=device, dtype=torch.float64)
    finite = torch.ones((), device=device, dtype=torch.bool)
    for p in active:
        if p.device != device or p.grad.device != device:
            raise ValueError("initial optimizer requires one parameter/gradient device")
        flat = p.grad.detach().view(-1)
        for start in range(0, flat.numel(), tile_elements):
            tile = flat[start : start + tile_elements]
            finite = finite & torch.isfinite(tile).all()
            squares = squares + torch.linalg.vector_norm(tile, dtype=torch.float64).square()
    norm = squares.sqrt()
    # One small aggregate transfer, never one Python scalar per tile.
    value, valid = torch.stack((norm, (finite & torch.isfinite(norm)).double())).tolist()
    if not valid:
        raise ValueError("nonfinite native gradient/norm; update rejected before any state mutation")
    return GradientReport(active, value, tuple((id(p), id(p.grad), p.grad._version) for p in active))
