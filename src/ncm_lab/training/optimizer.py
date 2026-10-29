"""Tiled FP32 AdamW, FP32 moments and private stochastic BF16 write-back."""

import math
import torch
from ncm_lab.runtime.io import canonical_hash
from ncm_lab.training.precision import (
    POLICY,
    OPTIMIZER_SCHEMA,
    check_gradients,
    bf16_supported,
    stochastic_bf16,
)


class TiledAdamW:
    def __init__(
        self,
        named_parameters,
        lr=1e-4,
        new_lr=None,
        weight_decay=0.01,
        tile_elements=1048576,
        policy=POLICY,
        seed=1729,
    ):
        self.parameters = {}
        seen = set()
        for name, p in named_parameters:
            if id(p) not in seen:
                self.parameters[name] = p
                seen.add(id(p))
        devices = {p.device for p in self.parameters.values()}
        if len(devices) != 1 or any(
            p.dtype not in (torch.float32, torch.bfloat16) for p in self.parameters.values()
        ):
            raise ValueError("native AdamW supports one device with FP32/BF16 parameters")
        if policy != POLICY or type(seed) is not int or seed < 0:
            raise ValueError(
                "explicit supported stochastic BF16 numerical policy and nonnegative seed required"
            )
        if (
            type(tile_elements) is not int
            or tile_elements <= 0
            or min(lr, new_lr or lr) <= 0
            or not all(math.isfinite(x) for x in (lr, new_lr or lr, weight_decay))
            or weight_decay < 0
        ):
            raise ValueError("invalid finite AdamW/tile configuration")
        self.groups = [
            {
                "name": "pretrained",
                "names": [n for n in self.parameters if n.startswith("backbone.")],
                "lr": lr,
            },
            {
                "name": "new",
                "names": [n for n in self.parameters if not n.startswith("backbone.")],
                "lr": new_lr or lr,
            },
        ]
        self.state = {}
        self.weight_decay, self.tile_elements = weight_decay, tile_elements
        self.policy, self.seed = policy, seed
        self.generator = torch.Generator(device=next(iter(devices))).manual_seed(seed)
        self.rounding_draws = 0
        self.last_update_summary = {}
        self.traversal_sha256 = canonical_hash(
            [(n, list(p.shape), str(p.dtype), str(p.device)) for n, p in self.parameters.items()]
        )

    def precision_contract(self):
        return {
            "schema": OPTIMIZER_SCHEMA,
            "policy": self.policy,
            "seed": self.seed,
            "rng_algorithm": "torch.Generator/" + str(self.generator.device),
            "tile_elements": self.tile_elements,
            "traversal_sha256": self.traversal_sha256,
            "moments_dtype": "float32",
            "master_parameters": False,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
        }

    def zero_grad(self):
        for p in self.parameters.values():
            p.grad = None

    def check_gradients(self):
        return check_gradients(self.parameters.values(), self.tile_elements)

    def _tiles(self):
        for group in self.groups:
            for name in group["names"]:
                p = self.parameters[name]
                if not p.requires_grad or p.grad is None:
                    continue
                state = self.state.get(name)
                step = state["step"] + 1 if state else 1
                for start in range(0, p.numel(), self.tile_elements):
                    sl = slice(start, start + self.tile_elements)
                    g = p.grad.view(-1)[sl].float()
                    old_mean = state["exp_avg"].view(-1)[sl] if state else torch.zeros_like(g)
                    old_variance = state["exp_avg_sq"].view(-1)[sl] if state else torch.zeros_like(g)
                    mean = old_mean * 0.9 + g * 0.1
                    variance = old_variance * 0.999 + g.square() * 0.001
                    denominator = variance.sqrt().div_(math.sqrt(1 - 0.999**step)).add_(1e-8)
                    updated = p.view(-1)[sl].float().clone()
                    updated.mul_(1 - group["lr"] * self.weight_decay)
                    updated.addcdiv_(mean, denominator, value=-group["lr"] / (1 - 0.9**step))
                    yield name, p, sl, step, mean, variance, updated

    @torch.no_grad()
    def step(self, gradient_report=None):
        report = gradient_report or self.check_gradients()
        report.validate(self.parameters.values())
        device = next(iter(self.parameters.values())).device
        valid = torch.ones((), device=device, dtype=torch.bool)
        intended = torch.zeros((), device=device, dtype=torch.float64)
        # Two bounded passes validate all prospective arithmetic before any state mutation.
        for _, p, sl, _, mean, variance, updated in self._tiles():
            valid = valid & torch.isfinite(mean).all() & torch.isfinite(variance).all()
            valid = valid & (
                bf16_supported(updated) if p.dtype == torch.bfloat16 else torch.isfinite(updated).all()
            )
            intended = intended + (updated.double() - p.view(-1)[sl].double()).square().sum()
        if not bool(valid):
            raise ValueError("nonfinite/overflow optimizer arithmetic; entire update rejected")
        steps = {}
        stored = torch.zeros_like(intended)
        for name, p, sl, step, mean, variance, updated in self._tiles():
            if name not in self.state:
                self.state[name] = {
                    "step": step - 1,
                    "exp_avg": torch.zeros_like(p, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
                }
            state = self.state[name]
            state["exp_avg"].view(-1)[sl].copy_(mean)
            state["exp_avg_sq"].view(-1)[sl].copy_(variance)
            if p.dtype == torch.bfloat16:
                updated = stochastic_bf16(updated, self.generator, validated=True)
                self.rounding_draws += updated.numel()
            stored = stored + (updated.double() - p.view(-1)[sl].double()).square().sum()
            p.view(-1)[sl].copy_(updated)
            steps[name] = step
        for name, step in steps.items():
            self.state[name]["step"] = step
        intended_value, stored_value = torch.stack((intended.sqrt(), stored.sqrt())).tolist()
        self.last_update_summary = {
            "intended_fp32_l2": intended_value,
            "stored_parameter_l2": stored_value,
            "scalar_elements": sum(p.numel() for p in report.parameters),
            "scope": "last completed optimizer update; all active scalar elements",
        }

    def state_dict(self):
        return {
            "schema": OPTIMIZER_SCHEMA,
            "precision": self.precision_contract(),
            "groups": self.groups,
            "weight_decay": self.weight_decay,
            "tile_elements": self.tile_elements,
            "state": self.state,
            "rounding_draws": self.rounding_draws,
            "rounding_rng": self.generator.get_state(),
        }
