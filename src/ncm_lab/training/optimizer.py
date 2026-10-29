"""Full fine tuning: BF16/FP32 parameters and explicit FP32 Adam moments, no master copy."""

import math
import torch


class TiledAdamW:
    def __init__(self, named_parameters, lr=1e-4, new_lr=None, weight_decay=0.01, tile_elements=1048576):
        self.parameters = dict(named_parameters)
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
        self.weight_decay = weight_decay
        self.tile_elements = tile_elements
        if tile_elements <= 0 or lr <= 0 or new_lr is not None and new_lr <= 0:
            raise ValueError("positive optimizer rates/tile budget required")

    def zero_grad(self):
        for p in self.parameters.values():
            p.grad = None

    @torch.no_grad()
    def step(self):
        for group in self.groups:
            for name in group["names"]:
                p = self.parameters[name]
                if p.grad is None:
                    continue
                state = self.state.setdefault(name, {})
                if not state:
                    state.update(
                        step=0,
                        exp_avg=torch.zeros_like(p, dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(p, dtype=torch.float32),
                    )
                state["step"] += 1
                rate = group["lr"] / (1 - 0.9 ** state["step"])
                parameter = p.view(-1)
                gradient = p.grad.view(-1)
                mean = state["exp_avg"].view(-1)
                variance = state["exp_avg_sq"].view(-1)
                for start in range(0, p.numel(), self.tile_elements):
                    sl = slice(start, start + self.tile_elements)
                    g = gradient[sl].float()
                    if not bool(torch.isfinite(g).all()):
                        raise ValueError("nonfinite gradient: " + name)
                    mean[sl].mul_(0.9).add_(g, alpha=0.1)
                    variance[sl].mul_(0.999).addcmul_(g, g, value=0.001)
                    parameter[sl].mul_(1 - group["lr"] * self.weight_decay)
                    denominator = variance[sl].sqrt().div_(math.sqrt(1 - 0.999 ** state["step"])).add_(1e-8)
                    parameter[sl].addcdiv_(mean[sl], denominator, value=-rate)

    def state_dict(self):
        return {
            "schema": "fp32_moment_adamw_v1",
            "groups": self.groups,
            "weight_decay": self.weight_decay,
            "tile_elements": self.tile_elements,
            "state": self.state,
        }
