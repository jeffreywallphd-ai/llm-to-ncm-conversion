"""Causal heads and bridges; deliberately owns no future-state encoder."""
import copy
import torch
from torch import nn
from ncm_lab.models.codecs import Upsampler, pool, lift


class LatentAdapters(nn.Module):
    def __init__(self, hidden_size, width, codec=None, inject=True, single=False, train_upsamplers=False):
        super().__init__()
        self.inject, self.single = inject, single
        self.levels = (8,) if single else (4, 8, 15)
        self.predictors = nn.ModuleDict({str(n): nn.Linear(hidden_size, width) for n in self.levels})
        self.bridges = nn.ModuleDict({str(n): nn.Linear(width, hidden_size, bias=False) for n in self.levels})
        self.gates = nn.ParameterDict({str(n): nn.Parameter(torch.zeros(())) for n in self.levels})
        self.up4 = copy.deepcopy(codec.up4) if codec else Upsampler(4, 8, width)
        self.up8 = copy.deepcopy(codec.up8) if codec else Upsampler(8, 15, width)
        self.up4.requires_grad_(train_upsamplers)
        self.up8.requires_grad_(train_upsamplers)
        self.predictions = {}
        self.diagnostic = False
        self.disabled_levels = set()
        self.ablated_levels = set()

    def predict(self, hidden, layer_index, resolution):
        level = {1: 4, 2: 8, 3: 15}.get(layer_index)
        if level not in self.levels:
            return
        future = hidden[:, 1:]
        if resolution != level:
            if resolution != 15:
                raise ValueError("latent head boundary has incompatible future resolution")
            future, _ = pool(future, level)  # fixed inference-visible pooling, never a gold mask
        predicted = self.predictors[str(level)](future)
        if level == 8 and not self.single:
            predicted = predicted + self.up4(self.predictions[4])
        if level == 15 and not self.single:
            predicted = predicted + self.up8(self.predictions[8])
        self.predictions[level] = torch.zeros_like(predicted) if level in self.ablated_levels else predicted

    def condition(self, hidden, layer_index, resolution):
        level = {2: 4, 3: 8, 4: 15}.get(layer_index)
        if not self.inject or level not in self.predictions or level in self.disabled_levels:
            return hidden
        code = self.predictions[level]
        if level == 4:
            code = self.up4(code)
            if resolution == 15:
                code = lift(code, 8)
        elif level == 8:
            code = self.up8(code)
        contribution = self.gates[str(level)].tanh() * self.bridges[str(level)](code)
        return torch.cat([hidden[:, :1], hidden[:, 1:] + contribution], 1)
