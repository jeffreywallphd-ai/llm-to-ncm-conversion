"""One shared continuous fine encoder and position/level-conditioned decoder."""

import torch
from torch import nn
from ncm_lab.embedded.geometry import masked_pool, resize, position_features

SCHEMA = "continuous_multiscale_pyramid_v1"


class ContinuousCodec(nn.Module):
    def __init__(self, hidden_size, architecture):
        super().__init__()
        self.architecture = architecture
        width = architecture.latent_width
        self.spec = {"schema": SCHEMA, "hidden_size": hidden_size, "architecture": architecture.to_dict()}
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )
        self.decoder = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, hidden_size)
        )
        self.position = nn.Linear(4, width)
        self.levels = nn.Parameter(torch.empty(len(architecture.resolutions), width))
        nn.init.normal_(self.levels, std=0.02)

    def fine_codes(self, states):
        return self.encoder(states)

    def targets(self, fine, valid):
        previous = fine.new_zeros(fine.shape[0], self.architecture.initial_queries, fine.shape[-1])
        cumulative = []
        residuals = []
        masks = []
        for slots in self.architecture.resolutions:
            current, mask = masked_pool(fine, valid, slots)
            residuals.append(current - resize(previous, slots, self.architecture.canvas))
            cumulative.append(current)
            masks.append(mask)
            previous = current
        return tuple(cumulative), tuple(residuals), tuple(masks)

    def reconstruct(self, code, level):
        full = resize(code, self.architecture.canvas, self.architecture.canvas)
        positions = position_features(
            self.architecture.canvas,
            self.architecture.canvas,
            level,
            len(self.architecture.resolutions),
            torch.zeros(code.shape[0], device=code.device),
            code.dtype,
        )
        return self.decoder(full + self.position(positions) + self.levels[level])
