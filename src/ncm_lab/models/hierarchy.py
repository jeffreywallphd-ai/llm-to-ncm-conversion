import torch
from torch import nn
from ncm_lab.models.intervals import partition, validate_schedule
from ncm_lab.models.upstream import freeze, validate_compatibility


def prefix_visibility(prefix_positions, prefix_documents, anchors, anchor_documents):
    return ((prefix_positions < anchors[:, None]) & (prefix_positions >= 0)
            & (prefix_documents == anchor_documents[:, None]))


class Expansion(nn.Module):
    def __init__(self, parent, child, width, learned=True):
        super().__init__()
        self.register_buffer("parent_map", torch.tensor(parent.parents(child), dtype=torch.long))
        self.offset = nn.Parameter(torch.zeros(len(child.groups), width), requires_grad=learned)

    def forward(self, states):
        return torch.cat([states[:, :1], states[:, 1:].index_select(1, self.parent_map) + self.offset], dim=1)


class ProgressiveDrafter(nn.Module):
    """Calls the locked upstream layers at physically different query lengths.

    Each batch row is a separate anchor block. Features may contain multiple packed
    documents; visibility uses document identity AND that block's anchor cutoff.
    """
    def __init__(self, backbone, target, schedule=(4, 4, 8, 15, 15), skip=True, learned_expansion=True, rank=64,
                 initial_span_learned=None):
        super().__init__()
        validate_compatibility(target, backbone)
        if len(schedule) != len(backbone.layers):
            raise ValueError("one resolution per existing drafter layer is required")
        self.parts = validate_schedule(schedule, backbone.block_size - 1)
        self.backbone = backbone
        self.embeddings = freeze(target.get_input_embeddings())
        self.head = freeze(target.get_output_embeddings())
        self.schedule, self.horizon = tuple(schedule), backbone.block_size - 1
        self.sample_from_anchor = False
        self.expansions = nn.ModuleDict()
        width = backbone.config.hidden_size
        for i in range(1, len(self.parts)):
            if self.schedule[i] != self.schedule[i - 1]:
                self.expansions[str(i)] = Expansion(self.parts[i - 1], self.parts[i], width, learned_expansion)
        self.initial_span = nn.Parameter(torch.zeros(self.schedule[0], width),
            requires_grad=learned_expansion if initial_span_learned is None else initial_span_learned)
        self.skip_enabled = skip
        if skip:
            if len(schedule) < 3:
                raise ValueError("skip requires at least three layers")
            self.skip_down = nn.Linear(width, min(rank, width), bias=False)
            self.skip_up = nn.Linear(min(rank, width), width, bias=False)
            self.skip_gate = nn.Parameter(torch.zeros(()))
            leaf = partition(self.horizon, self.horizon)
            self.register_buffer("skip_map", torch.tensor(self.parts[1].parents(leaf)))
        self.stage_lengths = []
        self.trace = {}

    def forward(self, target_features, prefix_positions, prefix_documents, anchors, anchor_tokens,
                anchor_documents, retain_trace=False):
        b = anchors.shape[0]
        if anchors.ndim != 1 or (anchors < 0).any():
            raise ValueError("anchors must be nonnegative one-dimensional positions")
        allowed = prefix_visibility(prefix_positions, prefix_documents, anchors, anchor_documents)
        # Mask before projection too: disallowed NaNs must never contaminate logits.
        features = torch.where(allowed[..., None], target_features.detach(), 0)
        context = self.backbone.hidden_norm(self.backbone.fc(features))
        if getattr(self, "context_policy", "all_confirmed_features") == "mean_confirmed_features":
            context = (context * allowed[..., None]).sum(1, keepdim=True) / allowed.sum(1).clamp_min(1)[:,None,None]
            prefix_positions = torch.where(allowed, prefix_positions, -1).max(1, keepdim=True).values
            allowed = allowed.any(1,keepdim=True)
        anchor = self.embeddings(anchor_tokens).unsqueeze(1)
        mask_ids = torch.full((b, self.schedule[0]), self.backbone.mask_token_id,
                              dtype=torch.long, device=anchor_tokens.device)
        hidden = torch.cat([anchor, self.embeddings(mask_ids) + self.initial_span], dim=1)
        self.stage_lengths, self.trace = [], {}
        coarse = None
        if hasattr(self, "latent"):
            self.latent.predictions = {}
        for i, (layer, part) in enumerate(zip(self.backbone.layers, self.parts)):
            if str(i) in self.expansions:
                hidden = self.expansions[str(i)](hidden)
                if retain_trace:
                    hidden.retain_grad()
                    self.trace[f"expansion_{i}"] = hidden
            if self.skip_enabled and i == len(self.parts) - 1:
                lifted = torch.cat([coarse[:, :1], coarse[:, 1:].index_select(1, self.skip_map)], dim=1)
                hidden = hidden + self.skip_gate.tanh() * self.skip_up(self.skip_down(lifted))
            if hasattr(self, "latent"):
                hidden = self.latent.condition(hidden, i, self.schedule[i])
            offsets = torch.tensor([0, *part.representatives], device=anchors.device)
            query_positions = anchors[:, None] + offsets[None, :]
            positions = torch.cat([prefix_positions.clamp_min(0), query_positions], dim=1)
            mask = torch.cat([allowed[:, None, :].expand(b, hidden.shape[1], -1),
                              torch.ones(b, hidden.shape[1], hidden.shape[1], device=hidden.device, dtype=torch.bool)], -1)
            hidden = layer(hidden_states=hidden, target_hidden=context, attention_mask=mask[:, None],
                           position_ids=positions, position_embeddings=self.backbone.rotary_emb(hidden, positions),
                           use_cache=False)
            self.stage_lengths.append(hidden.shape[1])
            if i == 1:
                coarse = hidden
                if retain_trace:
                    coarse.retain_grad()
                    self.trace["coarse"] = coarse
            if hasattr(self, "latent") and (self.training or self.latent.inject or self.latent.diagnostic):
                self.latent.predict(hidden, i, self.schedule[i])
        fine = self.backbone.norm(hidden[:, 1:])
        # Frozen head remains in the graph. Never wrap this operation in no_grad.
        return self.backbone.compute_logits(fine, self.head)

    def parameter_counts(self):
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": sum(p.numel() for p in self.parameters())}
