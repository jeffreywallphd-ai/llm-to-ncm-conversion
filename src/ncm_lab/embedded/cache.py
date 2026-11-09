"""Read-only causal text KV access for variable-resolution native answer queries."""

import torch
from torch.utils.checkpoint import checkpoint


class ReadOnlyPrefix:
    def __init__(self, cache):
        self.cache = cache

    def update(self, key, value, layer_idx, *args, **kwargs):
        layer = self.cache.layers[layer_idx]
        return torch.cat((layer.keys, key), dim=-2), torch.cat((layer.values, value), dim=-2)

    def get_seq_length(self, *args):
        return self.cache.get_seq_length(*args)


def causal_text_states(backbone, text_ids, *, checkpoint_layers=False):
    """Ordinary causal Qwen states, with per-layer activation recomputation."""
    hidden = backbone.model.embed_tokens(text_ids)
    length = text_ids.shape[1]
    positions = torch.arange(length, device=text_ids.device)[None]
    allowed = torch.ones(length, length, device=text_ids.device, dtype=torch.bool).tril()
    mask = hidden.new_zeros(1, 1, length, length).masked_fill(~allowed[None, None], float("-inf"))
    rotary = backbone.model.rotary_emb(hidden, positions)
    for layer in backbone.model.layers:

        def block(value, module=layer):
            return module(
                value,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=rotary,
                use_cache=False,
                is_causal=False,
            )

        hidden = checkpoint(block, hidden, use_reentrant=False) if checkpoint_layers else block(hidden)
    return backbone.model.norm(hidden)
