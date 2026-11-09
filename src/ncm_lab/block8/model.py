"""Qwen text context is causal; only the active block attends bidirectionally."""

from dataclasses import asdict
import math
import torch
from torch import nn
from ncm_lab.block8.config import Architecture
from ncm_lab.teacher.model import load_original


class ReadOnlyPrefix:
    """Ephemeral active keys/values never mutate the committed text cache."""

    def __init__(self, cache):
        self.cache = cache

    def update(self, key, value, layer_idx, *args, **kwargs):
        layer = self.cache.layers[layer_idx]
        return torch.cat((layer.keys, key), dim=-2), torch.cat((layer.values, value), dim=-2)

    def get_seq_length(self, *args):
        return self.cache.get_seq_length(*args)


def active_geometry(context, block, dtype, device):
    total = context + block
    allowed = torch.ones(total, total, dtype=torch.bool, device=device).tril()
    allowed[:context, context:] = False
    allowed[context:, :] = True
    return torch.zeros(1, 1, total, total, dtype=dtype, device=device).masked_fill(
        ~allowed[None, None], float("-inf")
    )


class BlockDecoder(nn.Module):
    def __init__(self, backbone, architecture=None):
        super().__init__()
        self.architecture = architecture or Architecture()
        self.backbone = backbone
        if backbone.config.model_type != "qwen3" or backbone.config.tie_word_embeddings:
            raise ValueError("block8 requires untied Qwen3 input/output weights")
        if any(t != "full_attention" for t in backbone.config.layer_types):
            raise ValueError("block8 currently requires full attention in every Qwen layer")
        backbone.config._attn_implementation = "sdpa"
        backbone.config.use_cache = False
        pad = backbone.config.pad_token_id or 0
        self.mask_embedding = nn.Parameter(backbone.model.embed_tokens.weight[pad].detach().clone())

    def _ids(self, ids, *, masked=False):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype != torch.long:
            raise ValueError("block8 uses one nonempty unpadded token sequence per microbatch")
        if bool(((ids < (-1 if masked else 0)) | (ids >= self.backbone.config.vocab_size)).any()):
            raise ValueError("invalid block8 token ID")

    def _embedded(self, ids):
        raw = self.backbone.model.embed_tokens(ids.clamp_min(0))
        return torch.where(ids.eq(-1)[:, :, None], self.mask_embedding[None, None], raw)

    def denoise(self, context_ids, block_input_ids, *, prefix_cache=None):
        """Explicit denoising inputs only; labels, true lengths and validity masks are loss-only."""
        self._ids(context_ids)
        self._ids(block_input_ids, masked=True)
        size = block_input_ids.shape[1]
        if size not in {1, 2, 4, 8}:
            raise ValueError("active size must be a declared 1/2/4/8 curriculum block")
        context = context_ids.shape[1]
        embedded = self._embedded(block_input_ids)
        if prefix_cache is None:
            embedded = torch.cat((self.backbone.model.embed_tokens(context_ids), embedded), 1)
            mask = active_geometry(context, size, embedded.dtype, embedded.device)
            positions = torch.arange(context + size, device=embedded.device)[None]
            cache = None
        else:
            if self.training or torch.is_grad_enabled() or prefix_cache.get_seq_length() != context:
                raise ValueError("cached denoising is inference-only with an exact committed prefix")
            mask = embedded.new_zeros(1, 1, size, context + size)
            positions = torch.arange(context, context + size, device=embedded.device)[None]
            cache = ReadOnlyPrefix(prefix_cache)
        states = self.backbone.model(
            inputs_embeds=embedded,
            position_ids=positions,
            attention_mask={"full_attention": mask},
            past_key_values=cache,
            use_cache=False,
        ).last_hidden_state
        return states[:, -size:]

    def causal_states(self, text_ids):
        self._ids(text_ids)
        return self.backbone.model(input_ids=text_ids, use_cache=False).last_hidden_state

    @torch.no_grad()
    def prefill(self, text_ids, cache=None):
        self._ids(text_ids)
        if self.training:
            raise ValueError("committed caches are inference-only")
        return self.backbone.model(input_ids=text_ids, past_key_values=cache, use_cache=True).past_key_values


def load_model(settings):
    model = BlockDecoder(
        load_original(settings["model"], settings["device"], getattr(torch, settings["dtype"]), frozen=False),
        Architecture(**settings["architecture"]),
    )
    if settings["training"]["activation_checkpointing"]:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


@torch.inference_mode()
def generate(model, prompt_ids, decoding, eos_ids, *, max_context_tokens, suppressed_ids=(), charge=None):
    """Greedy confidence-ordered unmasking; a committed block is never AR-repaired."""
    from ncm_lab.block8.config import Decoding

    policy = Decoding(**decoding)
    if not prompt_ids or len(prompt_ids) + policy.max_new_tokens > max_context_tokens:
        raise ValueError("prompt/output exceeds explicit context capacity")
    if not eos_ids or any(
        type(i) is not int or not 0 <= i < model.backbone.config.vocab_size for i in eos_ids
    ):
        raise ValueError("valid explicit EOS IDs required")
    model.eval()
    device = model.mask_embedding.device
    context = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    calls, positions, attended = 0, 0, 0

    def work(inputs, visible, active):
        nonlocal calls, positions, attended
        if charge:
            charge(source=visible, canvas=active, answer=active, steps=1)
        calls += 1
        positions += inputs
        attended += visible

    work(context.shape[1], context.shape[1], 0)
    cache = model.prefill(context)
    answer, blocks, reason = [], 0, "output_token_cap"
    blocked = sorted(set(suppressed_ids) - set(eos_ids))
    if any(type(i) is not int or not 0 <= i < model.backbone.config.vocab_size for i in blocked):
        raise ValueError("invalid suppressed token")
    for _ in range(min(policy.max_blocks, policy.max_new_tokens // 8)):
        active = torch.full((1, 8), -1, dtype=torch.long, device=device)
        for step in range(policy.refinement_passes):
            work(8, context.shape[1] + 8, 8)
            states = model.denoise(context, active, prefix_cache=cache)
            logits = model.backbone.lm_head(states).float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite block8 generation logits")
            if blocked:
                logits[..., blocked] = -torch.inf
            probability = logits.softmax(-1)
            confidence, proposed = probability.max(-1)
            confidence = confidence.masked_fill(active.ne(-1), -torch.inf)
            remaining = int(active.eq(-1).sum())
            count = math.ceil(remaining / (policy.refinement_passes - step))
            # Stable positional tie breaking is deterministic on CPU and CUDA.
            indices = torch.argsort(confidence[0], descending=True, stable=True)[:count]
            active[0, indices] = proposed[0, indices]
        tokens = active[0].tolist()
        if any(t in eos_ids for t in tokens):
            tokens = tokens[: next(i for i, t in enumerate(tokens) if t in eos_ids) + 1]
            reason = "eos"
        answer.extend(tokens)
        blocks += 1
        # Recompute FINAL token embeddings under strict causality, never cache masked states.
        committed = torch.tensor([tokens], device=device, dtype=torch.long)
        work(len(tokens), context.shape[1] + len(tokens), 0)
        cache = model.prefill(committed, cache)
        context = torch.cat((context, committed), 1)
        if reason == "eos":
            break
    else:
        if blocks * 8 < policy.max_new_tokens:
            reason = "block_cap"
    return dict(
        token_ids=answer,
        stop_reason=reason,
        eos_genuine=reason == "eos",
        censored=reason != "eos",
        blocks=blocks,
        backbone_calls=calls,
        processed_input_positions=positions,
        attended_context_positions=attended,
        cache_positions=cache.get_seq_length(),
        architecture=asdict(model.architecture),
        verification_or_repair=False,
    )
