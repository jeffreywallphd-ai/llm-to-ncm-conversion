"""Native embedded hierarchy predicts one eight-token continuation per traversal."""

import torch
from ncm_lab.block8.config import Decoding
from ncm_lab.teacher.model import load_generator


def load_model(settings):
    model = load_generator(
        settings["model"], settings["architecture"], settings["device"], getattr(torch, settings["dtype"])
    )
    model.set_phase("full")
    return model


@torch.no_grad()
def prefill(model, text_ids, cache=None):
    if model.training or text_ids.ndim != 2 or text_ids.shape[0] != 1 or text_ids.shape[1] < 1:
        raise ValueError("hierarchical text cache requires nonempty unpadded inference tokens")
    return model.backbone.model(input_ids=text_ids, past_key_values=cache, use_cache=True).past_key_values


@torch.inference_mode()
def generate(model, prompt_ids, decoding, eos_ids, *, max_context_tokens, suppressed_ids=(), charge=None):
    policy = Decoding(**decoding)
    if (
        model.architecture.canvas != 8
        or not prompt_ids
        or len(prompt_ids) + policy.max_new_tokens > max_context_tokens
    ):
        raise ValueError("hierarchical prompt/output exceeds the explicit eight-token context contract")
    if not eos_ids or any(
        type(i) is not int or not 0 <= i < model.backbone.config.vocab_size for i in eos_ids
    ):
        raise ValueError("valid explicit EOS IDs required")
    model.eval()
    device = next(model.parameters()).device
    context = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    calls, query_positions, layer_positions, attended = 0, 0, 0, 0

    def work(inputs, visible, active, actual_layer_positions=None):
        nonlocal calls, query_positions, layer_positions, attended
        if charge:
            charge(source=visible, canvas=active, answer=active, steps=1)
        calls += 1
        query_positions += inputs
        layer_positions += (
            actual_layer_positions
            if actual_layer_positions is not None
            else inputs * model.architecture.blocks
        )
        attended += visible

    work(len(prompt_ids), len(prompt_ids), 0)
    cache = prefill(model, context)
    blocked = sorted(set(suppressed_ids) - set(eos_ids))
    if any(type(i) is not int or not 0 <= i < model.backbone.config.vocab_size for i in blocked):
        raise ValueError("invalid suppressed token")
    answer, blocks, reason = [], 0, "output_token_cap"
    for _ in range(min(policy.max_blocks, policy.max_new_tokens // 8)):
        # Charge the declared traversal before allocation; refinements remain physical native operations.
        work(
            8,
            context.shape[1] + 8,
            8,
            sum(
                [model.architecture.initial_queries] * model.architecture.early_blocks
                + [
                    model.architecture.resolutions[max(0, 2 * i - 1)]
                    if i
                    else model.architecture.initial_queries
                    for i in range(model.architecture.hierarchy_blocks)
                ]
                + [8] * model.architecture.late_blocks
            ),
        )
        output = model.forward_cached_context(context, torch.ones_like(context, dtype=torch.bool), cache)
        logits = model.backbone.lm_head(output.answer_states).float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite hierarchical generation logits")
        if blocked:
            logits[..., blocked] = -torch.inf
        tokens = logits.argmax(-1)[0].tolist()
        if any(t in eos_ids for t in tokens):
            tokens = tokens[: next(i for i, t in enumerate(tokens) if t in eos_ids) + 1]
            reason = "eos"
        answer.extend(tokens)
        blocks += 1
        committed = torch.tensor([tokens], device=device, dtype=torch.long)
        work(len(tokens), context.shape[1] + len(tokens), 0)
        cache = prefill(model, committed, cache)
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
        processed_input_positions=query_positions,
        actual_backbone_query_positions=layer_positions,
        attended_context_positions=attended,
        hierarchy_updates=blocks * len(model.architecture.resolutions),
        cache_positions=cache.get_seq_length(),
        architecture=model.architecture.to_dict(),
        verification_or_repair=False,
        generation_policy="single_traversal_native_hierarchy_then_causal_text_commit_v1",
    )
