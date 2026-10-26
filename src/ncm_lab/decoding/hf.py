import torch
from transformers import DynamicCache
from ncm_lab.decoding.reference import DecodeResult


@torch.no_grad()
def cached_greedy(target, propose, input_ids, max_new_tokens, eos=(), stop_sequences=()):
    """Single unpadded request. Actual HF cache rollback, anchor excluded from q.

    `propose(confirmed_ids, anchor_id)` returns [1, horizon] future token IDs.
    The adapter may recompute prefix features, so this remains a correctness path.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("cache reference supports one nonempty unpadded request")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    result = DecodeResult([])
    if max_new_tokens == 0:
        return result
    target.eval()
    cache = DynamicCache(config=target.config)
    out = target(input_ids=input_ids, past_key_values=cache, use_cache=True)
    anchor = out.logits[:, -1].argmax(-1, keepdim=True)
    result.tokens.append(anchor.item())
    confirmed = input_ids

    def stopped():
        return result.tokens[-1] in eos or any(s and result.tokens[-len(s):] == list(s) for s in stop_sequences)

    while len(result.tokens) < max_new_tokens and not stopped():
        q = propose(confirmed, anchor)
        if q.ndim != 2 or q.shape[0] != 1 or q.shape[1] == 0:
            raise ValueError("expected nonempty future-token proposal [1, horizon]")
        q = q[:, :max_new_tokens - len(result.tokens)]
        old_length = cache.get_seq_length()
        verify = torch.cat([anchor, q], dim=1)
        logits = target(input_ids=verify, past_key_values=cache, use_cache=True).logits
        choices = logits.argmax(-1)
        accepted = 0
        for j in range(q.shape[1]):
            if q[0, j] != choices[0, j]:
                break
            result.tokens.append(q[0, j].item())
            accepted += 1
            result.accepted += 1
            if len(result.tokens) >= max_new_tokens or stopped():
                break
        keep = old_length + 1 + accepted
        cache.crop(keep)
        confirmed = torch.cat([confirmed, verify[:, :1 + accepted]], dim=1)
        result.verification_calls += 1
        result.cycles.append({"anchor_position": old_length, "accepted": accepted, "cache_keep_length": keep})
        if len(result.tokens) >= max_new_tokens or stopped():
            break
        anchor = choices[:, accepted:accepted + 1]
        result.tokens.append(anchor.item())
        if accepted == q.shape[1]:
            result.bonus += 1
        else:
            result.corrections += 1
    return result
