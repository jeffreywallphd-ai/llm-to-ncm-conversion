"""Fixed-canvas geometry. Resize uses O(source+destination) overlap edges."""

from functools import lru_cache
import torch


def intervals(canvas, slots, device=None):
    if not 1 <= slots <= canvas:
        raise ValueError("nonempty canvas intervals required")
    edges = torch.arange(slots + 1, device=device) * canvas // slots
    return edges[:-1], edges[1:]


@lru_cache(maxsize=256)
def _overlap(canvas, source, destination):
    left_s, right_s = intervals(canvas, source)
    left_d, right_d = intervals(canvas, destination)
    i = j = 0
    sources = []
    destinations = []
    weights = []
    while i < source and j < destination:
        overlap = min(int(right_s[i]), int(right_d[j])) - max(int(left_s[i]), int(left_d[j]))
        if overlap > 0:
            sources.append(i)
            destinations.append(j)
            weights.append(overlap / int(right_d[j] - left_d[j]))
        if right_s[i] < right_d[j]:
            i += 1
        elif right_s[i] > right_d[j]:
            j += 1
        else:
            i += 1
            j += 1
    return torch.tensor(sources), torch.tensor(destinations), torch.tensor(weights)


def resize(value, slots, canvas):
    if value.shape[1] == slots:
        return value
    source, destination, weights = _overlap(canvas, value.shape[1], slots)
    selected = (
        value.index_select(1, source.to(value.device)) * weights.to(value.device, value.dtype)[None, :, None]
    )
    return value.new_zeros(value.shape[0], slots, value.shape[2]).index_add(
        1, destination.to(value.device), selected
    )


def masked_pool(value, valid, slots):
    canvas = value.shape[1]
    left, right = intervals(canvas, slots, value.device)
    sums = torch.cat(
        [
            value.new_zeros(value.shape[0], 1, value.shape[2], dtype=torch.float32),
            (value.float() * valid[:, :, None]).cumsum(1),
        ],
        1,
    )
    counts = torch.cat([valid.new_zeros(valid.shape[0], 1, dtype=torch.long), valid.long().cumsum(1)], 1)
    count = counts[:, right] - counts[:, left]
    result = (sums[:, right] - sums[:, left]) / count.clamp_min(1)[:, :, None]
    return result.to(value.dtype), count.gt(0)


def position_features(canvas, slots, level, levels, prompt_lengths, dtype):
    left, right = intervals(canvas, slots, prompt_lengths.device)
    center = (left + right - 1) // 2
    feature = torch.stack(
        [
            center.float() / canvas,
            (right - left).float() / canvas,
            torch.full_like(center, level / max(1, levels - 1), dtype=torch.float32),
        ],
        -1,
    )
    feature = feature[None].expand(prompt_lengths.shape[0], -1, -1)
    return torch.cat([feature, prompt_lengths[:, None, None].expand(-1, slots, 1).float() / 40960], -1).to(
        dtype
    )


def attention_geometry(prompt_mask, slots, canvas, dtype):
    batch, prompt = prompt_mask.shape
    prompt_mask = prompt_mask.bool()
    length = prompt_mask.sum(1)
    if bool(length.eq(0).any()):
        raise ValueError("every prompt needs a valid token")
    left, right = intervals(canvas, slots, prompt_mask.device)
    prompt_positions = (prompt_mask.long().cumsum(1) - 1).clamp_min(0)
    positions = torch.cat([prompt_positions, length[:, None] + ((left + right - 1) // 2)[None]], 1)
    allowed = torch.zeros(batch, prompt + slots, prompt + slots, device=prompt_mask.device, dtype=torch.bool)
    causal = torch.ones(prompt, prompt, device=prompt_mask.device, dtype=torch.bool).tril()
    allowed[:, :prompt, :prompt] = causal[None] & prompt_mask[:, :, None] & prompt_mask[:, None, :]
    allowed[:, prompt:, :prompt] = prompt_mask[:, None, :]
    allowed[:, prompt:, prompt:] = True
    mask = torch.zeros(
        batch, 1, prompt + slots, prompt + slots, device=prompt_mask.device, dtype=dtype
    ).masked_fill(~allowed[:, None], float("-inf"))
    return mask, positions
