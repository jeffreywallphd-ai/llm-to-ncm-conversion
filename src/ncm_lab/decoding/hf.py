import time
import torch
from transformers import DynamicCache
from ncm_lab.decoding.reference import DecodeResult
from ncm_lab.decoding.policy import GenerationPolicy,guard_decode
from ncm_lab.vendor.dflash.model import extract_context_feature


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class CachedProposer:
    """No target reference, future encoder, labels or response masks."""
    def __init__(self, drafter):
        self.drafter = drafter.eval()
        self.taps = list(drafter.backbone.target_layer_ids)

    def from_features(self, features, confirmed, anchor):
        positions = torch.arange(confirmed.shape[1], device=confirmed.device)[None]
        return self.drafter(features, positions, torch.zeros_like(positions),
            torch.tensor([confirmed.shape[1]], device=confirmed.device), anchor[:, 0],
            torch.zeros(1, dtype=torch.long, device=confirmed.device)).argmax(-1)


@torch.no_grad()
def cached_greedy(target, propose, input_ids, max_new_tokens=None, eos=(), stop_sequences=(), policy=None,guard=None):
    """Single unpadded request. Actual HF cache rollback, anchor excluded from q.

    `propose(confirmed_ids, anchor_id)` returns [1, horizon] future token IDs.
    The adapter may recompute prefix features, so this remains a correctness path.
    """
    decode_started=time.perf_counter()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("cache reference supports one nonempty unpadded request")
    policy = policy or GenerationPolicy(max_new_tokens, tuple(eos), tuple(tuple(s) for s in stop_sequences))
    max_new_tokens = policy.max_new_tokens
    if input_ids.shape[1] + max_new_tokens > target.config.max_position_embeddings:
        raise ValueError("declared context/output limit exceeds target position capacity")
    result = DecodeResult([])
    result.telemetry = {"policy_hash": policy.digest, "target_processed_positions": 0,
                        "prefill_seconds": 0.0, "draft_seconds": 0.0, "verification_seconds": 0.0,
                        "cache_seconds": 0.0, "feature_cache_bytes": 0, "stop_reason": "length"}
    result.telemetry.update(latent_head_seconds=None, bridge_seconds=None, expansion_seconds=None,
        component_missing_reason="included in synchronized draft duration; separate attribution requires profiler",
        gpu_peak_allocated_bytes=None, gpu_memory_missing_reason="CPU execution" if input_ids.device.type == "cpu" else None)
    result.telemetry.update(initial_anchor=None,initial_anchor_count=0)
    if input_ids.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    if max_new_tokens == 0:
        result.telemetry['generation_seconds']=time.perf_counter()-decode_started
        return result
    target.eval()
    incremental = hasattr(propose, "from_features")
    cache = DynamicCache(config=target.config)
    guard_decode(guard,result.tokens,result)
    synchronize(input_ids.device)
    started = time.perf_counter()
    out = target(input_ids=input_ids, past_key_values=cache, use_cache=True, output_hidden_states=incremental)
    synchronize(input_ids.device)
    result.telemetry["prefill_seconds"] = time.perf_counter() - started
    started=time.perf_counter()
    features = extract_context_feature(out.hidden_states, propose.taps).detach() if incremental else None
    feature_buffer = None
    if incremental:
        feature_buffer = features.new_empty((1, input_ids.shape[1]+max_new_tokens, features.shape[-1]))
        feature_buffer[:, :input_ids.shape[1]] = features
        features = feature_buffer[:, :input_ids.shape[1]]
        result.telemetry['feature_cache_bytes']=feature_buffer.numel()*feature_buffer.element_size()
        result.telemetry['feature_cache_allocated_bytes']=result.telemetry['feature_cache_bytes']
        result.telemetry['feature_cache_used_bytes']=features.numel()*features.element_size()
    synchronize(input_ids.device)
    result.telemetry['cache_seconds']+=time.perf_counter()-started
    result.telemetry['target_position_accounting_complete']=incremental
    anchor = out.logits[:, -1].argmax(-1, keepdim=True)
    result.telemetry.update(initial_anchor=anchor.item(),initial_anchor_count=1)
    result.tokens.append(anchor.item())
    confirmed = input_ids
    result.telemetry["target_processed_positions"] = input_ids.numel()

    def stopped():
        return policy.stopped(result.tokens)

    while len(result.tokens) < max_new_tokens and not stopped():
        guard_decode(guard,result.tokens,result)
        old_length = cache.get_seq_length()
        assert old_length == confirmed.shape[1]
        if incremental:
            assert features.shape[1] == old_length
        synchronize(input_ids.device)
        started = time.perf_counter()
        q = propose.from_features(features, confirmed, anchor) if incremental else propose(confirmed, anchor)
        synchronize(input_ids.device)
        draft_seconds = time.perf_counter() - started
        result.telemetry["draft_seconds"] += draft_seconds
        if q.ndim != 2 or q.shape[0] != 1 or q.shape[1] == 0:
            raise ValueError("expected nonempty future-token proposal [1, horizon]")
        q = q[:, :max_new_tokens - len(result.tokens)]
        verify = torch.cat([anchor, q], dim=1)
        guard_decode(guard,result.tokens,result)
        started = time.perf_counter()
        out = target(input_ids=verify, past_key_values=cache, use_cache=True, output_hidden_states=incremental)
        synchronize(input_ids.device)
        verification_seconds = time.perf_counter() - started
        result.telemetry["verification_seconds"] += verification_seconds
        result.telemetry["target_processed_positions"] += verify.numel()
        choices = out.logits.argmax(-1)
        accepted = 0
        for j in range(q.shape[1]):
            if q[0, j] != choices[0, j]:
                break
            result.tokens.append(q[0, j].item())
            accepted += 1
            result.accepted += 1
            if len(result.tokens) >= max_new_tokens or stopped():
                break
        started = time.perf_counter()
        keep = old_length + 1 + accepted
        remove=cache.get_seq_length()-keep
        if remove>0:
            cache.crop(-remove)
        confirmed = torch.cat([confirmed, verify[:, :1 + accepted]], dim=1)
        if incremental:
            new_features = extract_context_feature(out.hidden_states, propose.taps).detach()
            feature_buffer[:, old_length:keep] = new_features[:, :1 + accepted]
            features = feature_buffer[:, :keep]
            result.telemetry["feature_cache_used_bytes"] = features.numel() * features.element_size()
        synchronize(input_ids.device)
        result.telemetry["cache_seconds"] += time.perf_counter() - started
        result.verification_calls += 1
        result.cycles.append({"anchor_position": old_length, "accepted": accepted, "proposed": q.shape[1],
                              "cache_keep_length": keep, "committed_input_ids": verify[:, :1 + accepted].tolist(),
                              "draft_seconds": draft_seconds, "verification_seconds": verification_seconds})
        if len(result.tokens) >= max_new_tokens or stopped():
            break
        anchor = choices[:, accepted:accepted + 1]
        result.tokens.append(anchor.item())
        if accepted == q.shape[1]:
            result.bonus += 1
        else:
            result.corrections += 1
    if stopped():
        result.telemetry["stop_reason"] = "eos" if result.tokens[-1] in policy.eos_ids else "stop_sequence"
    if input_ids.device.type == "cuda":
        result.telemetry["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(input_ids.device)
    synchronize(input_ids.device)
    result.telemetry['generation_seconds']=time.perf_counter()-decode_started
    result.telemetry['component_timing_note']='component scopes are not an additive end-to-end duration; use generation_seconds'
    return result
