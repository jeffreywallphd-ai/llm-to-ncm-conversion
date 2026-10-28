from dataclasses import dataclass, field
import math
import random


def normalize(probs):
    if not probs or any(not math.isfinite(x) or x < 0 for x in probs) or sum(probs) <= 0:
        raise ValueError("invalid probability vector")
    total = sum(probs)
    return [x / total for x in probs]


def categorical(probs, rng):
    probs = normalize(probs)
    u, total = rng.random(), 0.0
    for i, p in enumerate(probs):
        total += p
        if u < total:
            return i
    return len(probs) - 1


def rejection_step(token, p, q, rng):
    p, q = normalize(p), normalize(q)
    if len(p) != len(q) or q[token] <= 0:
        raise ValueError("sampled token must have positive probability under the actual proposal")
    if rng.random() < min(1.0, p[token] / q[token]):
        return True, token
    residual = [max(pi - qi, 0.0) for pi, qi in zip(p, q)]
    return False, categorical(residual, rng)


@dataclass
class DecodeResult:
    tokens: list[int]
    accepted: int = 0
    corrections: int = 0
    bonus: int = 0
    verification_calls: int = 0
    cycles: list[dict] = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)


def _stopped(tokens, eos, stop_sequences):
    return bool(tokens and (tokens[-1] in eos or any(s and tokens[-len(s):] == list(s) for s in stop_sequences)))


def autoregressive(next_probs, prompt, max_new_tokens, eos=(), stop_sequences=(), rng=None):
    if not prompt:
        raise ValueError("empty prompt requires an explicit BOS token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    output = []
    for _ in range(max_new_tokens):
        p = next_probs(list(prompt) + output)
        output.append(categorical(p, rng) if rng else max(range(len(p)), key=p.__getitem__))
        if _stopped(output, eos, stop_sequences):
            break
    return output


def speculative(next_probs, propose, prompt, max_new_tokens, eos=(), stop_sequences=(), seed=None):
    """Recomputing reference. Proposal q[j] conditions only on prefix+known anchor.

    Stochastic latent or history-dependent processors require a different proposal
    interface and are deliberately unsupported here. Greedy is the initial protocol.
    """
    if not prompt:
        raise ValueError("empty prompt requires an explicit BOS token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    rng = random.Random(seed) if seed is not None else None
    result = DecodeResult([])
    if max_new_tokens == 0:
        return result
    first = next_probs(list(prompt))
    result.tokens.append(categorical(first, rng) if rng else max(range(len(first)), key=first.__getitem__))
    # Initial target anchor is emitted once, outside the first verification cycle.
    while len(result.tokens) < max_new_tokens and not _stopped(result.tokens, eos, stop_sequences):
        prefix = list(prompt) + result.tokens
        proposals = propose(prefix)
        if not proposals:
            raise ValueError("proposal must contain at least one actual distribution")
        result.verification_calls += 1
        cycle = {"anchor_position": len(prefix) - 1, "accepted": 0, "cache_keep_length": len(prefix)}
        rejected = False
        for q in proposals:
            q = normalize(q)
            token = categorical(q, rng) if rng else max(range(len(q)), key=q.__getitem__)
            p = normalize(next_probs(list(prompt) + result.tokens))
            if rng:
                accepted, committed = rejection_step(token, p, q, rng)
            else:
                correction = max(range(len(p)), key=p.__getitem__)
                accepted, committed = token == correction, correction
            result.tokens.append(committed)
            if accepted:
                result.accepted += 1
                cycle["accepted"] += 1
                cycle["cache_keep_length"] += 1
            else:
                result.corrections += 1
                rejected = True
            if not accepted or len(result.tokens) >= max_new_tokens or _stopped(result.tokens, eos, stop_sequences):
                break
        if not rejected and len(result.tokens) < max_new_tokens and not _stopped(result.tokens, eos, stop_sequences):
            p = next_probs(list(prompt) + result.tokens)
            result.tokens.append(categorical(p, rng) if rng else max(range(len(p)), key=p.__getitem__))
            result.bonus += 1
        result.cycles.append(cycle)
    return result
