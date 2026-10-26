import itertools
import random
import pytest
import torch
from ncm_lab.decoding.reference import autoregressive, speculative, rejection_step
from ncm_lab.decoding.hf import cached_greedy


def finite_probs(prefix):
    return [[.8,.1,.1], [.1,.2,.7], [.2,.6,.2]][sum(prefix) % 3]


@pytest.mark.parametrize("horizon", [1,2,4])
@pytest.mark.parametrize("limit", [0,1,2,5,17])
def test_exhaustive_greedy_rejection_full_acceptance_eos(horizon, limit):
    for prompt in ([0], [1], [2,1,0]):
        for proposal in itertools.product(range(3), repeat=horizon):
            q = [[float(i == tok) for i in range(3)] for tok in proposal]
            for eos in ((), (2,)):
                actual = speculative(finite_probs, lambda prefix: q, prompt, limit, eos, stop_sequences=((0,0),))
                assert actual.tokens == autoregressive(finite_probs, prompt, limit, eos, stop_sequences=((0,0),))
                if actual.tokens:
                    assert len(actual.tokens) == 1 + actual.accepted + actual.corrections + actual.bonus


def test_empty_prompt_requires_bos():
    with pytest.raises(ValueError, match="BOS"):
        autoregressive(finite_probs, [], 3)


def test_stochastic_residual_matches_enumerable_distribution():
    p, q = [.15,.25,.60], [.60,.30,.10]
    # Exact marginal algebra: accepted mass min(p,q), plus total rejection * residual.
    reject = sum(max(qi-pi,0) for pi,qi in zip(p,q))
    residual = [max(pi-qi,0) for pi,qi in zip(p,q)]
    marginal = [min(pi,qi) + reject * ri / sum(residual) for pi,qi,ri in zip(p,q,residual)]
    assert marginal == pytest.approx(p)
    rng, observed = random.Random(7), [0,0,0]
    from ncm_lab.decoding.reference import categorical
    for _ in range(15000):
        token = categorical(q,rng)
        _, output = rejection_step(token,p,q,rng)
        observed[output] += 1
    assert [n/15000 for n in observed] == pytest.approx(p, abs=.015)


@pytest.mark.parametrize("limit", [0,1,2,8,19])
def test_actual_hf_cache_rollback_parity(pair, limit):
    target, _ = pair
    ids = torch.tensor([[1,4,8]])
    with torch.no_grad():
        expected = target.generate(ids, max_new_tokens=limit, do_sample=False, eos_token_id=None, pad_token_id=0)[0,3:].tolist() if limit else []
    def bad_proposal(confirmed, anchor):
        return torch.tensor([[0,3,8,11]])
    actual = cached_greedy(target,bad_proposal,ids,limit)
    assert actual.tokens == expected
    assert all(c["cache_keep_length"] == c["anchor_position"] + 1 + c["accepted"] for c in actual.cycles)


@pytest.mark.parametrize("prompt", [[1],[1,5,8],[1,4,5,6,7,8,9]])
def test_actual_cache_full_acceptance_bonus_and_eos(pair,prompt):
    target,_ = pair
    ids = torch.tensor([prompt])
    def exact_proposer(confirmed,anchor):
        prefix = torch.cat([confirmed,anchor],dim=1)
        output = []
        for _ in range(3):
            with torch.no_grad():
                token = target(prefix,use_cache=False).logits[:,-1].argmax(-1,keepdim=True)
            output.append(token)
            prefix = torch.cat([prefix,token],dim=1)
        return torch.cat(output,dim=1)
    with torch.no_grad():
        expected = target.generate(ids,max_new_tokens=11,do_sample=False,eos_token_id=None,pad_token_id=0)[0,len(prompt):].tolist()
    actual = cached_greedy(target,exact_proposer,ids,11)
    assert actual.tokens == expected and actual.bonus > 0 and actual.corrections == 0
    stopped = cached_greedy(target,exact_proposer,ids,11,eos=[expected[0]])
    assert stopped.tokens == expected[:1] and stopped.verification_calls == 0
