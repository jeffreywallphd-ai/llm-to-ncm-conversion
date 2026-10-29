import math
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from ncm_lab.training.optimizer import TiledAdamW
from ncm_lab.training.precision import stochastic_bf16
from ncm_lab.training.checkpoints import save_checkpoint, CheckpointReader


def test_rounding_adjacent_probabilities_and_exact_endpoints():
    # Independent BF16 reference: endpoints are supplied exactly, not derived by
    # the implementation's nearest/nextafter algorithm.
    intervals = [
        (1.0, 1.0078125),
        (-1.0078125, -1.0),
        (0.99609375, 1.0),
        (-1.0, -0.99609375),
        (0.0, 2.0**-133),
        (-(2.0**-133), -0.0),
        (2.0**-126, 2.0**-126 + 2.0**-133),
    ]
    for lower, upper in intervals:
        for fraction in (0.25, 0.5, 0.75):
            value = torch.tensor(lower + (upper - lower) * fraction, dtype=torch.float32)
            count = 32768
            draws = stochastic_bf16(value.expand(count), torch.Generator().manual_seed(14)).float()
            assert bool(((draws == lower) | (draws == upper)).all())
            observed = float((draws == upper).float().mean())
            assert abs(observed - fraction) < 0.02  # >7 binomial standard errors at n=32768
    exact = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 0.5, 2.0**-133, -(2.0**-133), torch.finfo(torch.bfloat16).max],
        dtype=torch.float32,
    )
    result = stochastic_bf16(exact, torch.Generator().manual_seed(2))
    assert torch.equal(result.float(), exact)
    assert torch.equal(torch.signbit(result), torch.signbit(exact))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max])
def test_rounding_rejects_unsupported_values(value):
    with pytest.raises(ValueError, match="nonfinite/overflow"):
        stochastic_bf16(torch.tensor([value]), torch.Generator())


def test_repeated_small_updates_reproduce_old_cancellation_and_correct_ensemble_progress():
    starts = torch.tensor([1.0, 0.10009765625, 0.02001953125, 0.010009765625]).repeat_interleave(512)
    old = starts.bfloat16()
    corrected = torch.nn.Parameter(old.clone())
    reference = torch.nn.Parameter(starts.clone())
    opt = TiledAdamW([("weight", corrected)], lr=1e-5, weight_decay=0.01, seed=95, tile_elements=4096)
    fp32 = torch.optim.AdamW([reference], lr=1e-5, weight_decay=0.01)
    corrected.grad = torch.ones_like(corrected)
    reference.grad = torch.ones_like(reference)
    for _ in range(1000):
        # Local regression of the old discarded nearest-rounded increments.
        old.mul_(1 - 1e-5 * 0.01)
        old.add_(torch.full_like(old, -1e-5))
        opt.step()
        fp32.step()
    assert torch.equal(old.float(), starts)
    means = corrected.float().reshape(4, -1).mean(1)
    targets = reference.detach().reshape(4, -1).mean(1)
    assert torch.all((starts.reshape(4, -1).mean(1) - means) > 0.005)
    assert torch.allclose(means, targets, atol=0.0015, rtol=0)  # ensemble bound dominated by spacing near 1.0
    assert opt.rounding_draws == 1000 * starts.numel()
    assert all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in opt.state.values())
    assert all(set(s) == {"step", "exp_avg", "exp_avg_sq"} for s in opt.state.values())


def test_fp32_multigroup_reference_with_missing_gradients():
    torch.manual_seed(24)
    a = [torch.nn.Parameter(torch.randn(7)), torch.nn.Parameter(torch.randn(9))]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    ours = TiledAdamW(
        zip(("backbone.weight", "new.weight"), a), lr=1e-5, new_lr=3e-5, weight_decay=0.03, tile_elements=3
    )
    reference = torch.optim.AdamW(
        [{"params": [b[0]], "lr": 1e-5}, {"params": [b[1]], "lr": 3e-5}], weight_decay=0.03
    )
    for step in range(8):
        for i, (left, right) in enumerate(zip(a, b)):
            left.grad = None if (step + i) % 3 == 0 else torch.full_like(left, (-1.0) ** step * (step + 0.25))
            right.grad = None if left.grad is None else left.grad.clone()
        ours.step()
        reference.step()
    assert all(torch.allclose(x, y, atol=1e-7, rtol=1e-6) for x, y in zip(a, b))
    assert ours.rounding_draws == 0


def test_nonfinite_late_gradient_rejects_entire_update_without_rng_or_state_change():
    a = torch.nn.Parameter(torch.ones(11, dtype=torch.bfloat16))
    b = torch.nn.Parameter(torch.ones(11, dtype=torch.bfloat16))
    opt = TiledAdamW([("a", a), ("b", b)], tile_elements=3)
    a.grad = torch.ones_like(a)
    b.grad = torch.ones_like(b)
    b.grad[-1] = float("nan")
    rng = opt.generator.get_state().clone()
    with pytest.raises(ValueError, match="nonfinite"):
        opt.step()
    assert torch.equal(a, torch.ones_like(a)) and torch.equal(b, torch.ones_like(b))
    assert not opt.state and opt.rounding_draws == 0 and torch.equal(rng, opt.generator.get_state())


def test_exact_private_rng_and_bounded_shard_recovery(tmp_path):
    model = torch.nn.Linear(13, 11).bfloat16()
    opt = TiledAdamW(model.named_parameters(), lr=1e-5, tile_elements=17, seed=29)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    saved = save_checkpoint(
        tmp_path / "save",
        {"identity": "numerics", "model": model.state_dict(), "optimizer": opt.state_dict()},
        shard_bytes=1024,
    )
    replica = torch.nn.Linear(13, 11).bfloat16()
    restored = TiledAdamW(replica.named_parameters(), lr=1e-5, tile_elements=17, seed=29)
    reader = CheckpointReader(saved["manifest"])
    reader.load_model(replica)
    reader.load_optimizer(restored)
    for p in replica.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    restored.step()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), replica.parameters()))
    assert torch.equal(opt.generator.get_state(), restored.generator.get_state())
    assert opt.rounding_draws == restored.rounding_draws
    for name, state in opt.state.items():
        assert state["step"] == restored.state[name]["step"]
        assert torch.equal(state["exp_avg"], restored.state[name]["exp_avg"])
    bad = TiledAdamW(replica.named_parameters(), lr=1e-5, tile_elements=18, seed=29)
    with pytest.raises(ValueError, match="precision"):
        reader.load_optimizer(bad)


def test_absent_and_frozen_parameters_do_not_consume_randomness_or_moments():
    a = torch.nn.Parameter(torch.ones(10, dtype=torch.bfloat16))
    b = torch.nn.Parameter(torch.ones(10, dtype=torch.bfloat16), requires_grad=False)
    opt = TiledAdamW([("a", a), ("b", b)], weight_decay=0.5)
    b.grad = torch.ones_like(b)
    rng = opt.generator.get_state().clone()
    opt.step()
    assert not opt.state and torch.equal(rng, opt.generator.get_state()) and opt.rounding_draws == 0


def test_host_scalar_extractions_do_not_scale_with_tile_count():
    class Count(TorchDispatchMode):
        def __init__(self):
            self.scalars = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if "_local_scalar_dense" in str(func):
                self.scalars += 1
            return func(*args, **(kwargs or {}))

    counts = []
    for tile in (7, 1024):
        p = torch.nn.Parameter(torch.ones(1024, dtype=torch.bfloat16))
        p.grad = torch.ones_like(p)
        opt = TiledAdamW([("p", p)], tile_elements=tile)
        mode = Count()
        with mode:
            report = opt.check_gradients()
            report.clip_(1.0)
            opt.step(report)
        assert math.isfinite(report.norm)
        counts.append(mode.scalars)
    assert counts[0] == counts[1] and counts[0] <= 3


def test_legacy_or_incomplete_optimizer_policy_is_rejected_before_loading(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = TiledAdamW([("weight", parameter)])
    for index, mutation in enumerate(("legacy", "missing_rng", "wrong_rng_dtype")):
        state = optimizer.state_dict()
        if mutation == "legacy":
            state = {"schema": "fp32_moment_adamw_v1", "state": {}}
        elif mutation == "missing_rng":
            state.pop("rounding_rng")
        else:
            state["rounding_rng"] = torch.ones(4, dtype=torch.float32)
        saved = save_checkpoint(
            tmp_path / str(index), {"identity": "test", "optimizer": state}, shard_bytes=1024
        )
        reader = CheckpointReader(saved["manifest"])
        with pytest.raises(ValueError, match="policy|RNG"):
            reader.load_optimizer(optimizer)
        assert not optimizer.state
