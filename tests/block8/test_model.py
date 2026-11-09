from dataclasses import asdict
import inspect
import random
import pytest
import torch
from ncm_lab.block8.config import Architecture, Training, Decoding
from ncm_lab.block8.model import BlockDecoder, active_geometry, generate
from ncm_lab.block8.training import example_loss
from ncm_lab.teacher.model import load_original
from ncm_lab.validation.fixtures import model_spec


def model(layers=2):
    spec = model_spec()
    spec["config"]["num_hidden_layers"] = layers
    return BlockDecoder(load_original(spec, frozen=False))


def test_attention_geometry_and_rope_cached_full_prefix_equivalence():
    m = model().eval()
    context = torch.tensor([[1, 4, 5]])
    active = torch.tensor([[-1, 7, -1, -1, -1, -1, -1, -1]])
    mask = active_geometry(3, 8, torch.float32, "cpu")[0, 0]
    assert torch.isneginf(mask[:3, 3:]).all()
    assert torch.isneginf(mask[0, 1:3]).all() and mask[2, :3].eq(0).all()
    assert mask[3:, :].eq(0).all()
    with torch.inference_mode():
        cache = m.prefill(context)
        before = [(x.keys.clone(), x.values.clone()) for x in cache.layers]
        expected = m.denoise(context, active)
        actual = m.denoise(context, active, prefix_cache=cache)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert cache.get_seq_length() == 3
        for layer, (k, v) in zip(cache.layers, before):
            assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
        committed = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]])
        extended = torch.cat((context, committed), 1)
        cache = m.prefill(committed, cache)
        rebuilt = m.prefill(extended)
        for old, new in zip(cache.layers, rebuilt.layers):
            torch.testing.assert_close(old.keys, new.keys, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(old.values, new.values, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            m.denoise(extended, active, prefix_cache=cache), m.denoise(extended, active), atol=1e-6, rtol=1e-5
        )


def test_all_36_backbone_layers_receive_finite_gradients_and_gold_stays_outside_api():
    m = model(36).train()
    policy = Training()
    total, denoising, ar = example_loss(m, [1, 4, 5], [8, 9, 10, 2], 8, random.Random(3), policy)
    total.backward()
    assert denoising.isfinite() and ar.isfinite()
    assert torch.isfinite(m.mask_embedding.grad).all() and m.mask_embedding.grad.abs().sum() > 0
    for name, p in m.backbone.named_parameters():
        assert p.requires_grad and p.grad is not None and p.grad.isfinite().all(), name
        assert p.grad.abs().sum() > 0, name
    assert set(inspect.signature(m.denoise).parameters) == {"context_ids", "block_input_ids", "prefix_cache"}
    assert "labels" not in inspect.signature(m.causal_states).parameters


def test_shifted_ar_alignment_and_same_position_block_ce():
    m = model().eval()
    context, answer = [1, 4, 5], [8, 9, 2]
    policy = Training(fully_masked_fraction=1)
    total, denoising, ar = example_loss(m, context, answer, 8, random.Random(0), policy)
    states = m.causal_states(torch.tensor([context + answer[:-1]]))
    expected = torch.nn.functional.cross_entropy(
        m.backbone.lm_head(states[:, 2:]).float().flatten(0, 1), torch.tensor(answer)
    )
    torch.testing.assert_close(ar, expected)
    masked = m.denoise(torch.tensor([context]), torch.full((1, 8), -1, dtype=torch.long))
    direct = torch.nn.functional.cross_entropy(
        m.backbone.lm_head(masked[:, :3]).float().flatten(0, 1), torch.tensor(answer)
    )
    torch.testing.assert_close(denoising, direct)
    torch.testing.assert_close(total, direct + 0.5 * expected)


@pytest.mark.parametrize("position", range(8))
def test_eos_at_every_block_position_and_no_fake_eos(position):
    m = model().eval()
    with torch.no_grad():
        m.backbone.lm_head.weight.zero_()

    # A deterministic head seam controls emitted tokens; actual Qwen/cache paths still execute.
    class Head(torch.nn.Module):
        def forward(self, states):
            logits = states.new_full((1, 8, 32), -100)
            logits[:, :, 8] = 100
            logits[:, position, 2] = 200
            return logits

    m.backbone.lm_head = Head()
    result = generate(
        m,
        [1, 4],
        asdict(Decoding(refinement_passes=4, max_new_tokens=16, max_blocks=2)),
        [2],
        max_context_tokens=32,
    )
    assert result["token_ids"] == [8] * position + [2]
    assert result["eos_genuine"] and not result["censored"] and result["stop_reason"] == "eos"
    assert result["cache_positions"] == 2 + position + 1
    assert result["backbone_calls"] == 6  # prefill, four refinements, causal commit


def test_multi_block_output_cap_never_fabricates_eos_and_counts_commit_work():
    m = model().eval()
    with torch.no_grad():
        m.backbone.lm_head.weight.zero_()
    result = generate(
        m,
        [1, 4],
        asdict(Decoding(refinement_passes=2, max_new_tokens=24, max_blocks=3)),
        [2],
        max_context_tokens=32,
        suppressed_ids=[1, 2],
    )
    # EOS is not suppressed, but equal logits choose token zero. No artificial terminator is added.
    assert len(result["token_ids"]) == 24 and result["stop_reason"] == "output_token_cap"
    assert result["censored"] and not result["eos_genuine"]
    assert result["blocks"] == 3 and result["cache_positions"] == 26
    assert result["backbone_calls"] == 10 and result["processed_input_positions"] == 2 + 3 * 24


def test_context_states_cannot_read_active_or_future_blocks():
    m = model().eval()
    context = torch.tensor([[1, 4, 5]])
    # Explicitly perturb a later active block token: earlier causal text states must stay equal.
    outputs = []
    hook = m.backbone.model.register_forward_hook(
        lambda module, args, result: outputs.append(result.last_hidden_state.detach())
    )
    m.denoise(context, torch.full((1, 8), -1, dtype=torch.long))
    m.denoise(context, torch.tensor([[-1] * 7 + [9]]))
    hook.remove()
    assert torch.equal(outputs[0][:, :3], outputs[1][:, :3])


def test_invalid_architecture_and_training_contracts():
    with pytest.raises(ValueError):
        Architecture(block_size=16)
    with pytest.raises(ValueError):
        Training(steps=3)
    m = model()
    with pytest.raises(TypeError):
        m.denoise(torch.tensor([[1, 4]]), torch.full((1, 8), -1), labels=torch.tensor([[2]]))


def test_checkpointed_backbone_supports_both_losses():
    m = model().train()
    m.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    loss, _, _ = example_loss(m, [1, 4, 5], [8, 9, 2], 8, random.Random(0), Training())
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
