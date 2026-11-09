from dataclasses import asdict
import inspect
import pytest
import torch
from ncm_lab.block8.config import Training, Decoding
from ncm_lab.block8.model import prefill, generate
from ncm_lab.block8.training import example_loss
from ncm_lab.codec.model import ContinuousCodec
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.embedded.model import EmbeddedGenerator
from ncm_lab.embedded.geometry import resize
from ncm_lab.teacher.model import load_original, prediction_states
from ncm_lab.validation.fixtures import model_spec


def model():
    spec = model_spec()
    spec["config"].update(hidden_size=8, intermediate_size=16, num_attention_heads=1, head_dim=8)
    return EmbeddedGenerator(
        load_original(spec, frozen=False), tiny_architecture(8, activation_checkpointing=False)
    )


def test_all_32_updates_16_codes_and_cache_match_at_every_scale_and_commit():
    m = model().eval()
    context = torch.tensor([[1, 4, 5]])
    mask = torch.ones_like(context, dtype=torch.bool)
    with torch.inference_mode():
        cache = prefill(m, context)
        before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        observed = []
        m.diagnostic_observer = lambda update, slots, *args: observed.append(
            (update, slots, args[0].shape[1])
        )
        full = m(context, mask, return_aux=True)
        full_observed = list(observed)
        observed.clear()
        cached = m.forward_cached_context(context, mask, cache, return_aux=True)
        assert observed == full_observed and [r[0] for r in observed] == list(range(32))
        assert [r[1] for r in observed] == list(m.architecture.resolutions)
        assert len(full.cumulative_codes) == len(cached.cumulative_codes) == 16
        assert full.answer_states.shape == (1, 8, 8)
        assert full.trace[0]["answer_positions"] == 4
        assert cached.trace[0]["sequence_positions"] == 4
        torch.testing.assert_close(cached.answer_states, full.answer_states, rtol=2e-5, atol=2e-6)
        for a, b in zip(full.cumulative_codes, cached.cumulative_codes):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        assert cache.get_seq_length() == 3
        for layer, (k, v) in zip(cache.layers, before):
            assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
        for _ in range(2):
            emitted = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]])
            cache = prefill(m, emitted, cache)
            context = torch.cat((context, emitted), 1)
            rebuilt = prefill(m, context)
            for a, b in zip(cache.layers, rebuilt.layers):
                torch.testing.assert_close(a.keys, b.keys, rtol=2e-5, atol=2e-6)
            full = m(context, torch.ones_like(context, dtype=torch.bool), return_aux=True)
            cached = m.forward_cached_context(
                context, torch.ones_like(context, dtype=torch.bool), cache, return_aux=True
            )
            torch.testing.assert_close(full.answer_states, cached.answer_states, rtol=3e-5, atol=3e-6)
        m.diagnostic_observer = None


def test_full_backbone_hierarchy_gradients_and_frozen_target_boundary():
    m = model().train()
    fine = torch.randn(1, 8, 8, requires_grad=True)
    scales = {str(i): [1.0] * 8 for i in range(1, 32, 2)}
    loss, parts, counts = example_loss(m, [1, 4, 5], [8, 9, 10, 2], fine, scales, Training())
    loss.backward()
    assert len(counts) == 16 and all(c > 0 for c in counts)
    assert fine.grad is None
    for name, p in m.named_parameters():
        assert p.requires_grad and p.grad is not None and p.grad.isfinite().all(), name
        assert p.grad.abs().sum() > 0, name
    assert m.queries.shape == (4, 8) and m.level_embeddings.shape == (32, 8)
    assert set(inspect.signature(m.forward).parameters) == {
        "prompt_input_ids",
        "prompt_attention_mask",
        "return_aux",
    }
    assert not hasattr(m, "mask_embedding") and not hasattr(m, "denoise")
    with pytest.raises(TypeError):
        m(torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), labels=torch.tensor([[2]]))


def test_first_token_eos_and_teacher_block_alignment_and_codec_telescoping():
    m = model().eval()
    teacher = load_original(model_spec(), frozen=True)
    context, answer = [1, 4, 5], list(range(8, 24)) + [2]
    states = prediction_states(teacher, context, answer)
    for start in (0, 8, 16):
        block = answer[start : start + 8]
        direct = prediction_states(teacher, context + answer[:start], block)
        torch.testing.assert_close(states[start : start + 8], direct)
    fine = torch.randn(1, 8, 8)
    scales = {str(i): [1.0] * 8 for i in range(1, 32, 2)}
    total, parts, _ = example_loss(m, context, [8, 9, 2], fine, scales, Training())
    ids = torch.tensor([context + [8, 9]])
    hidden = m.backbone.model(input_ids=ids, use_cache=False).last_hidden_state[:, len(context) - 1 :]
    expected = torch.nn.functional.cross_entropy(
        m.backbone.lm_head(hidden).float().flatten(0, 1), torch.tensor([8, 9, 2])
    )
    torch.testing.assert_close(parts["ar_ce"], expected)
    output = m(torch.tensor([context]), torch.ones(1, 3, dtype=torch.bool))
    direct = torch.nn.functional.cross_entropy(
        m.backbone.lm_head(output.answer_states[:, :3]).float().flatten(0, 1), torch.tensor([8, 9, 2])
    )
    torch.testing.assert_close(parts["text_ce"], direct)
    codec = ContinuousCodec(8, m.architecture)
    valid = torch.tensor([[True] * 3 + [False] * 5])
    cumulative, residuals, masks = codec.targets(fine, valid)
    previous = torch.zeros(1, 4, 8)
    for slots, cum, residual in zip(m.architecture.resolutions, cumulative, residuals):
        previous = resize(previous, slots, 8) + residual
        torch.testing.assert_close(previous, cum)
    assert all(mask.dtype == torch.bool for mask in masks)
    assert not any(p.requires_grad for p in teacher.parameters())


@pytest.mark.parametrize("position", range(8))
def test_eos_every_position_with_one_hierarchy_traversal(position):
    m = model().eval()

    class Head(torch.nn.Module):
        def forward(self, states):
            logits = states.new_full((1, 8, 32), -100)
            logits[:, :, 8] = 100
            logits[:, position, 2] = 200
            return logits

    m.backbone.lm_head = Head()
    result = generate(
        m, [1, 4], asdict(Decoding(max_new_tokens=16, max_blocks=2)), [2], max_context_tokens=32
    )
    assert result["token_ids"] == [8] * position + [2]
    assert result["eos_genuine"] and not result["censored"]
    assert result["hierarchy_updates"] == 32 and result["backbone_calls"] == 3
    assert result["cache_positions"] == 2 + position + 1


def test_multi_block_caps_and_no_fabricated_eos():
    m = model().eval()
    with torch.no_grad():
        m.backbone.lm_head.weight.zero_()
    result = generate(
        m, [1, 4], asdict(Decoding(max_new_tokens=24, max_blocks=3)), [2], max_context_tokens=32
    )
    assert len(result["token_ids"]) == 24 and result["stop_reason"] == "output_token_cap"
    assert result["censored"] and result["hierarchy_updates"] == 96
    assert result["backbone_calls"] == 7 and result["cache_positions"] == 26


def test_native_checkpointed_hierarchy_backward():
    m = model().train()
    from dataclasses import replace

    m.architecture = replace(m.architecture, activation_checkpointing=True)
    fine = torch.randn(1, 8, 8)
    loss, _, _ = example_loss(
        m, [1, 4], [8, 9, 2], fine, {str(i): [1.0] * 8 for i in range(1, 32, 2)}, Training()
    )
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())


def test_inference_geometry_constants_do_not_poison_later_backward():
    from ncm_lab.embedded.geometry import _overlap

    _overlap.cache_clear()
    with torch.inference_mode():
        resize(torch.zeros(1, 4, 8), 5, 8)
    value = torch.randn(1, 4, 8, requires_grad=True)
    resize(value, 5, 8).square().sum().backward()
    assert value.grad.isfinite().all()
