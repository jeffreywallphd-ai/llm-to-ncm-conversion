from dataclasses import replace
import inspect
import pytest
import torch
from ncm_lab.config.architecture import Architecture, RESOLUTIONS, tiny_architecture
from ncm_lab.embedded.geometry import resize, masked_pool, attention_geometry
from ncm_lab.embedded.model import EmbeddedGenerator
from ncm_lab.teacher.model import load_original
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.training.losses import chunked_ce, auxiliary_loss


def generator(canvas=16):
    return EmbeddedGenerator(
        load_original(model_spec(), frozen=False), tiny_architecture(canvas, activation_checkpointing=False)
    )


def test_exact_default_schedule_and_validation():
    assert RESOLUTIONS == tuple(round(4 * 256 ** (i / 31)) for i in range(32))
    assert Architecture().blocks == 36 and len(Architecture().resolutions) == 32
    assert len(set(tiny_architecture().resolutions)) < 32
    for kwargs in [{"resolutions": [4, 16]}, {"gate_initial": 0}, {"canvas": 0}, {"longer_skips": True}]:
        with pytest.raises(ValueError):
            Architecture(**kwargs)


def test_fixed_overlap_resize_and_masked_nondivisible_pooling():
    values = torch.tensor([[[1.0], [3.0], [7.0]]], requires_grad=True)
    output = resize(values, 4, 10)
    assert torch.allclose(output.flatten(), torch.tensor([1.0, 7 / 3, 5.0, 7.0]))
    output.sum().backward()
    assert torch.allclose(values.grad.flatten(), torch.tensor([4 / 3, 7 / 6, 1.5]))
    source = torch.arange(16.0).reshape(1, 16, 1)
    mask = torch.arange(16)[None] < 5
    pooled, valid = masked_pool(source, mask, 5)
    assert torch.equal(valid, torch.tensor([[True, True, False, False, False]]))
    assert torch.equal(pooled.flatten(), torch.tensor([1.0, 3.5, 0, 0, 0]))
    assert torch.equal(resize(torch.ones(2, 7, 3), 13, 16), torch.ones(2, 13, 3))


def test_masks_positions_padding_and_no_cross_example_attention():
    prompt_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    mask, positions = attention_geometry(prompt_mask, 5, 16, torch.float32)
    allowed = torch.isfinite(mask[:, 0])
    assert not allowed[:, :3, 3:].any()
    assert not allowed[0, 2].any()
    assert not allowed[0, :, 2].any()
    assert allowed[:, 3:, 3:].all()
    assert positions[0, 3:].tolist() == [3, 6, 9, 12, 15]
    model = generator().eval()
    ids = torch.tensor([[1, 4, 0], [1, 5, 6]])
    with torch.no_grad():
        first = model(ids, prompt_mask).answer_states
        changed = ids.clone()
        changed[1] = torch.tensor([8, 9, 10])
        changed[0, 2] = 17
        second = model(changed, prompt_mask).answer_states
    assert torch.isfinite(first).all()
    assert torch.equal(first[0], second[0])


def test_full_1024_canvas_32_updates_and_each_original_block_once():
    spec = model_spec()
    spec["config"] = {
        **spec["config"],
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 1,
        "head_dim": 8,
    }
    model = EmbeddedGenerator(
        load_original(spec, frozen=False),
        replace(Architecture(), latent_width=8, activation_checkpointing=False),
    ).eval()
    lengths = []
    updates = []
    handles = [
        layer.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1] - 3))
        for layer in model.backbone.model.layers
    ]
    handles += [
        group.register_forward_hook(lambda module, args, out: updates.append(out[0].shape[1]))
        for group in model.refinements
    ]
    with torch.no_grad():
        output = model(torch.tensor([[1, 4, 5]]), torch.ones(1, 3, dtype=torch.bool), return_aux=True)
    assert output.answer_states.shape == (1, 1024, 8) and len(output.cumulative_codes) == 16
    assert updates == list(RESOLUTIONS) and len(lengths) == 36
    assert lengths[:8] == [4] * 8
    assert lengths[8:24] == [4] + list(RESOLUTIONS[1:30:2])
    assert lengths[24:] == [1024] * 12
    assert all(x["block"] == i + 1 for i, x in enumerate(output.trace))
    for handle in handles:
        handle.remove()


def test_second_update_consumes_first_and_earlier_refinement_changes_downstream():
    model = generator().eval()
    inputs = (torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool))
    captured = []

    def capture(module, args, output):
        captured.append(
            (tuple(x.detach().clone() for x in args[:2]), tuple(x.detach().clone() for x in output))
        )

    handle = model.refinements[0].register_forward_hook(capture)
    with torch.no_grad():
        before = model(*inputs, return_aux=True)
    first = captured[0][1]
    second = captured[1][0]
    assert torch.equal(second[0], resize(first[0], model.architecture.resolutions[1], 16))
    assert torch.equal(second[1], resize(first[1], model.architecture.resolutions[1], 16))
    handle.remove()
    with torch.no_grad():
        model.gates[0].add_(0.3)
        after = model(*inputs, return_aux=True)
    assert not torch.equal(before.cumulative_codes[-1], after.cumulative_codes[-1])
    assert not torch.equal(before.answer_states, after.answer_states)


def test_forward_has_no_gold_or_target_inputs_and_losses_are_external():
    assert set(inspect.signature(EmbeddedGenerator.forward).parameters) == {
        "self",
        "prompt_input_ids",
        "prompt_attention_mask",
        "return_aux",
    }
    model = generator().eval()
    ids = torch.tensor([[1, 4]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    one = model(ids, mask, return_aux=True)
    two = model(ids, mask, return_aux=True)
    gold = torch.full((1, 16), 2, dtype=torch.long)
    valid = torch.ones_like(gold, dtype=torch.bool)
    assert torch.equal(one.answer_states, two.answer_states)
    assert chunked_ce(model.backbone.lm_head, one.answer_states, gold, valid) != chunked_ce(
        model.backbone.lm_head, one.answer_states, gold + 1, valid
    )
    for field in ("labels", "answer_length", "teacher_states", "target_codes"):
        with pytest.raises(TypeError):
            model(ids, mask, **{field: gold})
    targets = torch.zeros(1, 16, 8)
    scales = {str(i): [1.0] * 8 for i in range(1, 32, 2)}
    a, _, _ = auxiliary_loss(one.cumulative_codes, targets, valid, model.architecture.resolutions, scales)
    b, _, _ = auxiliary_loss(one.cumulative_codes, targets + 1, valid, model.architecture.resolutions, scales)
    assert a != b


def test_chunked_parallel_ce_matches_full_reference_and_gradients():
    torch.manual_seed(19)
    head = torch.nn.Linear(8, 32, bias=False)
    states = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.randint(0, 32, (2, 5))
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    reference = torch.nn.functional.cross_entropy(head(states)[valid], labels[valid])
    reference.backward()
    expected_states = states.grad.clone()
    expected_weights = head.weight.grad.clone()
    head.zero_grad()
    states.grad = None
    actual = chunked_ce(head, states, labels, valid, 2)
    actual.backward()
    assert torch.allclose(actual, reference, atol=1e-6)
    assert torch.allclose(states.grad, expected_states, atol=1e-6)
    assert torch.allclose(head.weight.grad, expected_weights, atol=1e-6)


def test_warmup_keeps_autograd_then_full_updates_all_exercised_components():
    model = generator()
    model.architecture = replace(model.architecture, activation_checkpointing=True)
    model.set_phase("warmup")
    ids = torch.tensor([[1, 4, 5]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    labels = torch.tensor([[8, 9, 2] + [0] * 13])
    valid = torch.arange(16)[None] < 3
    loss = chunked_ce(model.backbone.lm_head, model(ids, mask).answer_states, labels, valid)
    loss.backward()
    assert model.queries.grad.abs().sum() > 0 and model.refinements[0].down.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    model.zero_grad()
    model.set_phase("full")
    names = [
        "backbone.model.embed_tokens.weight",
        "backbone.model.layers.0.self_attn.q_proj.weight",
        "backbone.model.layers.8.mlp.down_proj.weight",
        "backbone.model.layers.35.self_attn.o_proj.weight",
        "backbone.model.norm.weight",
        "backbone.lm_head.weight",
        "queries",
        "refinements.0.down.weight",
        "refinements.15.up.weight",
        "gates",
    ]
    parameters = dict(model.named_parameters())
    before = {n: parameters[n].detach().clone() for n in names}
    output = model(ids, mask, return_aux=True)
    loss = (
        chunked_ce(model.backbone.lm_head, output.answer_states, labels, valid)
        + sum(x.square().mean() for x in output.cumulative_codes) * 0.001
    )
    loss.backward()
    from ncm_lab.training.optimizer import TiledAdamW

    optimizer = TiledAdamW(model.named_parameters(), lr=0.001, new_lr=0.002)
    optimizer.step()
    assert all(parameters[n].grad is not None and torch.isfinite(parameters[n].grad).all() for n in names)
    assert all(not torch.equal(parameters[n], before[n]) for n in names)
    assert not torch.equal(
        parameters["backbone.model.embed_tokens.weight"][1], before["backbone.model.embed_tokens.weight"][1]
    )
