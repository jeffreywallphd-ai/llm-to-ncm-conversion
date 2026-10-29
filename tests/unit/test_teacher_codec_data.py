from dataclasses import asdict
import random
import pytest
import torch
from ncm_lab.validation.fixtures import model_spec, fixture_data
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.teacher.model import load_original, prediction_states
from ncm_lab.codec.model import ContinuousCodec
from ncm_lab.embedded.geometry import resize
from ncm_lab.training.losses import codec_loss, auxiliary_loss
from ncm_lab.data.records import Example, collate
from ncm_lab.data.artifacts import Dataset, Cursor
from ncm_lab.runtime.io import write_json


@pytest.mark.parametrize("answer", [[8, 9, 2], [2]])
def test_teacher_prediction_positions_first_interior_and_eos(answer, monkeypatch):
    teacher = load_original(model_spec(), frozen=True)
    prompt = [1, 4, 5]
    states = prediction_states(teacher, prompt, answer)
    with torch.no_grad():
        reference = teacher(torch.tensor([prompt + answer]), use_cache=False).logits[
            0, len(prompt) - 1 : len(prompt) - 1 + len(answer)
        ]
        assert torch.allclose(teacher.lm_head(states), reference, atol=1e-6)

    def forbidden(*args, **kwargs):
        raise AssertionError("extraction materialized vocabulary logits")

    monkeypatch.setattr(teacher.lm_head, "forward", forbidden)
    assert prediction_states(teacher, prompt, answer).shape == (len(answer), 16)


def test_full_answer_labels_eos_censoring_and_legacy_rejection(tmp_path):
    e = Example("a", "g", "train", "software_fixture", [1, 4], [8, 9, 2], True, False, {})
    ids, mask, labels, valid = collate([e], 16)
    assert ids.tolist() == [[1, 4]] and mask.all()
    assert labels[0, :3].tolist() == [8, 9, 2] and valid.sum() == 3
    assert labels[0, 3:].eq(-100).all()
    with pytest.raises(ValueError, match="censoring"):
        Example("a", "g", "train", "software_fixture", [1], [8], True, True, {})
    with pytest.raises(ValueError, match="schema"):
        Example(**{**asdict(e), "schema": "response_records_v2"})
    path = tmp_path / "legacy.json"
    write_json(path, {"schema_version": 3, "state": "known_response_anchor"})
    with pytest.raises(ValueError, match="schema"):
        Dataset(path)


def test_shared_codec_progressive_residuals_and_head_gradients():
    architecture = tiny_architecture()
    teacher = load_original(model_spec(), frozen=True)
    before = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    codec = ContinuousCodec(16, architecture)
    states = torch.randn(2, 16, 16)
    valid = torch.arange(16)[None].expand(2, -1) < torch.tensor([[5], [3]])
    calls = []
    handle = codec.encoder.register_forward_hook(lambda *args: calls.append(1))
    fine = codec.fine_codes(states)
    cumulative, residuals, masks = codec.targets(fine, valid)
    assert len(calls) == 1 and len(cumulative) == 32
    previous = torch.zeros(2, 4, 8)
    for slots, target, residual, mask in zip(architecture.resolutions, cumulative, residuals, masks):
        assert torch.allclose(resize(previous, slots, 16) + residual, target, atol=1e-6)
        assert target[~mask].eq(0).all()
        previous = target
    handle.remove()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=0.001)
    loss, metrics = codec_loss(
        codec, teacher.lm_head, states, valid, 1.0, random.Random(7), max_head_positions=4, chunk_positions=2
    )
    loss.backward()
    assert metrics["head_positions"] == 4
    assert codec.encoder[0].weight.grad.abs().sum() > 0 and codec.decoder[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(
        p.grad is None and not p.requires_grad and torch.equal(p, before[n])
        for n, p in teacher.named_parameters()
    )


def test_auxiliary_loss_uses_raw_cumulative_coordinates_and_equal_group_weighting():
    architecture = tiny_architecture()
    fine = torch.ones(1, 16, 8) * 0.5
    valid = torch.ones(1, 16, dtype=torch.bool)
    predicted = tuple(torch.ones(1, architecture.resolutions[i], 8) for i in range(1, 32, 2))
    scales = {str(i): [2.0] * 8 for i in range(1, 32, 2)}
    loss, groups, counts = auxiliary_loss(predicted, fine, valid, architecture.resolutions, scales)
    assert loss == 0.0625 and all(v == 0.0625 for v in groups) and len(counts) == 16
    with pytest.raises(ValueError, match="empty"):
        auxiliary_loss(predicted, fine, valid & False, architecture.resolutions, scales)


def test_content_bound_artifact_shards_and_group_split(tmp_path):
    path = fixture_data(tmp_path / "source")
    dataset = Dataset(path, "native_data_v1")
    assert not set(dataset.manifest["groups"]["train"]) & set(dataset.manifest["groups"]["dev"])
    assert dataset.manifest["stats"]["unique_answer_positions"] == 36
    assert dataset.manifest["stats"]["padding_positions"] == 12 * 13
    source = tmp_path / "source/records.jsonl"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="index changed"):
        Dataset(path)


def test_cursor_does_not_repeat_to_fake_unique_data_or_exceed_epoch_boundary():
    sampler = random.Random(7)
    cursor = Cursor([0, 1, 2], sampler)
    first = cursor.take(3, 1)
    assert sorted(first) == [0, 1, 2] and cursor.take(1, 1) == []


def test_prediction_head_term_backpropagates_into_both_codec_modules():
    teacher = load_original(model_spec(), frozen=True)
    codec = ContinuousCodec(16, tiny_architecture())
    states = torch.randn(1, 16, 16)
    valid = torch.arange(16)[None] < 5
    loss, _ = codec_loss(
        codec, teacher.lm_head, states, valid, 1.0, random.Random(3), beta=0, max_head_positions=4
    )
    loss.backward()
    encoder = codec.encoder[0].weight.grad.clone()
    decoder = codec.decoder[-1].weight.grad.clone()
    codec.zero_grad()
    loss, _ = codec_loss(
        codec, teacher.lm_head, states, valid, 1.0, random.Random(3), beta=0.1, max_head_positions=4
    )
    loss.backward()
    assert (codec.encoder[0].weight.grad - encoder).abs().sum() > 1e-8
    assert (codec.decoder[-1].weight.grad - decoder).abs().sum() > 1e-8
    assert teacher.lm_head.weight.grad is None
