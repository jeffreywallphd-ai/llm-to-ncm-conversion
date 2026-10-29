import copy
import json
from pathlib import Path
import pytest
import torch
from ncm_lab.runtime.runner import execute
from ncm_lab.runtime.io import digest
from ncm_lab.training.checkpoints import CheckpointReader
from ncm_lab.training.engine import architecture_from_metadata
from ncm_lab.teacher.stages import load_codec
from ncm_lab.data.artifacts import Dataset
from ncm_lab.inference.export import load_export, generate_tokens


def terminal_state(pointer):
    reader = CheckpointReader(pointer)
    metadata = reader.materialize(reader.field("metadata"))
    model = architecture_from_metadata(metadata, "cpu")
    reader.load_model(model)
    training = reader.materialize(reader.field("training"))
    return model, training, reader


def test_actual_stages_fit_select_freeze_target_and_export(native_artifacts):
    kit = native_artifacts
    selected = Path(kit["results"]["codec"]["artifact"])
    codec, metadata = load_codec(selected)
    assert metadata["frozen"] and all(not p.requires_grad for p in codec.parameters())
    assert len(metadata["scales"]) == 16 and metadata["normalization"]["source"] == "training only"
    features = Dataset(kit["refs"] / "teacher.json", "native_features_v1")
    targets = Dataset(kit["refs"] / "targets.json", "native_targets_v1")
    example, states = features[0]
    other, codes = targets[0]
    with torch.no_grad():
        expected = codec.fine_codes(states)
    assert example.identity == other.identity and torch.equal(codes, expected)
    assert targets.manifest["identity"]["codec_sha256"] == digest(selected)
    before = digest(selected)
    export = Path(kit["results"]["generator"]["artifact"])
    model, tokenizer, manifest = load_export(export)
    result = generate_tokens(model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), [2])
    assert result["computed_answer_positions"] == 16 and result["backbone_traversals"] == 1
    assert digest(selected) == before and manifest["teacher_or_codec_required"] is False
    assert not (export.parent / "teacher").exists() and not (export.parent / "codec").exists()


def test_uninterrupted_and_stop_resume_match_next_loss_cursor_optimizer_and_weights(native_artifacts):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["training"]["steps"] = 3
    cfg["output_ref"] = None
    uninterrupted = execute(cfg)
    resumed = execute(cfg, native_artifacts["results"]["generator"]["checkpoint"])
    assert uninterrupted["status"] == resumed["status"] == "completed", (uninterrupted, resumed)
    left, ls, lr = terminal_state(uninterrupted["checkpoint"])
    right, rs, rr = terminal_state(resumed["checkpoint"])
    assert ls["cursor"] == rs["cursor"] and ls["optimizer_steps"] == rs["optimizer_steps"] == 3
    assert ls["processed_source_tokens"] == rs["processed_source_tokens"]
    assert all(torch.equal(value, right.state_dict()[name]) for name, value in left.state_dict().items())
    a = [
        json.loads(line) for line in (Path(uninterrupted["path"]) / "metrics.jsonl").read_text().splitlines()
    ]
    b = [json.loads(line) for line in (Path(resumed["path"]) / "metrics.jsonl").read_text().splitlines()]
    assert a[-1]["loss"] == pytest.approx(b[0]["loss"], abs=1e-6)
    from ncm_lab.training.optimizer import TiledAdamW

    ao = TiledAdamW(left.named_parameters(), 0.001, 0.002)
    bo = TiledAdamW(right.named_parameters(), 0.001, 0.002)
    lr.load_optimizer(ao)
    rr.load_optimizer(bo)
    for name, state in ao.state.items():
        assert state["step"] == bo.state[name]["step"]
        assert torch.equal(state["exp_avg"], bo.state[name]["exp_avg"]) and torch.equal(
            state["exp_avg_sq"], bo.state[name]["exp_avg_sq"]
        )


def test_inference_requires_no_teacher_cache_or_codec_encoder(native_artifacts, monkeypatch):
    import ncm_lab.teacher.model as teacher
    import ncm_lab.codec.model as codec

    def forbidden(*args, **kwargs):
        raise AssertionError("offline target dependency used in deployment")

    monkeypatch.setattr(teacher, "load_original", forbidden)
    monkeypatch.setattr(teacher, "prediction_states", forbidden)
    monkeypatch.setattr(codec.ContinuousCodec, "fine_codes", forbidden)
    monkeypatch.setattr(Dataset, "__getitem__", forbidden)
    model, _, _ = load_export(native_artifacts["results"]["generator"]["artifact"])
    result = generate_tokens(model, torch.tensor([[1, 5]]), torch.ones(1, 2, dtype=torch.bool), [2])
    assert len(result["tokens"]) <= 16


def test_missing_targets_block_before_generator_or_teacher_loading(native_artifacts, monkeypatch, tmp_path):
    import ncm_lab.training.engine as engine

    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["inputs"] = {"targets": str(tmp_path / "missing.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("model loaded before target prerequisite")

    monkeypatch.setattr(engine, "load_generator", forbidden)
    result = execute(cfg)
    assert result["status"] == "blocked" and result["model_loaded"] is False


def test_wrong_source_or_recovery_contract_rejected(native_artifacts):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    cfg["training"]["auxiliary_weight"] = 0.2
    cfg["training"]["steps"] = 3
    result = execute(cfg, native_artifacts["results"]["generator"]["checkpoint"])
    assert result["status"] == "failed" and "identity mismatch" in result["error"]
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["model"]["tokenizer_revision"] = "different"
    result = execute(cfg)
    assert result["status"] == "failed" and "identity mismatch" in result["error"]


def test_actual_full_mode_recovery_probe_releases_and_restores(native_artifacts):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["stage"] = "profile"
    cfg["output_ref"] = None
    cfg["training"].update(steps=3, recovery_probe=True)
    result = execute(cfg)
    assert result["status"] == "completed" and result["optimizer_steps"] == 3, result
    events = [json.loads(line) for line in (Path(result["path"]) / "events.jsonl").read_text().splitlines()]
    assert any(row["status"] == "recovery_probe_restored" and row["step"] == 2 for row in events)
    assert all(result["representative_weights_changed"].values())


def test_parallel_eos_truncation_never_repairs_or_reduces_computed_canvas(native_artifacts, monkeypatch):
    model, _, _ = load_export(native_artifacts["results"]["generator"]["artifact"])

    def head(states):
        values = states.new_zeros(*states.shape[:-1], 32)
        values[..., 2] = 10
        return values

    monkeypatch.setattr(model.backbone.lm_head, "forward", head)
    result = generate_tokens(model, torch.tensor([[1, 5]]), torch.ones(1, 2, dtype=torch.bool), [2])
    assert result["tokens"] == [2] and result["eos_found"] and result["computed_answer_positions"] == 16
    result = generate_tokens(model, torch.tensor([[1, 5]]), torch.ones(1, 2, dtype=torch.bool), [31])
    assert result["truncated"] and len(result["tokens"]) == 16 and not result["eos_found"]


def test_standalone_export_can_generate_without_repository_marker(native_artifacts, monkeypatch, tmp_path):
    import ncm_lab.runtime.generation as generation
    import ncm_lab.runtime.provenance as provenance

    def missing():
        raise ValueError("no repository marker at deployment")

    monkeypatch.setattr(generation, "root", missing)
    monkeypatch.setattr(provenance, "root", missing)
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("token4", encoding="utf-8")
    result = generation.run_generation(native_artifacts["results"]["generator"]["artifact"], prompt)
    assert result["status"] == "completed" and result["computed_answer_positions"] == 16, result


def test_final_partial_accumulation_uses_actual_microbatch_count(native_artifacts):
    results = []
    for accumulation in (2, 3):
        cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
        cfg["output_ref"] = None
        cfg["training"].update(steps=1, epochs=1, batch_size=2, accumulation=accumulation)
        result = execute(cfg)
        assert result["status"] == "completed" and result["optimizer_steps"] == 1, result
        results.append(CheckpointReader(result["checkpoint"]))
        metric = json.loads((Path(result["path"]) / "metrics.jsonl").read_text().splitlines()[0])
        assert metric["microbatches"] == 2
    left = results[0].materialize(results[0].field("model"))
    right = results[1].materialize(results[1].field("model"))
    for name in left:
        assert torch.allclose(left[name], right[name], atol=1e-6, rtol=1e-5), name
