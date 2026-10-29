import copy
import json
from pathlib import Path
import random
import torch
import pytest
from ncm_lab.runtime.runner import execute
from ncm_lab.training.checkpoints import CheckpointReader
from ncm_lab.monitoring.samples import preserve_training_state, observe_samples
from ncm_lab.inference.export import load_export
from ncm_lab.runtime.provenance import Run
from ncm_lab.data.artifacts import Dataset


@pytest.mark.parametrize("stage", ["codec", "generator"])
def test_monitoring_and_samples_leave_training_updates_rng_order_unchanged(
    native_artifacts, stage, monkeypatch
):
    outputs = []
    backward = torch.Tensor.backward
    calls = []

    def counted(tensor, *args, **kwargs):
        calls.append(1)
        return backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", counted)
    for enabled in (False, True):
        cfg = copy.deepcopy(native_artifacts["configs"][stage])
        cfg["output_ref"] = None
        cfg["monitoring"] = {
            "enabled": enabled,
            "sample_interval": 1,
            "sample_prompts": 1,
            "detail_interval": 1,
        }
        result = execute(cfg)
        assert result["status"] == "completed", result
        assert len(calls) == 2
        calls.clear()
        reader = CheckpointReader(result["checkpoint"])
        outputs.append((reader, result))
    left, right = [r.materialize(r.field("model")) for r, _ in outputs]
    assert all(torch.equal(left[n], right[n]) for n in left)
    for field in ("training", "rng"):
        a, b = [r.materialize(r.field(field)) for r, _ in outputs]
        if field == "training":
            assert a["cursor"] == b["cursor"] and a["processed_source_tokens"] == b["processed_source_tokens"]
        else:
            assert torch.equal(a["torch"], b["torch"])
    folder = Path(outputs[1][1]["path"])
    records = [json.loads(x) for x in (folder / "monitor_metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["split"] == "train"]
    assert [r["step"] for r in train] == [1, 2]
    if stage == "generator":
        assert (folder / "monitor_samples.jsonl").exists()
        for row in train:
            m = row["metrics"]
            prefix = "generator/train/"
            contributions = sum(
                v["value"]
                for k, v in m.items()
                if k.startswith(prefix + "aux/") and k.endswith("/weighted_contribution")
            )
            assert m[prefix + "loss/total_objective"]["value"] == pytest.approx(
                m[prefix + "loss/text_ce"]["value"] + contributions, abs=1e-6
            )
    else:
        for row in train:
            m = row["metrics"]
            prefix = "codec/train/loss/"
            assert m[prefix + "total_objective"]["value"] == pytest.approx(
                sum(
                    m[prefix + k]["value"]
                    for k in ("full_weighted", "sampled_cutoff_weighted", "head_kl_weighted")
                ),
                abs=1e-6,
            )
            cutoff_tags = [
                k
                for k in m
                if "/cutoff_" in k
                and k.endswith("/reconstruction_normalized_mse")
                and m[k]["value"] is not None
            ]
            assert len(cutoff_tags) == 1


def test_sample_path_never_reads_targets_or_teacher_and_restores_mixed_modes(native_artifacts, monkeypatch):
    model, tokenizer, _ = load_export(native_artifacts["results"]["generator"]["artifact"])
    model.train()
    model.backbone.model.layers[0].eval()
    modes = [m.training for m in model.modules()]
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    run = Run(cfg)
    run.start()

    def forbidden(*a, **kw):
        raise AssertionError("sample accessed offline targets")

    monkeypatch.setattr(Dataset, "__getitem__", forbidden)
    import ncm_lab.teacher.model as teacher

    monkeypatch.setattr(teacher, "prediction_states", forbidden)
    python = random.getstate()
    cpu = torch.get_rng_state()
    rows = observe_samples(
        model, tokenizer, [{"id": "heldout", "prompt_ids": [1, 4]}], run.recorder, run.meter, [2]
    )
    assert rows[0]["computed_canvas_positions"] == 16
    assert [m.training for m in model.modules()] == modes
    assert random.getstate() == python and torch.equal(cpu, torch.get_rng_state())
    import numpy as np

    numpy_state = np.random.get_state()
    with preserve_training_state(model):
        random.random()
        torch.rand(3)
        np.random.random(2)
    assert random.getstate() == python and torch.equal(cpu, torch.get_rng_state())
    restored_numpy = np.random.get_state()
    assert (
        numpy_state[0] == restored_numpy[0]
        and np.array_equal(numpy_state[1], restored_numpy[1])
        and numpy_state[2:] == restored_numpy[2:]
    )
    run.close("completed", {"test": "sample boundary"})


def test_training_needs_no_tensorboard_import(native_artifacts, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("tensorboard") or name.startswith("torch.utils.tensorboard"):
            raise ImportError("optional dashboard unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    result = execute(cfg)
    assert result["status"] == "completed", result


def test_canonical_cleanup_failure_preserves_original_training_error(native_artifacts, monkeypatch):
    import ncm_lab.training.engine as engine

    def failed(*args, **kwargs):
        raise ValueError("original training error")

    def broken_close(*args, **kwargs):
        raise OSError("canonical disk failure")

    monkeypatch.setattr(engine, "fit", failed)
    monkeypatch.setattr(Run, "close", broken_close)
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    result = execute(cfg)
    assert result["status"] == "failed" and result["error"] == "ValueError: original training error"
    assert "canonical disk failure" in result["canonical_storage_error"]


def test_documented_monitor_smoke_runs_complete_native_pipeline(tmp_path):
    from ncm_lab.config.stages import config

    cfg = config("configs/monitoring/smoke.yaml")
    cfg["run_root"] = str(tmp_path / "runs")
    result = execute(cfg)
    assert result["status"] == "completed", result
    assert result["fresh_process_generation"]["status"] == "completed"
    assert all(row["status"] == "completed" for row in result["micro_overfit"])
    assert len(result["stages"]) == 8
