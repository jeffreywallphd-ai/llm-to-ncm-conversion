"""Tiny production workflow; synthetic transition plus separately captured predecessor evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import pytest
import torch
from ncm_lab.runtime.runner import execute
from ncm_lab.runtime.io import read, digest, resolve_reference
from ncm_lab.training.checkpoints import CheckpointReader, save_checkpoint, native_identity
from ncm_lab.training.recovery_compatibility import PREDECESSOR, POLICY


def generator_config(native_artifacts, tmp_path):
    value = copy.deepcopy(native_artifacts["configs"]["generator"])
    value["run_root"] = str(tmp_path / "runs")
    value["output_ref"] = str(tmp_path / "generator.json")
    value["dtype"] = "bfloat16"
    value["budget"].update(phase_id="clock-" + uuid.uuid4().hex, enforce_wall_time_limits=False)
    value["monitoring"] = {"sample_prompts": 0, "first_steps": 3}
    return value


def advancing_clock(monkeypatch):
    import ncm_lab.runtime.budget as budgets

    clock = SimpleNamespace(now=0.0)

    def tick():
        clock.now += 1_000_000
        return clock.now

    monkeypatch.setattr(budgets, "time", SimpleNamespace(perf_counter=tick))
    return clock


def payload(path):
    reader = CheckpointReader(path)
    return reader.materialize(reader.manifest["tree"])


def synthetic_predecessor(path, folder):
    """New independent fixture, not modification of an original checkpoint's identity."""
    value = payload(path)
    value["metadata"]["implementation_sha256"] = PREDECESSOR
    value["identity"] = native_identity(value["metadata"])
    return save_checkpoint(folder, value)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_disabled_pilot_completes_and_projects_truthful_monitoring(native_artifacts, tmp_path, monkeypatch):
    from ncm_lab.inference.export import load_export
    from ncm_lab.monitoring.projector import Projector
    import ncm_lab.monitoring.projector as projection_module
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    value = generator_config(native_artifacts, tmp_path)
    advancing_clock(monkeypatch)
    result = execute(value)
    assert result["status"] == "completed" and result["optimizer_steps"] == 2, result
    assert result["completion"]["code"] == "objective_attained"
    assert Path(result["checkpoint"]).is_file()
    assert resolve_reference(value["output_ref"]) == Path(result["artifact"]).resolve()
    load_export(result["artifact"], "cpu")  # Verify all export/tokenizer files with the public loader.
    folder = Path(result["path"])
    events = [json.loads(line) for line in (folder / "monitor_events.jsonl").read_text().splitlines()]
    assert any(row.get("status") == "validation_completed" for row in events)
    assert any(row.get("status") == "checkpoint_completed" for row in events)
    assert (
        "Wall-clock limits: disabled"
        in next(row for row in events if row.get("status") == "started")["wall_policy_message"]
    )
    rows = [json.loads(line) for line in (folder / "monitor_metrics.jsonl").read_text().splitlines()]
    train = [row["metrics"] for row in rows if row["split"] == "train"]
    elapsed = [row["resources/budget_wall_seconds_used"]["value"] for row in train]
    assert elapsed[-1] > elapsed[0] > value["budget"]["max_phase_wall_seconds"]
    for row in train:
        assert row["resources/budget_wall_seconds_remaining"]["value"] is None
        assert row["resources/budget_wall_seconds_remaining"]["status"] == "disabled_by_policy"
        assert row["resources/enforce_wall_time_limits"]["value"] == 0
        assert row["budgets/phase/wall_seconds_reserved"]["value"] == 0
    assert result["wall_policy"]["attempt_reserved_seconds"] == 0
    assert result["wall_policy"]["phase"]["incomplete_unbounded_attempts"] == []
    projection = tmp_path / "tensorboard"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        projection_module.shutil, "disk_usage", lambda path: SimpleNamespace(free=2_000_000_000_000)
    )
    projector = Projector(folder.parents[1], projection, attempt_dir=folder, headroom_bytes=0)
    try:
        while projector.poll():
            pass
    finally:
        projector.close()
    event_file = next(projection.rglob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event_file.parent)).Reload()
    assert "resources/enforce_wall_time_limits" in accumulator.Tags()["scalars"]
    assert "resources/budget_wall_seconds_remaining" not in accumulator.Tags()["scalars"]
    assert "resources/budget_wall_seconds_remaining/status/text_summary" in accumulator.Tags()["tensors"]


@pytest.mark.parametrize("at_target", [False, True])
def test_same_implementation_recovers_exact_state_and_target_finalization(
    native_artifacts, tmp_path, monkeypatch, at_target
):
    value = generator_config(native_artifacts, tmp_path)
    full = execute(value)
    assert full["status"] == "completed", full
    original = full if at_target else execute(value, stop_after=1)
    assert original["status"] == "completed", original
    old = {"manifest": str(CheckpointReader(original["checkpoint"]).path)}
    before = digest(old["manifest"])
    advancing_clock(monkeypatch)
    result = execute(value, old["manifest"], recovery_compatibility=POLICY)
    assert result["status"] == "completed" and result["optimizer_steps"] == 2, result
    assert result["completion"]["code"] == "objective_attained"
    assert resolve_reference(value["output_ref"]) == Path(result["artifact"]).resolve()
    assert digest(old["manifest"]) == before
    a, b = payload(full["checkpoint"]), payload(result["checkpoint"])
    for field in ("model", "optimizer", "rng", "scheduler", "scaler"):
        equal(a[field], b[field])
    for field in (
        "optimizer_steps",
        "processed_source_tokens",
        "answer_position_exposures",
        "allocated_canvas_positions",
        "cursor",
    ):
        equal(a["training"][field], b["training"][field])
    # At-target finalization performs validation charges, but no extra optimizer update.
    assert result["charged_work"]["steps"] == (0 if at_target else 1)
    events = [json.loads(line) for line in (Path(result["path"]) / "events.jsonl").read_text().splitlines()]
    assert not any(row["status"] == "policy_recovery_transition" for row in events)
    if at_target:
        assert any(
            row.get("finalization_only")
            for row in (
                json.loads(line)
                for line in (Path(result["path"]) / "monitor_events.jsonl").read_text().splitlines()
            )
        )


@pytest.mark.parametrize("case", ["no_option", "known_old_source", "unknown", "optimizer", "phase", "model", "corrupt"])
def test_transition_cannot_bypass_integrity_or_contract(native_artifacts, tmp_path, case):
    value = generator_config(native_artifacts, tmp_path)
    original = execute(value, stop_after=1)
    assert original["status"] == "completed", original
    data = payload(original["checkpoint"])
    data["metadata"]["implementation_sha256"] = "f" * 64 if case == "unknown" else PREDECESSOR
    data["identity"] = native_identity(data["metadata"])
    old = save_checkpoint(tmp_path / "negative", data)
    if case == "optimizer":
        value["training"]["learning_rate"] *= 2
    elif case == "phase":
        value["budget"]["phase_id"] += "-changed"
    elif case == "model":
        value["model"]["seed"] += 1
    elif case == "corrupt":
        manifest = read(old["manifest"])
        shard = Path(old["manifest"]).parent / next(iter(manifest["files"]))
        with shard.open("r+b") as stream:
            stream.seek(100)
            stream.write(b"corruption")
    value["output_ref"] = str(tmp_path / "must-not-publish.json")
    result = execute(value, old["manifest"], recovery_compatibility=None if case == "no_option" else POLICY)
    assert result["status"] == "failed", result
    assert not Path(value["output_ref"]).exists()
    assert any(word in result["error"] for word in ("identity", "unreviewed", "corrupt", "mismatch", "receipt does not approve"))


def test_disabled_clock_cancellation_is_still_cancelled(native_artifacts, tmp_path, monkeypatch):
    import ncm_lab.training.engine as engine

    value = generator_config(native_artifacts, tmp_path)
    advancing_clock(monkeypatch)

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(engine, "load_generator", interrupt)
    result = execute(value)
    assert result["status"] == "cancelled" and not Path(value["output_ref"]).exists()
    assert result["wall_policy"]["attempt_reserved_seconds"] == 0
    assert not result["wall_policy"]["phase"]["incomplete_unbounded_attempts"]
