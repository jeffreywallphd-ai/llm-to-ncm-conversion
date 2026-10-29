from dataclasses import asdict
import json
from types import SimpleNamespace
import uuid
import pytest
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.provenance import Run
from ncm_lab.monitoring.projector import Projector, main as project, size
from ncm_lab.monitoring.records import Measure


def attempt(tmp_path):
    value = {
        "stage": "generator",
        "model": model_spec(),
        "architecture": tiny_architecture().to_dict(),
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "run_root": str(tmp_path / "runs"),
        "budget": asdict(Budget("replay-" + uuid.uuid4().hex)),
    }
    return Run(value)


def args(run, output, *extra):
    return [
        "--runs-dir",
        str(run.path.parents[1]),
        "--attempt-dir",
        str(run.path),
        "--output-dir",
        str(output),
        "--headroom-bytes",
        "0",
        "--max-seconds",
        "15",
        "--poll-seconds",
        ".01",
        *extra,
    ]


def test_aggregate_counts_registered_views_not_unrelated_siblings(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    run = attempt(tmp_path)
    run.recorder.metric({"value": Measure(2.0)})
    original = (run.path / "monitor_metrics.jsonl").read_bytes()
    views = tmp_path / "custom"
    views.mkdir()
    (views / "unrelated-model.bin").write_bytes(b"x" * 2000000)
    first = Projector(run.path.parents[1], views / "first", aggregate_bytes=10000, headroom_bytes=0)
    second = Projector(run.path.parents[1], views / "second", aggregate_bytes=10000, headroom_bytes=0)
    try:
        first.check_disk()
        second.check_disk()
        second.aggregate_bytes = size(first.output) + size(second.output) - 1
        with pytest.raises(RuntimeError, match="disk budget"):
            second.check_disk()
        second.aggregate_bytes = 10000
        second.max_bytes = 1
        with pytest.raises(RuntimeError, match="disk budget"):
            second.check_disk()
        second.max_bytes = 10000
        second.headroom = 1
        import ncm_lab.monitoring.projector as module

        monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
        with pytest.raises(RuntimeError, match="headroom"):
            second.check_disk()
        assert original == (run.path / "monitor_metrics.jsonl").read_bytes()
    finally:
        first.close()
        second.close()


def test_replay_missing_empty_complete_and_real_cli_outcome(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    run = attempt(tmp_path)
    assert project(args(run, tmp_path / "empty")) == 0
    assert json.loads((tmp_path / "empty/status.json").read_text())["outcome"] == "empty"
    missing = tmp_path / "runs/generator/missing"
    assert (
        project(
            [
                "--runs-dir",
                str(tmp_path / "runs"),
                "--attempt-dir",
                str(missing),
                "--output-dir",
                str(tmp_path / "missing-view"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "missing-view").exists()
    run.recorder.metric({"value": Measure(1.0)})
    run.recorder.metric({"value": Measure(3.0)})
    assert project(args(run, tmp_path / "complete")) == 0
    status = json.loads((tmp_path / "complete/status.json").read_text())
    assert status["outcome"] == "complete" and status["remaining_bytes"] == 0 and status["records"] == 2
    before = (tmp_path / "complete/status.json").read_bytes()
    assert project(args(run, tmp_path / "complete")) == 2
    assert before == (tmp_path / "complete/status.json").read_bytes()
    from ncm_lab.cli import main

    assert (
        main(
            [
                "monitor",
                "replay",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--attempt-dir",
                str(run.path),
                "--output-dir",
                str(tmp_path / "cli"),
                "--max-seconds",
                "15",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert '"outcome": "complete"' in printed and '"status": "replayed"' not in printed


def test_deadline_backlog_partial_tail_and_corrupt_complete_record(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    run = attempt(tmp_path)
    run.recorder.metric({"value": Measure(2.0)})
    import ncm_lab.monitoring.projector as module
    import itertools

    clock = itertools.count(0, 2)
    actual_time = module.time
    # Force a deadline independently of platform clock granularity and IO speed.
    with monkeypatch.context() as deadline:
        deadline.setattr(
            module,
            "time",
            SimpleNamespace(perf_counter=lambda: next(clock), time=actual_time.time, sleep=lambda _: None),
        )
        assert project(args(run, tmp_path / "deadline", "--max-seconds", "1")) == 3
    status = json.loads((tmp_path / "deadline/status.json").read_text())
    assert status["outcome"] == "incomplete" and status["remaining_bytes"] > 0
    path = run.path / "monitor_metrics.jsonl"
    row = json.loads(path.read_text())
    row["sequence"] = 2
    partial = json.dumps(row).encode()
    with path.open("ab") as f:
        f.write(partial)
    before = path.read_bytes()
    assert project(args(run, tmp_path / "partial")) == 3
    status = json.loads((tmp_path / "partial/status.json").read_text())
    assert status["outcome"] == "incomplete" and status["remaining_bytes"] == len(partial)
    assert status["offsets"][str(path)] == len(before) - len(partial)
    assert status["partial_tail_bytes"][str(path)] == len(partial)
    assert path.read_bytes() == before
    # Live reader waits for newline and then consumes the newly complete record once.
    p = Projector(run.path.parents[1], tmp_path / "live", attempt_dir=run.path, headroom_bytes=0)
    try:
        assert p.poll() == 1
        with path.open("ab") as f:
            f.write(b"\n")
        assert p.poll() == 1 and p.poll() == 0
    finally:
        p.close()
    with path.open("ab") as f:
        f.write(b"{broken}\n")
    assert project(args(run, tmp_path / "corrupt")) == 2
    assert json.loads((tmp_path / "corrupt/status.json").read_text())["outcome"] == "invalid_source"


def test_snapshot_does_not_claim_new_appends_and_identity_mismatch_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    run = attempt(tmp_path)
    run.recorder.metric({"value": Measure(2.0)})
    p = Projector(run.path.parents[1], tmp_path / "snapshot", attempt_dir=run.path, headroom_bytes=0)
    try:
        p.capture_snapshot()
        run.recorder.metric({"value": Measure(4.0)})
        assert p.poll() == 1 and p.poll() == 0
        assert (
            p.progress()["remaining_bytes"] == 0 and p.progress()["scope"] == "captured source byte extents"
        )
    finally:
        p.close()
    path = run.path / "monitor_metrics.jsonl"
    rows = path.read_text().splitlines()
    first = json.loads(rows[0])
    first["attempt_id"] = "different"
    path.write_text(json.dumps(first) + "\n" + rows[1] + "\n")
    assert project(args(run, tmp_path / "identity")) == 2
    assert json.loads((tmp_path / "identity/status.json").read_text())["outcome"] == "invalid_source"
