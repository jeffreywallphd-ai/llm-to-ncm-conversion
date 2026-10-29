import copy
import json
from pathlib import Path
import socket
import sys
import time
import pytest
from ncm_lab.monitoring.projector import Projector, filesystem_path
from ncm_lab.monitoring.server import Session, stop_owned, spawn
from ncm_lab.monitoring.records import Measure
from ncm_lab.runtime.provenance import Run
from ncm_lab.runtime.runner import execute

pytest.importorskip("tensorboard")


def reader(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(
        filesystem_path(path), size_guidance={"scalars": 0, "tensors": 0, "images": 0}
    ).Reload()


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError("bounded monitor wait expired")


def test_real_projection_matches_actual_training_records(native_artifacts, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    attempt = Path(native_artifacts["results"]["generator"]["path"])
    projector = Projector(attempt.parents[1], tmp_path / "view", attempt_dir=attempt, headroom_bytes=0)
    try:
        while projector.poll():
            pass
        meta = json.loads((attempt / "attempt.json").read_text())
        events = reader(projector.events / meta["stage"] / meta["run_id"] / meta["attempt_id"])
        tag = "generator/train/loss/text_ce"
        rows = [json.loads(x) for x in (attempt / "monitor_metrics.jsonl").read_text().splitlines()]
        source = [r for r in rows if tag in r.get("metrics", {})]
        actual = events.Scalars(tag)
        assert len(actual) == len(source)
        for event, row in zip(actual, source):
            assert event.step == row["step"] and event.wall_time == row["wall_time"]
            assert event.value == pytest.approx(row["metrics"][tag]["value"], abs=1e-6)
        assert "heatmaps/aux" in events.Tags()["images"]
        assert "heatmaps/gates" in events.Tags()["images"]
        assert (
            len(
                [
                    tag
                    for tag in events.Tags()["scalars"]
                    if tag.startswith("hierarchy/update_") and tag.endswith("/gate_effective")
                ]
            )
            == 32
        )
        assert (
            len(
                [
                    tag
                    for tag in events.Tags()["scalars"]
                    if tag.startswith("generator/train/aux/group_") and tag.endswith("/normalized_mse")
                ]
            )
            == 16
        )
    finally:
        projector.close()


def test_live_partial_lines_corruption_nonfinite_and_refuse_replay_overwrite(
    native_artifacts, tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["run_root"] = str(tmp_path / "runs")
    cfg["output_ref"] = None
    p = Projector(tmp_path / "runs", tmp_path / "view", headroom_bytes=0)
    try:
        assert p.poll() == 0
        run = Run(cfg)
        run.start()
        run.recorder.metric({"test/live": Measure(3.0)})
        assert p.poll() > 0
        meta = json.loads((run.path / "attempt.json").read_text())
        events = p.events / "generator" / run.id / meta["attempt_id"]
        assert reader(events).Scalars("test/live")[0].value == 3
        run.recorder.metric({"test/live": Measure(4.0), "test/bad": Measure(float("nan"))})
        assert p.poll() > 0
        assert [x.value for x in reader(events).Scalars("test/live")] == [3.0, 4.0]
        assert "test/bad" not in reader(events).Tags()["scalars"]
        lines = (run.path / "monitor_metrics.jsonl").read_bytes().splitlines(keepends=True)
        row = json.loads(lines[-1])
        row["sequence"] += 1
        encoded = (json.dumps(row) + "\n").encode()
        with (run.path / "monitor_metrics.jsonl").open("ab") as f:
            f.write(encoded[: len(encoded) // 2])
            f.flush()
        assert p.poll() == 0
        with (run.path / "monitor_metrics.jsonl").open("ab") as f:
            f.write(encoded[len(encoded) // 2 :])
            f.flush()
        assert p.poll() == 1
        with pytest.raises(FileExistsError):
            Projector(tmp_path / "runs", tmp_path / "view", headroom_bytes=0)
        with (run.path / "monitor_metrics.jsonl").open("ab") as f:
            f.write(b"{broken}\n")
        with pytest.raises(ValueError):
            p.poll()
        run.close("completed", {"test": "live canonical fixture"})
    finally:
        p.close()


def test_derived_disk_cap_preserves_canonical_sources(native_artifacts, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    attempt = Path(native_artifacts["results"]["generator"]["path"])
    before = (attempt / "monitor_metrics.jsonl").read_bytes()
    p = Projector(attempt.parents[1], tmp_path / "view", max_bytes=1, headroom_bytes=0)
    try:
        with pytest.raises(RuntimeError, match="disk budget"):
            p.poll()
        assert (attempt / "monitor_metrics.jsonl").read_bytes() == before
    finally:
        p.close()


def test_real_server_starts_before_training_discovers_later_resume(native_artifacts, tmp_path):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["run_root"] = str(tmp_path / "runs")
    cfg["output_ref"] = None
    session = Session(tmp_path / "runs", port=port(), max_seconds=90, headroom_bytes=0)
    try:
        state = session.start()
        assert state["http_ready"] and state["projector"]["state"] == "waiting_for_runs"
        result = execute(cfg)
        assert result["status"] == "completed", result
        wait_for(lambda: session.status()["projector"].get("attempts") == 1)
        restored = execute(cfg, result["checkpoint"])
        assert restored["status"] == "completed", restored
        wait_for(lambda: session.status()["projector"].get("attempts") == 2)
        assert session.server.poll() is None and session.projector.poll() is None
    finally:
        session.close()
    assert session.server.poll() is not None and session.projector.poll() is not None


def test_occupied_port_and_startup_failure_cleanup(tmp_path):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        session = Session(tmp_path / "runs", port=occupied.getsockname()[1])
        with pytest.raises(ValueError, match="occupied"):
            session.start()
        assert session.server is None and session.projector is None
    session = Session(tmp_path / "runs", port=port(), max_bytes=1, headroom_bytes=0)
    with pytest.raises(RuntimeError, match="projector exited"):
        session.start()
    assert session.projector.poll() is not None
    assert session.server is None or session.server.poll() is not None


def test_view_failures_do_not_stop_independent_training(native_artifacts, tmp_path):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["run_root"] = str(tmp_path / "runs")
    cfg["output_ref"] = None
    cfg["training"].update(steps=12, validation_interval=4, checkpoint_interval=4)
    cfg["monitoring"] = {"scalar_interval": 1, "detail_interval": 100, "sample_prompts": 0}
    config = tmp_path / "train.json"
    config.write_text(json.dumps(cfg))
    session = Session(tmp_path / "runs", port=port(), max_seconds=90, headroom_bytes=0)
    process = None
    try:
        session.start()
        with (tmp_path / "trainer.log").open("wb") as log:
            process = spawn(
                [sys.executable, "-m", "ncm_lab", "generator", "train", "--config", str(config)], log
            )

            def progressed():
                return any(p.stat().st_size for p in (tmp_path / "runs").glob("generator/*/metrics.jsonl"))

            wait_for(progressed, 30)
            assert process.poll() is None
            # A stalled projector and stopped HTTP service are optional view failures.
            stop_owned(session.projector)
            stop_owned(session.server)
            process.wait(timeout=60)
        assert process.returncode == 0, (tmp_path / "trainer.log").read_text()
        summary = next((tmp_path / "runs").glob("generator/*/summary.json"))
        assert json.loads(summary.read_text())["optimizer_steps"] == 12
    finally:
        stop_owned(process)
        session.close()
