"""Console/process-control fixtures; no model allocation or real clock waiting."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ncm_lab.stage1.workflow as flow
from ncm_lab.runtime.io import atomic_json, digest, read, reference, write_json
from ncm_lab.stage1.config import resolve


@pytest.fixture
def child_harness(tmp_path, monkeypatch):
    import ncm_lab.runtime.pilot as pilot

    def make(
        stage="generator",
        chunks=(),
        trailing=b"",
        status="completed",
        exit_code=0,
        limit=1000000,
        interrupt_at=None,
        checkpoint=False,
    ):
        folder = tmp_path / "experiment"
        folder.mkdir()
        native = folder / "native_runs/mock-worker"
        write_json(native / "manifest.json", {"software_fixture": True, "status": status})
        h = SimpleNamespace(
            elapsed=0,
            sleeps=[],
            polls=0,
            resource_checks=0,
            stopped=[],
            captures=[],
            result={
                "status": status,
                "path": str(native),
                "elapsed_seconds": 12.5,
                "diagnostic": "actual worker result fixture",
            },
        )
        h.path = folder / "manifest.json"
        h.attempt = folder / "stages" / stage / "mock-attempt"
        h.value = {
            "stage1_context": {"stage": stage},
            "training": {"steps": 100, "batch_size": 1, "accumulation": 8},
            "dtype": "float32" if stage == "codec" else "bfloat16",
            "output_ref": str(folder / "handoffs" / (stage + ".json")),
        }
        h.state = {"repository_root": str(tmp_path), "stages": {stage: {}}}
        h.settings = {"canvas": {"positions": 64}, "resources": {"max_controller_bytes": limit}}
        h.pointer = None
        if checkpoint:
            point = native / "mock-checkpoint.json"
            write_json(point, {"software_fixture": True})
            h.pointer = reference(point)

        class Child:
            pid = 123456  # No OS process is created or signalled.
            returncode = None

            def __init__(self, command, **kwargs):
                h.command, h.options, h.child = command, kwargs, self
                assert kwargs["shell"] is False

            def poll(self):
                h.elapsed += 900  # Many former heartbeat intervals, without waiting.
                index = h.polls
                h.polls += 1
                if index == interrupt_at:
                    raise KeyboardInterrupt
                if index < len(chunks):
                    h.options["stderr"].write(chunks[index])
                    h.options["stderr"].flush()
                    return None
                h.options["stderr"].write(trailing)
                h.options["stderr"].flush()
                h.options["stdout"].write(json.dumps(h.result).encode())
                h.options["stdout"].flush()
                self.returncode = exit_code
                return exit_code

        def sleep(seconds):
            h.sleeps.append(seconds)
            h.elapsed += seconds

        def stop(child):
            assert child is h.child
            h.stopped.append(child)
            if child.returncode is None:
                child.returncode = -15

        def capture(*args):
            h.captures.append(args)
            return h.pointer

        original_glob = Path.rglob

        def scan(path, pattern):
            if path == folder / "stages":
                h.resource_checks += 1
            return original_glob(path, pattern)

        monkeypatch.setattr(flow, "time", SimpleNamespace(monotonic=lambda: h.elapsed, sleep=sleep))
        monkeypatch.setattr(flow.subprocess, "Popen", Child)
        monkeypatch.setattr(pilot, "stop_child", stop)
        monkeypatch.setattr(flow, "capture_checkpoint", capture)
        monkeypatch.setattr(Path, "rglob", scan)
        h.execute = lambda: flow.execute_child(h.value, h.attempt, h.path, h.state, h.settings)
        return h

    return make


@pytest.mark.parametrize("stage", flow.STAGES)
def test_idle_child_timer_advance_is_silent_with_polling_backoff_and_checks(child_harness, capsys, stage):
    h = child_harness(stage=stage, chunks=[b""] * 8)
    record = h.execute()
    out, err = capsys.readouterr()
    target = "100" if stage in {"codec", "generator"} else "bounded preprocessing/evaluation"
    assert out == (
        f"Stage {stage}: canvas=64 target={target} batch=1 accumulation=8 "
        f"dtype={h.value['dtype']} clocks=disabled\nLogs: {h.attempt}\nHandoff: {h.value['output_ref']}\n"
    )
    assert err == "" and h.elapsed > 7200
    assert h.polls == 9 and h.sleeps == [0.2] * 8 and h.resource_checks == 8
    assert h.stopped == [h.child] and len(h.captures) == 1
    assert read(h.attempt / "process.json")["execution_deadline"] is None
    assert record["result"] == h.result and record["exit_status"] == 0
    assert record["run"]["sha256"] == digest(Path(h.result["path"]) / "manifest.json")
    assert read(h.attempt / "stdout.json") == h.result
    assert (h.attempt / "stderr.log").read_bytes() == b""


def test_real_events_duplicates_partial_reads_and_unterminated_trailing_error_survive(child_harness, capsys):
    chunks = [
        b"",
        b"generator step 10/100 loss=3.1\n",
        b"validation dev_loss=3.0\n",
        b"checkpoint completed\n",
        b"checkpoint completed\n",
        b"warning: quoted process active text is worker output\n" * 1600,
        b"",
        b"Stage completed; artifact saved\n",
    ]
    trailing = b"warning: final worker diagnostic without newline"
    h = child_harness(chunks=chunks, trailing=trailing)
    record = h.execute()
    captured = capsys.readouterr()
    expected = b"".join(chunks) + trailing
    assert captured.err == expected.decode()
    assert (h.attempt / "stderr.log").read_bytes() == expected
    assert captured.out.count("Stage generator:") == 1
    assert record["result"] == h.result  # Includes elapsed observations, unfiltered.
    assert read(h.attempt / "stdout.json") == h.result


def test_trailing_failure_and_exit_status_are_preserved(child_harness, capsys):
    h = child_harness(chunks=[b""], trailing=b"error: actual diagnostic", status="failed", exit_code=2)
    record = h.execute()
    assert capsys.readouterr().err == "error: actual diagnostic"
    assert record["result"] == h.result and record["exit_status"] == 2 and record["run"] is None
    assert read(h.attempt / "exit.json")["exit_status"] == 2
    assert (h.attempt / "stderr.log").read_bytes() == b"error: actual diagnostic"


@pytest.mark.parametrize("interruption", ["cancel", "resource"])
def test_owned_cleanup_checkpoint_capture_and_resource_limits_survive(child_harness, capsys, interruption):
    h = child_harness(
        chunks=[b"worker observed work\n"] * 3,
        checkpoint=True,
        interrupt_at=2 if interruption == "cancel" else None,
        limit=1000000 if interruption == "cancel" else 1,
    )
    error = KeyboardInterrupt if interruption == "cancel" else flow.BudgetStop
    with pytest.raises(error):
        h.execute()
    assert h.stopped == [h.child] and len(h.captures) == 1
    assert h.resource_checks > 0
    assert h.sleeps == ([0.2] * 2 if interruption == "cancel" else [])
    assert read(h.attempt / "exit.json")["exit_status"] == -15
    assert read(h.path)["stages"]["generator"]["last_checkpoint"] == h.pointer
    assert capsys.readouterr().err.startswith("worker observed work\n")


@pytest.mark.parametrize("status", ["failed", "blocked", "cancelled", "budget_stopped"])
def test_failure_outcome_stops_all_route_and_cli_returns_nonzero(tmp_path, monkeypatch, capsys, status):
    from ncm_lab.stage1.cli import main

    path = Path(flow.plan(resolve(), tmp_path / "outcome fixture")["experiment"])
    calls = []

    def child(value, *args):
        calls.append(value["stage1_context"]["stage"])
        attempt = args[0]
        attempt.mkdir(parents=True)
        return {"result": {"status": status, "error": "preserved actionable diagnostic"}, "exit_status": 2}

    monkeypatch.setattr(flow, "execute_child", child)
    monkeypatch.setattr(flow, "reports", lambda *args: {})
    assert main(["--experiment", str(path), "--stage", "all", "--execute"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == status and result["failure"]["error"] == "preserved actionable diagnostic"
    assert calls == ["prepare"]
    assert all(s == "pending" for name, s in result["stages"].items() if name != "prepare")


def test_review_outcome_remains_actionable_and_blocks_dependencies(tmp_path, monkeypatch, capsys):
    from ncm_lab.stage1.cli import main

    path = Path(flow.plan(resolve(), tmp_path / "review fixture")["experiment"])
    _, state, _ = flow.load(path)
    for name in ("prepare", "teacher", "codec"):
        receipt = path.parent / (name + "-mock-receipt.json")
        write_json(
            receipt, {"result": {"stage1_diagnostics": {"codec_qualification": {"artifacts_valid": True}}}}
        )
        state["stages"][name].update(
            status="review_required" if name == "codec" else "completed", receipt=reference(receipt)
        )
    atomic_json(path, state)
    monkeypatch.setattr(flow, "verify_record", lambda *args: {})
    monkeypatch.setattr(flow, "ensure_resources", lambda *args: None)
    monkeypatch.setattr(flow, "reports", lambda *args: {})
    monkeypatch.setattr(flow, "execute_child", lambda *args: pytest.fail("review started a child"))
    assert main(["--experiment", str(path), "--stage", "all", "--execute"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "review_required" and "inspect its report" in result["failure"]
    assert (
        result["stages"]["targets"]
        == result["stages"]["generator"]
        == result["stages"]["evaluate"]
        == "pending"
    )


def test_implementation_change_is_rejected_without_rewriting_existing_experiment(tmp_path, monkeypatch):
    path = Path(flow.plan(resolve(), tmp_path / "identity fixture")["experiment"])
    files = [path, path.parent / "plan.json", path.parent / "configs/proposal.json"]
    before = {p: digest(p) for p in files}
    monkeypatch.setattr(flow, "implementation_hash", lambda: "0" * 64)
    with pytest.raises(ValueError, match="implementation identity changed"):
        flow.load(path)
    assert all(digest(p) == sha for p, sha in before.items())
    assert not (path.parent / "native_runs").exists()
