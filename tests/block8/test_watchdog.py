import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ncm_lab.block8 import watchdog
from ncm_lab.block8.config import resolve
from ncm_lab.block8.runtime import checked
from ncm_lab.monitoring.block8 import StepView
from ncm_lab.runtime.io import atomic_json, read, root, write_json


def test_full_block8_defaults_to_progress_based_supervision():
    settings = resolve()
    assert settings["resources"]["max_stall_seconds"] == 14400
    for budget in (settings["budget"], *settings["stage_budgets"].values()):
        assert budget["enforce_wall_time_limits"] is False
        assert budget["max_phase_steps"] >= budget["max_steps"] > 0
        assert budget["max_phase_storage_bytes"] >= budget["max_storage_bytes"] > 0


def no_lifetime_limit(fixture):
    config = read(fixture["config"])
    config["budget"]["enforce_wall_time_limits"] = False
    config["resources"]["max_stall_seconds"] = 1
    write_json(fixture["config"], config)
    return resolve(fixture["config"])


def options(fixture, name):
    return dict(
        execute=True,
        prepare_only=True,
        experiment_dir=fixture["folder"] / name,
        resume=None,
        data_manifest=fixture["data"],
        evaluation_manifest=fixture["evaluation"],
    )


def test_normal_cli_prepares_a_dataset_in_a_supervised_worker(block8_fixture):
    f = block8_fixture
    config = read(f["config"])
    config["budget"]["enforce_wall_time_limits"] = False
    write_json(f["config"], config)
    experiment = f["folder"] / "supervised dataset"
    result = subprocess.run(
        [
            sys.executable, "-u", str(root() / "scripts/run_block8.py"),
            "--execute", "--prepare-only", "--config", str(f["config"]),
            "--experiment-dir", str(experiment), "--data-manifest", str(f["data"]),
            "--evaluation-manifest", str(f["evaluation"]),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = read(experiment / "manifest.json")
    receipt = read(checked(state["stages"]["prepare"]["receipt"]))
    assert receipt["status"] == "completed"
    assert receipt["wall_policy"]["numeric_wall_limits_active"] is False
    assert receipt["charged_work"]["steps"] == len(f["parents"])
    assert checked(receipt["result"]["dataset"]).is_file()


@pytest.fixture
def fork_worker(monkeypatch):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("forked fault injection requires POSIX")
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(watchdog.multiprocessing, "get_context", lambda _: context)
    return context


def test_charged_work_extends_supervision_past_the_first_interval(block8_fixture, fork_worker, monkeypatch):
    f = block8_fixture
    settings = no_lifetime_limit(f)
    advance = fork_worker.Event()
    snapshot = watchdog.progress_snapshot

    def worker(sender, stop, _settings, args):
        with sender:
            file = args["experiment_dir"] / "ledgers" / "work.json"
            atomic_json(file, {"charges": {"steps": 1}})
            if advance.wait(timeout=5) and not stop.is_set():
                sender.send({"status": "completed"})

    def observe(manifest):
        result = snapshot(manifest)
        if result[1]:
            advance.set()
        return result

    monkeypatch.setattr(watchdog, "_worker", worker)
    monkeypatch.setattr(watchdog, "progress_snapshot", observe)
    assert watchdog.supervise(settings, options(f, "progressing"), timeout=1)["status"] == "completed"


def test_idle_worker_closes_its_attempt_at_the_next_work_boundary(block8_fixture, fork_worker, monkeypatch):
    f = block8_fixture
    settings = no_lifetime_limit(f)
    from ncm_lab.block8.workflow import run_workflow

    def worker(sender, stop, settings, args):
        with sender:
            if stop.wait(timeout=5):
                sender.send(run_workflow(settings, **args, stop_requested=stop.is_set))

    monkeypatch.setattr(watchdog, "_worker", worker)
    result = watchdog.supervise(settings, options(f, "idle cooperative"), timeout=1)
    assert result["status"] == "budget_stopped", result
    receipt = read(checked(result["failure"]["receipt"]))
    assert "no progress" in receipt["result"]["error"]
    assert receipt["wall_policy"]["phase"]["incomplete_unbounded_attempts"] == []
    assert not (Path(result["experiment"]).parent / "writer.lock").exists()


def test_unresponsive_worker_is_stopped_and_viewer_reports_its_snapshot(block8_fixture, fork_worker, monkeypatch):
    f = block8_fixture
    settings = no_lifetime_limit(f)
    monkeypatch.setattr(watchdog, "STOP_GRACE_SECONDS", 0.1)
    monkeypatch.setattr("ncm_lab.block8.workflow.prepare", lambda *_: time.sleep(30))
    result = watchdog.supervise(settings, options(f, "idle unresponsive"), timeout=1)
    assert result["status"] == "budget_stopped", result
    assert "unresponsive worker" in result["failure"]["error"]
    manifest = Path(result["experiment"])
    report = read(checked(result["failure"]["receipt"]))
    assert report["child_exit_status"] != 0
    assert read(manifest)["stages"]["prepare"]["status"] == "running"
    lines = []
    assert StepView(manifest, emit=lines.append).poll()
    assert "watchdog stopped" in lines[-1]
