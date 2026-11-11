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
from ncm_lab.runtime.io import atomic_json, digest, read, root, write_json
from scripts import run_block8 as launcher


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


def test_resume_launcher_waits_for_owner_then_reuses_sealed_preparation(block8_fixture, monkeypatch, capsys):
    from ncm_lab.block8.workflow import run_workflow

    f = block8_fixture
    result = run_workflow(f["settings"], **options(f, "queued resume"))
    assert result["status"] == "completed", result
    manifest = Path(result["experiment"])
    original = manifest.read_bytes()
    lock = manifest.with_name("writer.lock")
    lock.write_text("existing writer\n")
    waits = []

    def release_after_wait(seconds):
        waits.append(seconds)
        assert lock.read_text() == "existing writer\n"
        lock.unlink()

    monkeypatch.setattr(launcher, "sleep", release_after_wait)
    assert launcher.main(["--resume", str(manifest), "--execute", "--prepare-only"]) == 0
    assert len(waits) == 1
    assert "Waiting for the current Block8 writer" in capsys.readouterr().err
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("forced_stop", [False, True])
def test_queued_resume_reports_incomplete_preparation(tmp_path, monkeypatch, capsys, forced_stop):
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "writer.lock"
    write_json(manifest, {"stages": {"prepare": {"status": "running"}}})
    lock.touch()
    if forced_stop:
        report = tmp_path / "watchdog" / f"{digest(manifest)}.json"
        write_json(report, {"error": "Block8 watchdog stopped an unresponsive worker"})
        monkeypatch.setattr(launcher, "sleep", lambda _: pytest.fail("waited after watchdog stop"))
        error = "watchdog stopped an unresponsive worker"
    else:
        monkeypatch.setattr(launcher, "sleep", lambda _: lock.unlink())
        error = "before preparation completed"
    monkeypatch.setattr(launcher, "dispatch", lambda _: pytest.fail("resumed incomplete preparation"))
    assert launcher.main(["--resume", str(manifest), "--execute"]) == 2
    assert error in capsys.readouterr().out
    assert lock.exists() == forced_stop


def test_queued_generation_resumes_without_a_preparation_stage(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "writer.lock"
    write_json(manifest, {"generation_export": {"manifest": "export.json"}})
    lock.touch()
    monkeypatch.setattr(launcher, "sleep", lambda _: lock.unlink())
    monkeypatch.setattr(launcher, "dispatch", lambda _: {"status": "completed"})
    assert launcher.main(["--resume", str(manifest), "--execute"]) == 0


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
