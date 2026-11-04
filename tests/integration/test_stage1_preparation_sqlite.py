"""Fresh Stage 1 Parquet preparation and genuine filesystem failure isolation."""

from pathlib import Path
import pytest
from ncm_lab.data.artifacts import Dataset
from ncm_lab.runtime.io import read, digest
from ncm_lab.stage1.config import resolve
from ncm_lab.stage1.workflow import plan, run, load, checked, verify_record


def test_fresh_prepare_publishes_verified_artifacts_without_downstream_work(tmp_path, stage1_oasst_fixture):
    fixture = stage1_oasst_fixture
    experiment = Path(plan(
        resolve(fixture["config"], canvas_size="xsmall"), tmp_path / "fresh experiment with spaces"
    )["experiment"])
    assert not (experiment.parent / "native_runs").exists()
    before = digest(fixture["oasst"]["path"])
    result = run(experiment, stage="prepare")
    assert result["status"] == "ready", result
    path, state, _ = load(experiment)
    record = read(checked(state["stages"]["prepare"]["receipt"]))
    verify_record("prepare", record, path)
    prepared = record["result"]
    native_run = Path(prepared["path"])
    assert (native_run / "artifacts/source-index.sqlite").is_file()
    assert prepared["model_loaded"] is False
    assert prepared["origin"] == "software_fixture"
    assert read(native_run / "manifest.json")["status"] == "completed"
    assert (path.parent / "handoffs/prepare.json").is_file()
    qualification = read(checked(prepared["qualification"]))
    assert qualification["status"] == "qualified" and qualification["diagnostic"]
    assert qualification["source_rows_inspected"] == len(fixture["oasst"]["rows"])
    groups = []
    for split, count in [("train", 8), ("dev", 4), ("test", 4)]:
        assert qualification["selected"][split]["examples"] == count
        groups.append(set(qualification["selected"][split]["groups"]))
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    data, locked = Dataset(prepared["artifact"]), Dataset(checked(prepared["evaluation_data"]))
    assert len(data) == 12 and len(locked) == 4
    for dataset in (data, locked):
        for i in range(len(dataset)):
            example, _ = dataset[i]
            assert example.eos_genuine and not example.censored and example.answer_ids[-1] == 2
    assert all(
        entry["status"] == "pending" and not entry["attempts"]
        for name, entry in state["stages"].items() if name != "prepare"
    )
    assert {p.name for p in (path.parent / "native_runs").iterdir()} == {"stage1_prepare", "ledgers"}
    assert digest(fixture["oasst"]["path"]) == before


# Test-only child shim: inject failures at the actual native Run's index path,
# preserving the real worker, controller, saved traceback, exit code and manifests.
INJECT_INDEX_FAILURE = """
import sqlite3
import sys
from pathlib import Path
from ncm_lab.runtime.io import disk_path
from ncm_lab.runtime.provenance import Run
from ncm_lab.stage1.worker import main

mode = sys.argv[1]
config_path = sys.argv[2]
original_start = Run.start
def start(run):
    result = original_start(run)
    parent = disk_path(run.path / "artifacts")
    if mode == "parent_file":
        parent.write_text("preserve partial diagnostic", encoding="utf-8")
    elif mode == "mkdir":
        original_mkdir = Path.mkdir
        def mkdir(path, *args, **kwargs):
            if path == parent:
                raise PermissionError("injected index parent denial")
            return original_mkdir(path, *args, **kwargs)
        Path.mkdir = mkdir
    elif mode == "connect":
        original_connect = sqlite3.connect
        def connect(path, *args, **kwargs):
            if Path(path) == parent / "source-index.sqlite":
                raise sqlite3.OperationalError("injected index I/O failure")
            return original_connect(path, *args, **kwargs)
        sqlite3.connect = connect
    return result
Run.start = start
sys.argv = ["worker", "--config", config_path]
raise SystemExit(main())
"""


@pytest.mark.parametrize("operation", ["parent_file", "mkdir", "connect"])
def test_index_failure_is_nonzero_retained_and_stops_all_dependencies(
    tmp_path, stage1_oasst_fixture, monkeypatch, capsys, operation
):
    import ncm_lab.stage1.workflow as flow
    from ncm_lab.stage1.cli import main

    fixture = stage1_oasst_fixture
    experiment = Path(plan(
        resolve(fixture["config"], canvas_size="xsmall"), tmp_path / "failed experiment"
    )["experiment"])
    original_popen = flow.subprocess.Popen

    def launch(command, **kwargs):
        assert command[1:4] == ["-u", "-m", "ncm_lab.stage1.worker"]
        assert command[4] == "--config"
        return original_popen(
            [command[0], "-u", "-c", INJECT_INDEX_FAILURE, operation, command[5]], **kwargs
        )

    monkeypatch.setattr(flow.subprocess, "Popen", launch)
    assert main(["--experiment", str(experiment), "--execute"]) == 2
    path, state, _ = load(experiment)
    assert state["status"] == state["stages"]["prepare"]["status"] == "failed"
    entry = state["stages"]["prepare"]
    assert "receipt" not in entry and len(entry["attempts"]) == 1
    attempt = Path(entry["attempts"][0]["path"])
    assert read(attempt / "exit.json")["exit_status"] == 2
    result = read(attempt / "result.json")["result"]
    native_run = Path(result["path"])
    index = native_run / "artifacts/source-index.sqlite"
    operation_name = "connection" if operation == "connect" else "parent directory creation"
    assert str(index) in result["error"] and operation_name in result["error"]
    failure = read(native_run / "failure.json")
    assert str(index) in failure["error"]
    assert "direct cause" in failure["traceback"]
    causes = {
        "connect": ("OperationalError",), "mkdir": ("PermissionError",),
        "parent_file": ("FileExistsError", "NotADirectoryError"),
    }[operation]
    assert any(cause in failure["traceback"] for cause in causes)
    assert read(native_run / "manifest.json")["status"] == "failed"
    assert not (path.parent / "handoffs/prepare.json").exists()
    assert not (native_run / "artifacts/data/manifest.json").exists()
    assert not index.exists()
    if operation == "parent_file":
        assert index.parent.read_text(encoding="utf-8") == "preserve partial diagnostic"
    assert all(
        other["status"] == "pending" and not other["attempts"]
        for name, other in state["stages"].items() if name != "prepare"
    )
    report = read(path.parent / "reports/stage1_summary.json")
    assert report["workflow_completed"] is False
    assert report["failure"]["stage"] == "prepare"
    assert "Stage 1 complete." not in capsys.readouterr().out
    assert digest(fixture["oasst"]["path"]) == fixture["oasst"]["settings"]["source_sha256"]
