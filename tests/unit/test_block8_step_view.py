import hashlib
import json
import os
import subprocess
import sys
import pytest
from ncm_lab.monitoring.block8 import StepView


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


@pytest.fixture
def observed(tmp_path):
    settings = tmp_path / "resolved_config.json"
    settings.write_text(json.dumps(dict(codec=dict(steps=20), training=dict(steps=25))))
    state = dict(
        schema="hierarchical_block8_experiment_v1",
        settings=dict(manifest=str(settings), sha256=hashlib.sha256(settings.read_bytes()).hexdigest()),
        status="running",
        stages=dict(source=dict(status="running")),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(state))
    return manifest, state


def append(path, step, *, complete=True, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(dict(step=step, losses=dict(text_ce=1.5), **fields)))
        if complete:
            stream.write("\n")


def test_idle_polling_is_silent_and_never_writes_to_the_experiment(observed):
    manifest, _ = observed
    before = {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}
    lines, clock = [], Clock()
    view = StepView(manifest, emit=lines.append, clock=clock)
    assert not view.poll()
    assert lines == ["stage source: running", "experiment: running"]
    lines.clear()
    clock.value = 100000
    for _ in range(10):
        assert not view.poll()
    assert lines == []
    assert before == {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}


def test_attach_shows_latest_step_then_follows_partial_records_and_ten_step_estimates(observed):
    manifest, _ = observed
    path = manifest.parent / "attempts/generator/first/metrics.jsonl"
    for step in range(1, 11):
        append(path, step)
    lines, clock = [], Clock()
    view = StepView(manifest, emit=lines.append, clock=clock)
    view.poll()
    assert sum("generator step" in line for line in lines) == 1
    assert "step 10/25" in lines[1] and "eta=pending" in lines[1]
    for step in range(11, 21):
        clock.value += 2
        append(path, step)
        view.poll()
    assert "avg_step=2.00s eta=0d 0h 1m estimate_step=20" in lines[-1]
    clock.value += 200
    append(path, 21)
    view.poll()
    assert "avg_step=2.00s eta=0d 0h 1m estimate_step=20" in lines[-1]
    count = len(lines)
    append(path, 22, complete=False)
    view.poll()
    assert len(lines) == count
    with path.open("a") as stream:
        stream.write("\n")
    view.poll()
    assert "step 22/25" in lines[-1]
    assert "estimate_source=viewer_observations" in lines[-1]


def test_resumed_attempt_keeps_its_real_steps_and_new_estimator(observed):
    manifest, state = observed
    first = manifest.parent / "attempts/generator/first/metrics.jsonl"
    append(first, 20)
    lines = []
    view = StepView(manifest, emit=lines.append)
    view.poll()
    second = manifest.parent / "attempts/generator/second/metrics.jsonl"
    append(second, 11)
    state.update(completed_steps=dict(generator=10))
    manifest.write_text(json.dumps(state))
    view.poll()
    assert "step 11/25" in lines[-2] and "attempt=second" in lines[-2]
    assert "eta=pending" in lines[-2]
    assert lines[-1] == "generator recovery checkpoint: step 10"


def test_first_batch_after_an_empty_file_cannot_create_a_zero_time_replay_estimate(observed):
    manifest, _ = observed
    path = manifest.parent / "attempts/generator/first/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.touch()
    lines, clock = [], Clock()
    view = StepView(manifest, emit=lines.append, clock=clock)
    view.poll()
    for step in range(1, 11):
        append(path, step)
    clock.value = 100
    view.poll()
    reports = [line for line in lines if "generator step" in line]
    assert len(reports) == 1
    assert "step 10/25" in reports[0] and "avg_step=pending eta=pending" in reports[0]


def test_oversized_records_and_truncated_history_stop_the_viewer(observed):
    manifest, _ = observed
    path = manifest.parent / "attempts/generator/first/metrics.jsonl"
    append(path, 1)
    view = StepView(manifest, emit=lambda _: None)
    view.poll()
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="shrank"):
        view.poll()
    path.write_bytes(b"x" * 131072)
    with pytest.raises(ValueError, match="byte allowance"):
        StepView(manifest, emit=lambda _: None).poll()


def test_recorded_trainer_estimate_is_used_instead_of_backlog_arrival_times(observed):
    manifest, state = observed
    path = manifest.parent / "attempts/generator/first/metrics.jsonl"
    append(path, 11, progress=dict(average_seconds_per_step=600, remaining_seconds=90000, estimate_step=10))
    lines = []
    view = StepView(manifest, emit=lines.append)
    view.poll()
    step = next(line for line in lines if "generator step" in line)
    assert "avg_step=600.00s eta=1d 1h 0m estimate_step=10" in step
    assert "viewer_observations" not in step
    state["status"] = "completed"
    manifest.write_text(json.dumps(state))
    assert view.poll()
    assert lines[-1] == "experiment: completed"


def test_read_budget_and_changed_settings_stop_only_the_viewer(observed):
    manifest, state = observed
    path = manifest.parent / "attempts/generator/first/metrics.jsonl"
    append(path, 1)
    with pytest.raises(ValueError, match="read allowance"):
        StepView(manifest, emit=lambda _: None, max_bytes=2).poll()
    settings = manifest.parent / "resolved_config.json"
    settings.write_text(settings.read_text() + " ")
    with pytest.raises(ValueError, match="settings receipt"):
        StepView(manifest)
    assert json.loads(manifest.read_text()) == state


def test_one_shot_viewer_imports_no_model_framework_and_starts_no_training(observed):
    manifest, _ = observed
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from ncm_lab.monitoring.block8 import main; "
         "assert main(sys.argv[1:]) == 0; assert 'torch' not in sys.modules", "--experiment", str(manifest), "--once"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "stage source: running" in result.stdout


def test_major_source_phases_follow_artifacts_without_counting_rows_or_repeating_status(observed):
    manifest, _ = observed
    attempt = manifest.parent / "attempts/source/first"
    attempt.mkdir(parents=True)
    (attempt / "resolved_config.json").write_text("{}")
    lines, clock = [], Clock()
    view = StepView(manifest, emit=lines.append, clock=clock)
    view.poll()
    assert any("checking source inputs and tokenizer" in line for line in lines)
    artifacts = attempt / "artifacts"
    artifacts.mkdir()
    index = artifacts / "preparation.sqlite"
    index.touch()
    view.poll()
    assert "building the source index" in lines[-1]
    count = len(lines)
    # Source work and elapsed time can increase without a new preparation phase.
    index.write_bytes(b"more source data")
    clock.value = 10000
    view.poll()
    assert len(lines) == count
    (artifacts / "candidates").mkdir()
    view.poll()
    assert "publishing qualified training and evaluation data" in lines[-1]
    (attempt / "qualification.json").write_text("{}")
    view.poll()
    assert "finalizing qualified source data" in lines[-1]
    prep_lines = [line for line in lines if line.startswith("data prep")]
    assert len(prep_lines) == 4
    assert not any(word in " ".join(prep_lines) for word in ("rows", "bytes", "eta=", "%"))
    before = {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}
    view.poll()
    assert before == {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "stage,index,marker,work,finish",
    [
        ("prepare", "dataset/parents.sqlite", "dataset/manifest.json", "eight-token blocks", "block dataset"),
        ("teacher", "cache/index.sqlite", "cache/manifest.json", "frozen teacher features", "teacher-feature cache"),
        ("targets", "cache/index.sqlite", "cache/manifest.json", "frozen latent targets", "target-code cache"),
    ],
)
def test_dataset_and_cache_milestones_are_reported_only_when_they_change(
    observed, stage, index, marker, work, finish
):
    manifest, state = observed
    state["stages"] = {stage: dict(status="running")}
    manifest.write_text(json.dumps(state))
    attempt = manifest.parent / "attempts" / stage / "first"
    attempt.mkdir(parents=True)
    (attempt / "resolved_config.json").write_text("{}")
    lines = []
    view = StepView(manifest, emit=lines.append)
    view.poll()
    index = attempt / index
    index.parent.mkdir()
    index.touch()
    view.poll()
    assert work in lines[-1]
    count = len(lines)
    for _ in range(3):
        view.poll()
    assert len(lines) == count
    (attempt / marker).write_text("{}")
    view.poll()
    assert "finalizing" in lines[-1] and finish in lines[-1]
    state["stages"][stage]["status"] = "completed"
    manifest.write_text(json.dumps(state))
    view.poll()
    assert lines[-1] == f"stage {stage}: completed"
    assert sum(line.startswith("data prep") for line in lines) == 3


def test_retry_waits_for_the_new_attempt_and_does_not_reuse_old_completed_markers(observed):
    manifest, state = observed
    first = manifest.parent / "attempts/source/first"
    first.mkdir(parents=True)
    config = first / "resolved_config.json"
    config.write_text("{}")
    os.utime(config, ns=(100, 100))
    (first / "qualification.json").write_text("{}")
    state["attempts"] = dict(source=2)
    manifest.write_text(json.dumps(state))
    lines = []
    view = StepView(manifest, emit=lines.append)
    view.poll()
    assert not any(line.startswith("data prep") for line in lines)
    second = first.with_name("second")
    second.mkdir()
    (second / "resolved_config.json").write_text("{}")
    os.utime(second / "resolved_config.json", ns=(200, 200))
    view.poll()
    assert "checking source inputs and tokenizer" in lines[-1]
    assert "attempt=second" in lines[-1]
    count = len(lines)
    view.poll()
    assert len(lines) == count
