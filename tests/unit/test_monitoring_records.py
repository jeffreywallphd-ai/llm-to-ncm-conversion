import json
import time
from types import SimpleNamespace
import pytest
import torch
from ncm_lab.monitoring.records import Monitoring, Measure, validate_record
from ncm_lab.monitoring.recorder import Recorder
from ncm_lab.monitoring.diagnostics import gradients
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.runtime.io import write_json


def test_strict_nonfinite_records_and_diagnostic_event(tmp_path):
    value = {
        "architecture": tiny_architecture().to_dict(),
        "budget": {"finite": "fixture"},
        "model": {"fixture": True},
    }
    write_json(tmp_path / "resolved_config.json", value)
    run = SimpleNamespace(
        path=tmp_path,
        value=value,
        id="record-test",
        stage="codec",
        meter=None,
        resolved={},
        started=time.perf_counter(),
    )
    recorder = Recorder(run)
    recorder.metric({"bad": Measure(float("inf")), "empty": Measure(None, 0), "zero": Measure(0.0)})
    row = json.loads(
        (tmp_path / "monitor_metrics.jsonl").read_text(), parse_constant=lambda x: pytest.fail(x)
    )
    validate_record(row)
    assert row["metrics"]["bad"]["value"] is None and row["metrics"]["bad"]["status"] == "nonfinite"
    assert row["metrics"]["empty"]["status"] == "unavailable" and row["metrics"]["zero"]["value"] == 0
    event = json.loads((tmp_path / "monitor_events.jsonl").read_text())
    assert event["status"] == "nonfinite_observation" and event["sequence"] == 1
    recorder.metric({"next": Measure(2.0)})
    rows = [json.loads(x) for x in (tmp_path / "monitor_metrics.jsonl").read_text().splitlines()]
    assert [x["sequence"] for x in rows] == [1, 2]
    with pytest.raises(ValueError, match="bounded size"):
        recorder.append("samples", text="x" * 1048576)
    assert not (tmp_path / "monitor_samples.jsonl").exists()
    with pytest.raises(ValueError, match="schema"):
        validate_record({"schema": "old drafter"})


@pytest.mark.parametrize(
    "options",
    [
        {"scalar_interval": 0},
        {"scalar_seconds": float("inf")},
        {"histograms": True},
        {"sample_prompts": 5},
        {"enabled": "yes"},
    ],
)
def test_monitoring_configuration_has_finite_bounds(options):
    with pytest.raises(ValueError):
        Monitoring(**options)


def test_gradient_statuses_are_distinct_and_aliases_count_once():
    model = torch.nn.Module()
    model.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.absent = torch.nn.Parameter(torch.ones(2))
    model.zero = torch.nn.Parameter(torch.ones(2))
    model.zero.grad = torch.zeros(2)
    model.bad = torch.nn.Parameter(torch.ones(2))
    model.bad.grad = torch.tensor([float("inf"), 0.0])
    model.alias = model.zero
    rows = gradients(model)
    assert len(rows) == 4
    assert rows["gradients/frozen/combined_l2"].json()["status"] == "frozen"
    assert rows["gradients/absent/combined_l2"].json()["status"] == "absent"
    assert rows["gradients/zero/combined_l2"].json()["status"] == "finite_zero"
    assert rows["gradients/bad/combined_l2"].json()["status"] == "nonfinite"


@pytest.mark.parametrize("limit", ["nan", "inf", "-1"])
def test_view_lifetimes_are_finite_before_children_launch(tmp_path, limit, monkeypatch):
    from ncm_lab.cli import main
    from ncm_lab.monitoring.projector import main as project
    import subprocess

    def forbidden(*a, **kw):
        raise AssertionError("invalid budget launched a child")

    monkeypatch.setattr(subprocess, "run", forbidden)
    args = [
        "--attempt-dir",
        str(tmp_path),
        "--runs-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "new"),
        "--max-seconds",
        limit,
    ]
    assert main(["monitor", "replay", *args]) == 2
    with pytest.raises(SystemExit):
        project(args)
    assert not (tmp_path / "new").exists()
