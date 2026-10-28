from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import pytest
from ncm_lab.experiments.registry import registry, plan
from ncm_lab.evaluation.analysis import summarize, paired_group_interval
from ncm_lab.runtime.budget import Budget, Meter, BudgetExceeded
from ncm_lab.runtime.io import root, read, write_json
from ncm_lab.runtime.provenance import validate_run
from ncm_lab.training.loop import select_checkpoint


def test_every_protocol_and_documented_command_resolves_offline(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request,"urlopen",lambda *a,**kw: pytest.fail("planning attempted network"))
    all_experiments = registry()
    assert len(all_experiments) == 12
    assert "ACL-005" in all_experiments
    for experiment_id in all_experiments:
        assert plan(experiment_id,"spark")["research_run_started"] is False
    for command in (["--help"],["data","--help"],["experiment","--help"],["results","--help"],["publication","--help"]):
        result = subprocess.run([sys.executable,"-m","ncm_lab",*command],capture_output=True,text=True)
        assert result.returncode == 0,result.stderr


def test_budget_limits_and_finite_caps(tmp_path):
    b = Budget(**read(root()/"configs/budgets/cpu_smoke.yaml"))
    with pytest.raises(ValueError):
        replace(b,max_wall_seconds=float("inf"))
    meter = Meter(b,tmp_path)
    with pytest.raises(BudgetExceeded):
        meter.check(source_tokens=b.max_training_source_token_passes+1)
    meter.steps = b.max_steps
    with pytest.raises(BudgetExceeded):
        meter.check()


def test_failures_in_denominator_and_actual_elapsed_time():
    rows = [{"correct":True,"error":None,"request_seconds":1},
            {"correct":False,"error":"timeout","request_seconds":10}]
    result = summarize(rows,elapsed_seconds=10)
    assert result["requests"] == 2 and result["accuracy"] == .5
    assert result["correct_requests_per_hour"] == 360
    assert result["p95_seconds"] == 10
    assert paired_group_interval([{"group_id":"a","difference":1}])["ci95"] is None


def test_checkpoint_selection_dev_only_and_deterministic():
    records = [{"split":"dev","dev_loss":1.0,"step":2,"artifact_id":"b"},
               {"split":"dev","dev_loss":1.0,"step":1,"artifact_id":"a"}]
    assert select_checkpoint(records)["artifact_id"] == "a"
    with pytest.raises(ValueError):
        select_checkpoint([{**records[0],"split":"test"}])


@pytest.mark.integration
def test_cpu_integration_immutable_runs_and_fixture_exclusion(tmp_path):
    from ncm_lab.experiments.runner import run
    config = read(root()/"configs/smoke.yaml")
    config["artifact_root"] = str(tmp_path/"artifacts")
    config_path = tmp_path/"config.yaml"
    write_json(config_path,config)
    result = run("SHR-001",config_path,root()/"configs/budgets/cpu_smoke.yaml")
    assert result["status"] == "completed",result
    path = Path(result["path"])
    manifest = validate_run(path)
    assert manifest["result_origin"] == "test_fixture" and not manifest["publication_eligible"]
    metrics = read(path/"metrics.json")
    assert metrics["training"]["optimizer_steps"] == 2
    assert metrics["stage_lengths"] == [5,5,9,16,16]
    (path/"predictions.jsonl").write_text("tampered")
    with pytest.raises(ValueError,match="changed"):
        validate_run(path)
    from ncm_lab.reporting.reports import run_paths
    assert path/'manifest.json' not in run_paths()
