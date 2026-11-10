"""Raw-source work remains charged, finite, and separate from training budgets."""

from copy import deepcopy
from pathlib import Path
import pytest
from ncm_lab.block8.budgets import source_work_budget
from ncm_lab.block8.config import resolve
from ncm_lab.block8.runtime import Attempt, checked, plan
from ncm_lab.block8.workflow import run_workflow
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.io import canonical_hash, digest, read, write_json


def test_source_replays_reported_work_without_spending_a_training_stage_allowance(block8_fixture):
    settings = block8_fixture["settings"]
    before = deepcopy(settings)
    source = Attempt(block8_fixture["folder"] / "source replay", "source", settings)
    source.meter.charge(steps=392111, source=349289920, answer=199999655)
    source.meter.charge(answer=400000001)
    assert source.meter.used["answer_positions"] == 599999656
    cap = source.meter.budget.max_answer_positions
    with pytest.raises(BudgetStop, match="answer_positions"):
        source.meter.charge(answer=cap - source.meter.used["answer_positions"] + 1)
    receipt = read(checked(source.close("completed", {})))
    assert receipt["charged_work"]["answer_positions"] == 599999656
    resolved = read(source.path / "resolved_config.json")
    assert resolved["budget"]["max_answer_positions"] == cap
    assert resolved["budget"]["phase_id"] == source.meter.budget.phase_id
    prepare = settings["stage_budgets"]["prepare"]
    assert prepare["max_steps"] == settings["budget"]["max_steps"]
    assert prepare["max_answer_positions"] >= settings["data"]["max_parents"] * settings["data"]["max_row_bytes"]
    for stage in ("codec", "generator", "targets"):
        run = Attempt(block8_fixture["folder"] / (stage + " isolation"), stage, settings)
        assert run.meter.budget.max_answer_positions == settings["budget"]["max_answer_positions"]
        with pytest.raises(BudgetStop, match="answer_positions"):
            run.meter.charge(answer=run.meter.budget.max_answer_positions + 1)
        run.close("budget_stopped", {})
    assert settings == before


def test_source_aggregate_allowance_survives_new_attempts(block8_fixture):
    settings = block8_fixture["settings"]
    folder = block8_fixture["folder"] / "aggregate"
    cap = settings["stage_budgets"]["source"]["max_answer_positions"]
    for _ in range(settings["resources"]["max_attempts_per_stage"]):
        run = Attempt(folder, "source", settings)
        run.meter.charge(answer=cap)
        run.close("completed", {})
    run = Attempt(folder, "source", settings)
    with pytest.raises(BudgetStop, match="answer_positions"):
        run.meter.charge(answer=1)
    run.close("budget_stopped", {})


def test_offline_source_resolution_preserves_clocks_storage_and_larger_explicit_caps(block8_fixture):
    settings = block8_fixture["settings"]
    source = settings["stage_budgets"]["source"]
    for field, value in settings["budget"].items():
        if field not in {
            "max_steps", "max_phase_steps", "max_source_tokens", "max_phase_source_tokens",
            "max_answer_positions", "max_phase_answer_positions",
        }:
            assert source[field] == value
    assert source["enforce_wall_time_limits"] is True
    assert source["max_phase_answer_positions"] >= source["max_answer_positions"] * 3
    report = plan(settings)
    assert report["stage_budgets"]["source"] == source
    assert report["source_work_budget"]["overlap_reference_row_allowance"] == 200000
    assert report["research_work_started"] is False
    larger = deepcopy(settings)
    larger["budget"]["max_answer_positions"] = source["max_answer_positions"] * 2
    larger["budget"]["max_phase_answer_positions"] = source["max_answer_positions"] * 8
    budget, _ = source_work_budget(larger)
    assert budget["max_answer_positions"] == larger["budget"]["max_answer_positions"]
    assert budget["max_phase_answer_positions"] == larger["budget"]["max_phase_answer_positions"]


@pytest.mark.parametrize("key,value", [("max_source_rows", 0), ("max_row_chars", True), ("max_messages", 301)])
def test_invalid_raw_work_bounds_fail_offline(block8_fixture, key, value):
    settings = deepcopy(block8_fixture["settings"])
    settings["source"]["resolved_data"][key] = value
    with pytest.raises(ValueError, match="finite source-work bound"):
        source_work_budget(settings)


def test_long_rejected_answers_reproduce_old_cap_then_full_scan_completes(
    block8_fixture, tulu_fixture, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = block8_fixture
    long_row = deepcopy(tulu_fixture["rows"][0])
    long_row.update(id="overlength-answer", group_id="overlength-conversation")
    long_row["messages"][-1]["content"] *= 200
    asset = tulu_fixture["assets"][0]
    file = Path(tulu_fixture["data"]["data"]["assets_root"]) / asset["path"]
    pq.write_table(pa.Table.from_pylist(tulu_fixture["rows"][::2] + [long_row]), file)
    asset.update(size_bytes=file.stat().st_size, sha256=digest(file))
    write_json(tulu_fixture["lock"], dict(assets=tulu_fixture["assets"]))
    source_config = f["folder"] / "pinned raw-source regression.json"
    write_json(source_config, tulu_fixture["data"])
    cfg = read(f["config"])
    cfg["source"].update(
        config=str(source_config), max_candidates=40, max_source_index_bytes=2000000, max_candidate_bytes=1000000
    )
    cfg["budget"].update(max_answer_positions=1500, max_phase_answer_positions=4500)
    write_json(f["config"], cfg)
    settings = resolve(f["config"])
    assert settings["source_work_budget"]["overlap_reference_row_allowance"] == 4
    monkeypatch.setattr(
        "ncm_lab.teacher.model.load_original", lambda *a, **k: pytest.fail("preparation loaded backbone")
    )
    old = deepcopy(settings)
    old["stage_budgets"]["source"] = dict(old["budget"])
    old.pop("identity")
    old["identity"] = canonical_hash(old)
    failed = run_workflow(old, experiment_dir=f["folder"] / "old cap", execute=True, prepare_only=True)
    assert failed["status"] == "budget_stopped", failed
    assert "answer_positions" in failed["failure"]["error"]
    failed_manifest = Path(failed["experiment"])
    failure_sha = digest(failed_manifest)
    failed_receipt = checked(read(failed_manifest)["stages"]["source"]["receipt"])
    receipt_sha = digest(failed_receipt)
    old_prep = deepcopy(settings)
    old_prep["stage_budgets"]["prepare"] = dict(old_prep["budget"])
    old_prep.pop("identity")
    old_prep["identity"] = canonical_hash(old_prep)
    second_failure = run_workflow(
        old_prep, experiment_dir=f["folder"] / "old preparation cap", execute=True, prepare_only=True
    )
    assert second_failure["status"] == "budget_stopped"
    assert second_failure["failure"]["stage"] == "prepare"
    assert "answer_positions" in second_failure["failure"]["error"]
    result = run_workflow(settings, experiment_dir=f["folder"] / "finite source budget", execute=True, prepare_only=True)
    assert result["status"] == "completed", result
    state = read(result["experiment"])
    source = read(checked(state["stages"]["source"]["receipt"]))
    scan = read(Path(source["result"]["data"]["manifest"]).parent.parent / "scan.json")
    assert scan["complete"] and scan["rows_read"] == 51
    assert read(checked(result["dataset"]))["excluded"]["answer_length"] >= 1
    assert source["charged_work"]["answer_positions"] > 1500
    assert set(state["stages"]) == {"source", "prepare"}
    assert digest(failed_manifest) == failure_sha and digest(failed_receipt) == receipt_sha
