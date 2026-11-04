"""Stage 1 contracts and operational gates, never research-quality evidence."""

import json
import subprocess
import sys
import pytest
from ncm_lab.stage1.config import resolve, DataSettings
from ncm_lab.stage1.policy import Policy, codec_gate, GENERATOR_SELECTION
from ncm_lab.stage1.cli import parser
from ncm_lab.stage1.workflow import plan, load
from ncm_lab.runtime.io import read, digest


def metrics(**changes):
    return {**dict(loss=1.0, mse_full=1.0, mse_cutoff=1.0, head_kl=0.1), **changes}


def test_stage1_defaults_and_independent_precision():
    settings = resolve()
    assert settings["canvas"]["positions"] == 64 and settings["seed"] == 17
    stages = settings["stages"]
    assert [stages[s]["dtype"] for s in ("teacher", "codec", "targets", "generator")] == [
        "bfloat16",
        "float32",
        "float32",
        "bfloat16",
    ]
    assert (stages["codec"]["training"]["steps"], stages["generator"]["training"]["steps"]) == (1000, 500)
    assert (stages["codec"]["training"]["batch_size"], stages["codec"]["training"]["accumulation"]) == (4, 4)
    assert (
        stages["generator"]["training"]["batch_size"],
        stages["generator"]["training"]["accumulation"],
    ) == (1, 8)
    assert stages["generator"]["stage1"]["selection"] == GENERATOR_SELECTION
    assert stages["generator"]["monitoring"]["sample_prompts"] == 8
    assert all(not v["budget"]["enforce_wall_time_limits"] for v in stages.values())
    assert stages["generator"]["training"]["warmup_processed_tokens"] == 0
    assert settings["qualification"]["split_seed"] == 17
    assert resolve(seed=23)["qualification"] == settings["qualification"]


@pytest.mark.parametrize(
    "size", ["xsmall", "small", "medium", "large", "xlarge", "8", "32", "64", "256", "1024"]
)
def test_all_presets_share_production_settings(size):
    settings = resolve(canvas_size=size)
    for cfg in settings["stages"].values():
        a = cfg["architecture"]
        assert a["latent_width"] == 384 and len(a["resolutions"]) == 32
        assert (a["early_blocks"], a["hierarchy_blocks"], a["late_blocks"]) == (8, 16, 12)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"loss": 0.9, "mse_full": 0.9, "mse_cutoff": 0.9, "head_kl": 0.09}, "qualified"),
        ({}, "review_required"),
        ({"loss": 0.9, "head_kl": 0.2}, "review_required"),
        ({"loss": 0.9, "mse_full": float("nan")}, "invalid"),
    ],
)
def test_codec_gate_components_and_finite_override_boundary(changes, expected):
    result = codec_gate(metrics(), metrics(**changes), Policy())
    assert result["status"] == expected
    assert result["training_completed"] is True
    assert result["artifacts_valid"] == (expected != "invalid")
    assert result["codec_qualified_for_generator"] == (expected == "qualified")
    if expected != "invalid":
        assert set(result["component_deltas"]) == {"loss", "mse_full", "mse_cutoff", "head_kl"}


def test_planning_is_lightweight_and_preserves_typed_settings(tmp_path, stage1_fixture):
    settings = resolve(stage1_fixture["config"])
    assert settings["qualification"]["diagnostic"] is True
    result = plan(settings, tmp_path / "experiment path with spaces")
    path, state, restored = load(result["experiment"])
    assert restored == settings and state["status"] == "planned"
    assert not list(path.parent.glob("**/ledgers/*"))
    with pytest.raises(ValueError, match="must be empty"):
        plan(settings, path.parent)
    before = digest(path)
    assert load(path)[1] == state and digest(path) == before


def test_help_plan_import_no_model_or_server(tmp_path):
    code = """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','transformers','werkzeug'}: raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from ncm_lab.stage1.cli import main
assert main(['--plan-only']) == 0
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["research_work_started"] is False
    for name in ("run_stage1", "train_stage1_codec", "train_stage1_generator", "evaluate_stage1"):
        check = subprocess.run(
            [sys.executable, f"scripts/{name}.py", "--help"], capture_output=True, text=True, timeout=15
        )
        assert check.returncode == 0 and "--experiment" in check.stdout


def test_option_conflicts_and_reduced_data_label():
    with pytest.raises(SystemExit):
        parser().parse_args(["--experiment", "a", "--resume", "b"])
    with pytest.raises(SystemExit):
        parser().parse_args(["--execute", "--plan-only"])
    with pytest.raises(ValueError, match="diagnostic"):
        DataSettings(train_examples=32)
    assert read("configs/stage1/overfit.yaml")["qualification"]["diagnostic"] is True
    assert resolve("configs/stage1/overfit.yaml")["stages"]["generator"]["training"]["steps"] == 200


def test_existing_manifest_does_not_ignore_invalid_or_inapplicable_options(tmp_path, stage1_fixture):
    from ncm_lab.stage1.cli import main

    experiment = plan(resolve(stage1_fixture["config"]), tmp_path / "option guards")["experiment"]
    assert main(["--experiment", experiment, "--ar-max-new-tokens", "4"]) == 2
    assert main(["--experiment", experiment, "--codec-steps", "0"]) == 2
    assert main(["--experiment", experiment, "--ar-reference", "--extend-generator-steps", "5"]) == 2


def test_locked_test_access_prevents_extension_even_when_main_evaluation_pending(tmp_path, stage1_fixture):
    from ncm_lab.stage1.workflow import run
    from ncm_lab.runtime.io import atomic_json

    experiment = plan(resolve(stage1_fixture["config"]), tmp_path / "test access guard")["experiment"]
    path, state, _ = load(experiment)
    # Optional AR inference can observe locked prompts before NCM evaluation.
    state["locked_test_observed"] = True
    assert state["stages"]["evaluate"]["status"] == "pending"
    atomic_json(path, state)
    with pytest.raises(ValueError, match="before locked evaluation"):
        run(path, stage="generator", extend_generator_steps=5, reason="must remain forbidden")


def test_write_accounting_persists_pruned_and_failed_spending(tmp_path):
    from ncm_lab.stage1.io import WriteMeter
    from ncm_lab.runtime.budget import BudgetStop

    meter = WriteMeter(
        {"ledger": str(tmp_path / "writes.json"), "max_run_bytes": 100, "max_phase_bytes": 120}, "a"
    )
    path = tmp_path / "shard"
    with meter.account(path, 80):
        path.write_bytes(b"a" * 60)
    path.unlink()
    with pytest.raises(RuntimeError):
        with meter.account(path, 30):
            path.write_bytes(b"b" * 10)
            raise RuntimeError("injected interruption")
    record = read(tmp_path / "writes.json")
    assert record["written_bytes"] == 60 and sum(v["bytes"] for v in record["reservations"].values()) == 30
    other = WriteMeter(meter.settings, "b")
    with pytest.raises(BudgetStop):
        with other.account(path, 31):
            pytest.fail("aggregate I/O ceiling bypassed")
