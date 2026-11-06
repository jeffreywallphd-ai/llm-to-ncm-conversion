"""Stage 3 fixture evidence only: no production assets, training, or servers."""

import json
from pathlib import Path
import pytest
import torch
from ncm_lab.runtime.io import read, write_json, root, reference, digest
from ncm_lab.stage3.config import resolve, STAGES
from ncm_lab.stage3.launcher import main
from ncm_lab.stage1.workflow import plan, run, load, checked, configuration


@pytest.fixture
def stage3_fixture(stage1_fixture):
    source = stage1_fixture["source"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["prompt"] = f"token4 token{i % 20 + 4} token{i // 20 + 4}"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    source_config = Path(stage1_fixture["experiment"]["stages"]["prepare"])
    source_value = read(source_config)
    source_value["data"]["source_sha256"] = digest(source)
    write_json(source_config, source_value)
    base = read(root() / "configs/stage3/experiment.yaml")
    base["qualification"] = {**stage1_fixture["experiment"]["qualification"], "minimum_train_examples": 7}
    base["resources"] = stage1_fixture["experiment"]["resources"]
    base["canvas_size"] = "xsmall"
    base["distillation"]["max_candidates"] = 120
    base["evaluation"] = {"warmup": 1, "repeats": 2}
    for name in STAGES:
        original = {"source": "prepare", "distill": "teacher"}.get(name, name)
        value = read(stage1_fixture["experiment"]["stages"][original])
        value["stage"] = {
            "source": "stage3_source",
            "distill": "stage3_distill",
            "prepare": "stage3_qualify",
            "evaluate": "stage3_evaluate",
        }.get(name, name)
        if name in {"codec", "generator"}:
            value["stage3_schedule"] = dict(
                schema="linear_warmup_cosine_v1",
                total=value["training"]["steps"],
                warmup=1,
                minimum_ratio=0.1,
            )
        path = stage1_fixture["folder"] / ("stage3-" + name + ".json")
        write_json(path, value)
        base["stages"][name] = str(path)
    path = stage1_fixture["folder"] / "stage3.json"
    write_json(path, base)
    return {**stage1_fixture, "config": path, "experiment": base}


@pytest.fixture
def inline_workers(monkeypatch):
    """Inject fixture workers at the process seam, preserving real Run/receipt checks."""
    from ncm_lab.runtime.runner import execute

    def inline(value, attempt, path, state, settings, resume_checkpoint=None):
        attempt.mkdir(parents=True)
        cfg = attempt / "resolved_config.json"
        write_json(cfg, value, exclusive=True)
        with monkeypatch.context() as patch:
            patch.setenv("NCM_CHILD_LAUNCH_RECORD", str(attempt / "launch.json"))
            result = execute(value, checked(resume_checkpoint) if resume_checkpoint else None)
        result = json.loads(json.dumps(result))
        return {
            "result": result,
            "exit_status": 0 if result["status"] == "completed" else 2,
            "config": reference(cfg),
            "run": reference(Path(result["path"]) / "manifest.json")
            if result["status"] == "completed"
            else None,
        }

    monkeypatch.setattr("ncm_lab.stage1.workflow.execute_child", inline)


@pytest.fixture
def synthetic_teacher(monkeypatch):
    from transformers import Qwen3ForCausalLM

    calls = []

    def generate(self, input_ids, attention_mask, generation_config):
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert generation_config.max_time is None and generation_config.forced_eos_token_id is None
        assert generation_config.min_new_tokens == 0 and not generation_config.do_sample
        calls.append(input_ids.tolist())
        return torch.cat([input_ids, input_ids.new_tensor([[8, 9, 2]])], dim=1)

    monkeypatch.setattr(Qwen3ForCausalLM, "generate", generate)
    return calls


def test_complete_sequence_distillation_training_evaluation_chat_resume(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher, monkeypatch, capsys
):
    folder = tmp_path / "Stage 3 experiment with spaces"
    calls = []

    def chat(**kwargs):
        state = read(folder / "manifest.json")
        assert state["status"] == "completed" and state["locked_test_observed"]
        assert (folder / "reports/stage3_summary.md").exists()
        calls.append(kwargs)
        return {"status": "stopped"}

    monkeypatch.setattr("ncm_lab.chat.server.serve", chat)
    assert (
        main(["--config", str(stage3_fixture["config"]), "--experiment-dir", str(folder), "--execute"]) == 0
    ), capsys.readouterr()
    path, state, settings = load(folder / "manifest.json")
    assert all(e["status"] == "completed" for e in state["stages"].values())
    assert [h["stage"] for h in state["history"] if h["event"] == "stage_completed"] == list(STAGES)
    from ncm_lab.data.artifacts import Dataset, LOCKED

    prep = read(checked(state["stages"]["prepare"]["receipt"]))["result"]
    data = Dataset(prep["artifact"])
    test = Dataset(checked(prep["evaluation_data"]), LOCKED)
    assert len(data) == 12 and len(test) == 4
    assert all(data[i][0].provenance["target_origin"] == "teacher_generated" for i in range(len(data)))
    assert all(data[i][0].source_type == "software_fixture" for i in range(len(data)))
    report = read(folder / "reports/stage3_summary.json")
    assert report["workflow_completed"] and report["origin"] == "software_fixture"
    assert (
        report["evaluation"]["serial_model_loading"] and len(report["evaluation"]["matched_prompt_ids"]) == 4
    )
    assert report["codec"]["scheduler"]["completed_updates"] == 12
    assert report["generator"]["scheduler"]["completed_updates"] == 3
    assert (
        calls[0]["checkpoint"] == read(checked(state["stages"]["generator"]["receipt"]))["result"]["artifact"]
    )
    before = digest(path)
    generation_count = len(synthetic_teacher)
    assert main(["--resume", str(path), "--execute", "--chat-port", "0"]) == 0, capsys.readouterr()
    assert len(synthetic_teacher) == generation_count and digest(path) == before and calls[-1]["port"] == 0
    assert main(["--resume", str(path), "--execute", "--seed", "17"]) == 2

    def occupied(**kwargs):
        raise OSError("fixture occupied loopback port")

    monkeypatch.setattr("ncm_lab.chat.server.serve", occupied)
    assert main(["--resume", str(path), "--execute"]) == 3
    assert read(folder / "chat_status.json")["training_status"] == "completed"
    assert read(path)["status"] == "completed" and digest(path) == before
    assert len(synthetic_teacher) == generation_count

    def cancelled(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("ncm_lab.chat.server.serve", cancelled)
    assert main(["--resume", str(path), "--execute"]) == 0
    assert read(folder / "chat_status.json")["status"] == "closed_by_operator"
    assert digest(path) == before


def test_source_only_real_parquet_pending_no_teacher(tmp_path, stage3_fixture, tulu_fixture, capsys):
    path = Path(stage3_fixture["experiment"]["stages"]["source"])
    cfg = read(path)
    cfg["data"] = tulu_fixture["data"]["data"]
    write_json(path, cfg)
    # Use 64 positions so the complete original fixture references fit the locked set.
    folder = tmp_path / "parquet source"
    assert (
        main(
            [
                "--config",
                str(stage3_fixture["config"]),
                "--canvas-size",
                "medium",
                "--experiment-dir",
                str(folder),
                "--qualify-only",
            ]
        )
        == 0
    ), capsys.readouterr()
    state = read(folder / "manifest.json")
    assert state["stages"]["source"]["status"] == "completed"
    assert all(state["stages"][n]["status"] == "pending" for n in STAGES[1:])
    result = read(checked(state["stages"]["source"]["receipt"]))["result"]
    assert result["final_target_capacity"] == "pending_distillation" and not result["model_loaded"]


def test_no_chat_reference_only_actual_children(tmp_path, stage3_fixture, capsys):
    folder = tmp_path / "child workflow"
    assert (
        main(
            [
                "--config",
                str(stage3_fixture["config"]),
                "--target-policy",
                "reference-only",
                "--experiment-dir",
                str(folder),
                "--execute",
                "--no-chat",
            ]
        )
        == 0
    ), capsys.readouterr()
    state = read(folder / "manifest.json")
    for name in STAGES:
        attempt = Path(state["stages"][name]["attempts"][0]["path"])
        assert read(attempt / "process.json")["execution_deadline"] is None
    assert not (folder / "chat_status.json").exists()


def test_scheduler_exact_recovery_and_selected_only_refusal(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher, monkeypatch
):
    from ncm_lab.runtime.runner import execute
    from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE

    settings = resolve(stage3_fixture["config"])
    path = Path(plan(settings, tmp_path / "recovery")["experiment"])
    for name in STAGES[:6]:
        result = run(path, stage=name)
        assert result["status"] == "ready", result
    _, state, settings = load(path)
    cfg = configuration("generator", path, state, settings)
    partial = execute(cfg, stop_after=1)
    assert partial["status"] == "incomplete", partial
    restored = execute(cfg, resume=partial["checkpoint"])
    full = execute(cfg)
    assert restored["status"] == full["status"] == "completed", (restored, full)
    a, b = CheckpointReader(restored["checkpoint"]), CheckpointReader(full["checkpoint"])

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x:
                equal(x[key], y[key])
        elif isinstance(x, (tuple, list)):
            for first, second in zip(x, y, strict=True):
                equal(first, second)
        else:
            assert x == y

    for key in ("model", "optimizer", "rng", "scheduler", "training"):
        left, right = a.materialize(a.field(key)), b.materialize(b.field(key))
        if key == "training":
            left.pop("selected")
            right.pop("selected")
            left.pop("attempt_id")
            right.pop("attempt_id")
        equal(left, right)
    # Selection-only schema can load weights but cannot restore Adam or sampler.
    selected_files = [
        p
        for p in Path(full["path"]).glob("checkpoints/version-*/manifest.json")
        if read(p)["schema"] == INFERENCE
    ]
    for selected in selected_files:
        with pytest.raises(ValueError, match="cannot resume"):
            CheckpointReader(selected)


def test_readme_exact_command_runs_full_fixture(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher, monkeypatch
):
    import runpy
    import shlex
    import subprocess
    import sys
    from ncm_lab.stage3 import launcher

    text = (root() / "README.md").read_text()
    block = text.split("<!-- BEGIN REQUIRED README STAGE3 COMMAND -->")[1].split(
        "<!-- END REQUIRED README STAGE3 COMMAND -->"
    )[0]
    command = block.split("```bash\n")[1].split("```")[0].strip()
    assert command == "python -u scripts/run_stage3.py --execute"
    subprocess.run(["bash", "-n"], input=command, text=True, check=True, timeout=5)
    settings = resolve(stage3_fixture["config"])
    monkeypatch.setattr(launcher, "resolve", lambda *a, **k: settings)
    monkeypatch.setattr(launcher, "root", lambda: tmp_path)
    monkeypatch.setattr("ncm_lab.chat.server.serve", lambda **kwargs: {"status": "fixture_closed"})
    monkeypatch.setattr(sys, "argv", shlex.split(command)[2:])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(root() / "scripts/run_stage3.py"), run_name="__main__")
    assert exit_info.value.code == 0
    manifests = list((tmp_path / "runs/stage3/xsmall").glob("*/manifest.json"))
    assert len(manifests) == 1 and read(manifests[0])["status"] == "completed"


def test_distillation_commits_resume_without_duplicate_generations(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher, monkeypatch
):
    from transformers import Qwen3ForCausalLM

    settings = resolve(stage3_fixture["config"])
    path = Path(plan(settings, tmp_path / "durable records")["experiment"])
    assert run(path, stage="source")["status"] == "ready"
    original = Qwen3ForCausalLM.generate
    count = 0

    def interrupted(self, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("fixture interruption before second record commit")
        return original(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Qwen3ForCausalLM, "generate", interrupted)
        failed = run(path, stage="distill")
    assert failed["status"] == "failed", failed
    assert run(path, stage="distill")["status"] == "ready"
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(path.parent / "distillation/committed.sqlite")) as db:
        rows = db.execute("SELECT attempts,payload FROM records").fetchall()
    assert sum(a == 2 for a, _ in rows) == 1
    assert all(payload for _, payload in rows)
    # The committed first output was generated once; only interrupted work retried.
    assert len(synthetic_teacher) == len(rows)
    _, state, _ = load(path)
    assert len(state["stages"]["distill"]["attempts"]) == 2


@pytest.mark.parametrize("minimum,expected", [(7, "ready"), (100, "blocked")])
def test_underfill_warning_or_hard_capacity_failure(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher, minimum, expected
):
    recipe = read(stage3_fixture["config"])
    recipe["qualification"].update(train_examples=100, minimum_train_examples=minimum)
    write_json(stage3_fixture["config"], recipe)
    path = Path(plan(resolve(stage3_fixture["config"]), tmp_path / "capacity")["experiment"])
    for name in ("source", "distill"):
        assert run(path, stage=name)["status"] == "ready"
    result = run(path, stage="prepare")
    assert result["status"] == expected, result
    state = read(path)
    if expected == "ready":
        receipt = read(checked(state["stages"]["prepare"]["receipt"]))["result"]
        report = read(checked(receipt["qualification"]))
        assert report["training_underfill_warning"] and report["actual_counts"]["train"] < 100
    assert state["stages"]["codec"]["status"] == "pending"


def test_finite_nonlearning_codec_warns_and_continues_only_stage3(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher
):
    codec_path = Path(stage3_fixture["experiment"]["stages"]["codec"])
    cfg = read(codec_path)
    cfg["training"].update(steps=2, learning_rate=1e-30, new_learning_rate=1e-30, validation_interval=1)
    cfg["stage3_schedule"].update(total=2, warmup=1)
    write_json(codec_path, cfg)
    path = Path(plan(resolve(stage3_fixture["config"]), tmp_path / "finite advisory")["experiment"])
    for name in STAGES[:5]:
        result = run(path, stage=name)
        assert result["status"] == "ready", result
    _, state, _ = load(path)
    result = read(checked(state["stages"]["codec"]["receipt"]))["result"]
    gate = result["stage1_diagnostics"]["codec_qualification"]
    assert gate["artifacts_valid"] and gate["quality_warning"]
    assert gate["codec_qualified_for_generator"] and gate["status"] == "qualified_with_warning"
    assert state["codec_review_override"] is None
    assert run(path, stage="targets")["status"] == "ready"


def test_final_reference_retry_reuses_outputs_and_warms_new_model(
    tmp_path, stage3_fixture, inline_workers, synthetic_teacher, monkeypatch
):
    from transformers import Qwen3ForCausalLM

    path = Path(plan(resolve(stage3_fixture["config"]), tmp_path / "evaluation retry")["experiment"])
    for name in STAGES[:-1]:
        result = run(path, stage=name)
        assert result["status"] == "ready", result
    original = Qwen3ForCausalLM.generate
    count = 0

    def interrupted(self, **kwargs):
        nonlocal count
        count += 1
        if count == 4:
            raise OSError("fixture interruption after first final reference record")
        return original(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Qwen3ForCausalLM, "generate", interrupted)
        assert run(path, stage="evaluate")["status"] == "failed"
    records = list((path.parent / "final_evaluation/original_qwen").glob("*.json"))
    assert len(records) == 1
    before = digest(records[0])
    generation_count = len(synthetic_teacher)
    result = run(path, stage="evaluate")
    assert result["status"] == "completed", result
    # Three uncached prompts, two measured generations each, one fresh warmup.
    assert len(synthetic_teacher) - generation_count == 7
    assert digest(records[0]) == before
    assert len(read(path)["stages"]["generator"]["attempts"]) == 1


@pytest.mark.parametrize("defect", ["role", "selection", "freezing", "objective"])
def test_invalid_scientific_stage_contract_rejected_before_allocation(stage3_fixture, defect):
    path = Path(stage3_fixture["experiment"]["stages"]["generator"])
    cfg = read(path)
    if defect == "role":
        cfg["stage"] = "codec"
    elif defect == "selection":
        cfg.pop("stage1")
    elif defect == "freezing":
        cfg["training"]["warmup_processed_tokens"] = 1
    else:
        cfg["training"]["auxiliary_weight"] = 0.2
    write_json(path, cfg)
    with pytest.raises(ValueError, match="Stage 3"):
        resolve(stage3_fixture["config"])
