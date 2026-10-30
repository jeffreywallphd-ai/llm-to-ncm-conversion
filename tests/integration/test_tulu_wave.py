import copy
import json
from pathlib import Path
import pytest
from ncm_lab.config.waves import STAGES, wave_config
from ncm_lab.runtime.io import read, atomic_json
from ncm_lab.runtime.wave import plan, run_wave
from ncm_lab.data.artifacts import Dataset


def test_plan_is_offline_unknown_and_read_only(wave_fixture, monkeypatch):
    import urllib.request
    import ncm_lab.teacher.model as models
    import ncm_lab.runtime.budget as budget

    def forbidden(*args, **kwargs):
        pytest.fail("offline plan attempted execution")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(models, "load_original", forbidden)
    monkeypatch.setattr(budget.Meter, "__init__", forbidden)
    before = set(wave_fixture["folder"].rglob("*"))
    result = plan(wave_fixture["wave"])
    assert result["capacity"]["status"] == "unknown" and not result["ledger_reserved"]
    assert result["stages"] == list(STAGES) and set(wave_fixture["folder"].rglob("*")) == before


def test_connected_wave_sequential_processes_resume_and_corruption(wave_fixture, monkeypatch):
    result = run_wave(wave_fixture["wave"])
    assert result["status"] == "completed", result
    folder = Path(result["manifest"]).parent
    state = read(folder / "status.json")
    assert list(state["completed"]) == list(STAGES)
    children = state["completed"]
    processes = [read(Path(r["child_folder"]) / "process.json")["pid"] for r in children.values()]
    assert len(set(processes)) == len(STAGES)
    for name in ("data", "teacher", "codec", "targets", "generator"):
        value = read(children[name]["config"]["manifest"])
        assert str(folder / "handoffs") in value["output_ref"]
        assert value["wave_context"]["wave_id"] == result["wave_id"]
    for name in ("teacher", "targets"):
        dataset = Dataset(children[name]["result"]["artifact"])
        for _, row in dataset.iter_metadata():
            p = row["example"]["provenance"]
            assert p["component_source"] == "ai2-adapt-dev/no_robots_converted"
            assert p["answer_origin"] == "unknown" and p["dataset_revision"].startswith("b14afda")
    generator = children["generator"]["result"]
    assert generator["completion"]["code"] == "objective_attained"
    assert generator["processed_source_tokens"] >= 100 and generator["last_executed_phase"] == "full"
    assert generator["completion"]["token_overshoot"] <= generator["completion"]["maximum_token_overshoot"]
    assert Path(generator["checkpoint"]).is_file() and Path(generator["artifact"]).is_file()
    from ncm_lab.runtime.generation import run_generation
    from ncm_lab.data.tokenization import prompt_tokens
    from ncm_lab.teacher.model import load_tokenizer

    messages = wave_fixture["rows"][0]["messages"][:-1]
    message_path = folder / "generation-messages.json"
    message_path.write_text(json.dumps(messages))
    generated = run_generation(generator["artifact"], messages_file=message_path)
    assert generated["status"] == "completed", generated
    assert generated["charged_work"]["source_tokens"] == len(
        prompt_tokens(load_tokenizer(wave_fixture["common"]["model"]), messages)
    )
    from ncm_lab.training.checkpoints import CheckpointReader

    reader = CheckpointReader(generator["checkpoint"])
    terminal = reader.materialize(reader.field("training"))
    assert terminal["processed_source_tokens"] == generator["processed_source_tokens"]
    assert terminal["cursor"] and terminal["best_dev_loss"] is not None
    from ncm_lab.runtime.wave import verify_completed

    with pytest.raises(ValueError, match="cross-wave"):
        verify_completed(
            children["generator"],
            lambda: None,
            read(children["generator"]["config"]["manifest"]),
            "unrelated-wave",
            read(result["manifest"])["implementation_sha256"],
        )
    # Completed exact-wave stages are verified without invoking another child.
    import ncm_lab.runtime.wave as controller

    monkeypatch.setattr(
        controller, "execute_child", lambda *a, **k: pytest.fail("completed stage relaunched")
    )
    resumed = run_wave(wave_fixture["wave"], result["manifest"])
    assert resumed["status"] == "completed", resumed
    index = Path(children["targets"]["result"]["artifact"]).parent / "records.jsonl"
    index.write_bytes(index.read_bytes() + b"\n")
    corrupt = run_wave(wave_fixture["wave"], result["manifest"])
    assert corrupt["status"] == "failed" and "changed" in corrupt["reason"]


@pytest.mark.parametrize(
    "status", ["failed", "blocked", "cancelled", "budget_stopped", "completed", "unexpected"]
)
def test_child_stops_prevent_downstream_launches(wave_fixture, monkeypatch, status):
    import ncm_lab.runtime.wave as controller

    calls = []

    def stopped(command, folder, *args):
        calls.append(command)
        folder.mkdir()
        return 2, {"status": status, "reason": "explicit software fault fixture"}

    monkeypatch.setattr(controller, "execute_child", stopped)
    result = run_wave(wave_fixture["wave"])
    assert result["status"] == (
        status if status in {"failed", "blocked", "cancelled", "budget_stopped"} else "failed"
    )
    assert len(calls) == 1 and not result["completed_stages"]


def test_wave_configuration_resume_mismatch_fails_closed(wave_fixture, monkeypatch):
    import ncm_lab.runtime.wave as controller

    def stopped(command, folder, *args):
        folder.mkdir()
        return 2, {"status": "blocked"}

    monkeypatch.setattr(controller, "execute_child", stopped)
    result = run_wave(wave_fixture["wave"])
    changed = copy.deepcopy(wave_fixture["wave"])
    changed["notes"] = "different identity"
    with pytest.raises(ValueError, match="resume configuration"):
        run_wave(changed, result["manifest"])


def test_real_lock_and_tokenizer_only_selection_preserve_pilot():
    from ncm_lab.config.stages import config
    from ncm_lab.runtime.prepare import selected_assets
    from ncm_lab.data.prepare import validate_source_binding

    pilot = read("configs/data/assets.lock.json")["assets"]
    assert selected_assets(config("configs/validation/real_pilot.yaml")["preparation"]) == pilot
    sources = selected_assets(config("configs/preparation/tulu3_sources.yaml")["preparation"])
    assert not any(a["filename"].endswith(".safetensors") for a in sources)
    shards = validate_source_binding(config("configs/data/tulu3_extension.yaml")["data"])
    assert len(shards) == 6 and sum(a["size_bytes"] for a in shards) == 1412954868
    assert len({a["sha256"] for a in shards}) == 6
    settings = config("configs/data/tulu3_extension.yaml")["data"]
    settings["source_asset_ids"] = ["tulu3:README.md"]
    with pytest.raises(ValueError, match="train/role"):
        validate_source_binding(settings)


def test_wave_resumes_explicit_complete_generator_recovery(wave_fixture):
    path = Path(wave_fixture["stage_paths"]["generator"])
    cfg = read(path)
    cfg["budget"]["max_steps"] = 1
    cfg["training"].update(validation_interval=1, checkpoint_interval=1)
    atomic_json(path, cfg)
    wave = wave_config(wave_fixture["config_path"])
    first = run_wave(wave)
    assert first["status"] == "budget_stopped" and first["current_stage"] == "generator", first
    folder = Path(first["manifest"]).parent
    before = read(Path(wave["run_root"]) / "ledgers" / (wave["phase_id"] + ".json"))
    state = read(folder / "status.json")
    launch = read(Path(state["last_attempt"]["folder"]) / "launch.json")
    assert Path(launch["path"], "checkpoints/current.json").is_file()
    resumed_attempts = 0
    while resumed_attempts < 4:
        second = run_wave(wave, first["manifest"])
        resumed_attempts += 1
        if second["status"] == "completed":
            break
        assert second["status"] == "budget_stopped", second
    assert second["status"] == "completed", second
    after = read(Path(wave["run_root"]) / "ledgers" / (wave["phase_id"] + ".json"))
    assert (
        after["charges"]["steps"] > before["charges"]["steps"]
        and len(after["runs"]) == len(before["runs"]) + resumed_attempts
    )
    events = (folder / "events.jsonl").read_text()
    assert "exact_training_resume" in events and "verified_reuse" in events


def test_partial_preprocessing_restart_retains_failed_charge(wave_fixture):
    path = Path(wave_fixture["stage_paths"]["teacher"])
    cfg = read(path)
    cfg["budget"]["max_source_tokens"] = 1
    atomic_json(path, cfg)
    wave = wave_config(wave_fixture["config_path"])
    first = run_wave(wave)
    assert first["status"] == "budget_stopped" and first["current_stage"] == "teacher", first
    before = read(Path(wave["run_root"]) / "ledgers" / (wave["phase_id"] + ".json"))
    second = run_wave(wave, first["manifest"])
    assert second["status"] == "budget_stopped", second
    after = read(Path(wave["run_root"]) / "ledgers" / (wave["phase_id"] + ".json"))
    assert after["wall"] > before["wall"] and len(after["runs"]) == len(before["runs"]) + 1
    folder = Path(first["manifest"]).parent
    assert "preprocessing_restart_charged" in (folder / "events.jsonl").read_text()
    assert len(list(Path(wave["run_root"]).glob("teacher/*/manifest.json"))) == 2


def test_wave_capacity_gate_prevents_weights_and_teacher(wave_fixture):
    path = Path(wave_fixture["stage_paths"]["data"])
    cfg = read(path)
    cfg["data"]["codec_minimum_answer_positions"] = 10000
    atomic_json(path, cfg)
    teacher = Path(wave_fixture["stage_paths"]["teacher"])
    cfg = read(teacher)
    cfg["data"]["max_unique_answer_positions"] = 10000
    atomic_json(teacher, cfg)
    result = run_wave(wave_config(wave_fixture["config_path"]))
    assert result["status"] == "blocked" and result["qualification"] == "blocked", result
    assert result["completed_stages"] == ["prepare_sources"] and result["current_stage"] == "data"
    assert not list(Path(wave_fixture["wave"]["run_root"]).glob("teacher/*"))
