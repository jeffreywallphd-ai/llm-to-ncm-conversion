"""Actual local fixture preparation and recovery; no real 8B feasibility claims."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.config.stages import config
from ncm_lab.runtime import pilot
from ncm_lab.runtime.canvas_pipeline import pipeline_configs, verify_stage, resume_config
from ncm_lab.runtime.io import write_json, read, digest, resolve_reference
from ncm_lab.runtime.runner import execute
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.data.artifacts import Dataset


def local_configs(tmp_path, canvas=8):
    from ncm_lab.runtime.budget import Budget

    source = tmp_path / "references.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "original_id": str(i),
                    "group_id": "group-" + str(i),
                    "prompt": f"token4 token{i % 28 + 4} {i}",
                    "answer": "token8 " * (7 if i < 20 else 8),
                }
            )
            for i in range(100)
        ),
        encoding="utf-8",
    )
    data = config("configs/data/real_pilot.yaml")
    data["data"] = dict(
        format="jsonl",
        source_path=str(source),
        source_sha256=digest(source),
        revision="software-fixture",
        license="GPL-3.0",
        max_source_rows=100,
        max_examples=100,
        max_unique_tokens=10000,
        max_prompt_tokens=512,
        dev_fraction=0.3,
        seed=17,
        overlength="exclude",
    )
    data["budget"] = asdict(
        Budget(
            "canvas-fixture",
            max_wall_seconds=120,
            max_steps=500,
            max_storage_bytes=200_000_000,
            max_phase_storage_bytes=1_000_000_000,
        )
    )
    data_path = tmp_path / "data-config.json"
    write_json(data_path, data)
    generator = config("configs/validation/generator_2_step_pilot.yaml")
    generator.update(
        model=model_spec(),
        architecture=tiny_architecture(canvas, activation_checkpointing=False).to_dict(),
        device="cpu",
        dtype="float32",
        hardware_profile="cpu",
        run_root=str(tmp_path / "runs"),
    )
    generator["budget"] = copy.deepcopy(data["budget"])
    generator["budget"]["enforce_wall_time_limits"] = False
    generator["training"].update(
        learning_rate=0.001,
        new_learning_rate=0.002,
        head_positions=8,
        head_chunk=4,
        max_validation_examples=2,
        shard_bytes=65536,
    )
    generator["monitoring"].update(sample_prompts=1, sample_interval=500)
    path = tmp_path / "generator-config.json"
    write_json(path, generator)
    return path, data_path


def test_real_fixture_preparation_two_then_100_and_exact_bound_resume(tmp_path, monkeypatch):
    generator, data = local_configs(tmp_path)
    monkeypatch.setattr(pilot, "find_spec", lambda *a: pytest.fail("no-chat checked server dependency"))
    monkeypatch.setattr(pilot.socket, "socket", lambda *a: pytest.fail("no-chat probed port"))
    first = pilot.run_pilot(
        2, config_path=generator, fresh=True, prepare=True, data_config=data, no_chat=True
    )
    assert first["status"] == "completed", first
    assert first["chat_started"] is False and first["training"]["optimizer_steps"] == 2
    values = pipeline_configs(config(generator), data)
    for stage in ("data", "teacher", "codec", "targets"):
        verify_stage(values[stage])
    # Same-capacity source/config, schedule, and codec changes never masquerade as reuse.
    for stage, mutate in (
        ("data", lambda v: v["data"].update(seed=99)),
        ("targets", lambda v: v["architecture"]["resolutions"].__setitem__(5, 5)),
        ("targets", lambda v: v["inputs"].update(codec=values["data"]["output_ref"])),
    ):
        changed = copy.deepcopy(values[stage])
        mutate(changed)
        with pytest.raises(ValueError, match="configuration/source/input binding differs"):
            verify_stage(changed)
    # An unavailable upstream cache requires rebuilding its downstream targets too.
    calls, invalidated = [], []
    original_verify, original_child = pilot.verify_stage, pilot.stage_child

    def verify_with_one_missing_codec(value):
        if value["stage"] == "codec" and not invalidated:
            invalidated.append(True)
            raise ValueError("software fixture: cache became unavailable after planning")
        return original_verify(value)

    def counted_child(value, *args, **kwargs):
        calls.append(value["stage"])
        return original_child(value, *args, **kwargs)

    monkeypatch.setattr(pilot, "verify_stage", verify_with_one_missing_codec)
    monkeypatch.setattr(pilot, "stage_child", counted_child)
    rebuilt = pilot.run_pilot(
        2, config_path=generator, fresh=True, prepare=True, data_config=data, no_chat=True
    )
    assert rebuilt["status"] == "completed", rebuilt
    assert calls == ["codec", "targets", "generator"]
    monkeypatch.setattr(pilot, "verify_stage", original_verify)
    monkeypatch.setattr(pilot, "stage_child", original_child)
    prepared = Dataset(values["data"]["output_ref"])
    q = prepared.manifest["extra"]["qualification"]
    assert q["source_rows_inspected"] == 100 and q["excluded"]["overlength"] == 80
    assert q["usable"] and sum(s["examples"] for s in q["splits"].values()) == 20
    assert all(
        len(e.answer_ids) == 8 and e.eos_genuine for e, _ in (prepared[i] for i in range(len(prepared)))
    )
    checkpoint = first["training"]["checkpoint"]
    parent_hash = digest(resolve_reference(values["generator"]["output_ref"]))
    # A mutable convenience pointer can advance/disappear; recovery keeps immutable bound inputs.
    Path(values["targets"]["output_ref"]).write_text('{"schema":"unrelated"}', encoding="utf-8")
    for flag in ("small", "64"):
        with pytest.raises(ValueError, match="resume canvas mismatch"):
            pilot.run_pilot(100, resume=checkpoint, canvas_size=flag, no_chat=True)
    resumed = pilot.run_pilot(100, resume=checkpoint, no_chat=True)
    assert resumed["status"] == "completed", resumed
    assert (
        resumed["training"]["optimizer_steps"] == 100 and resumed["training"]["charged_work"]["steps"] == 98
    )
    assert digest(resolve_reference(values["generator"]["output_ref"])) == parent_hash
    exported = read(resumed["training"]["artifact"])
    assert exported["architecture"]["canvas"] == 8 and exported["selected_optimizer_step"] <= 100
    samples = Path(resumed["training"]["path"]) / "monitor_samples.jsonl"
    assert {json.loads(line)["step"] for line in samples.read_text(encoding="utf-8").splitlines()} == {
        25,
        50,
        75,
        100,
    }
    attempt = read(Path(resumed["training"]["path"]) / "attempt.json")
    assert attempt["canvas"]["preset"] == "xsmall" and attempt["origin"] == "software_fixture"
    with pytest.raises(ValueError, match="cannot move a checkpoint backwards"):
        pilot.run_pilot(2, resume=resumed["training"]["checkpoint"], no_chat=True)


@pytest.mark.parametrize("stage", ["data", "teacher", "codec", "targets"])
@pytest.mark.parametrize("status", ["blocked", "failed", "cancelled", "budget_stopped", "incomplete"])
def test_unsuccessful_preparation_never_trains_or_starts_chat(tmp_path, monkeypatch, stage, status):
    path, data = local_configs(tmp_path)
    monkeypatch.setattr(
        pilot,
        "prerequisite_plan",
        lambda v: {n: {"reuse": n != stage} for n in ("data", "teacher", "codec", "targets")},
    )

    def verify(value):
        if value["stage"] == stage:
            raise ValueError("software fixture: prerequisite missing")
        return {}

    monkeypatch.setattr(pilot, "verify_stage", verify)
    seen = []

    def child(value, *a):
        seen.append(value["stage"])
        return {"status": status}

    monkeypatch.setattr(pilot, "stage_child", child)
    monkeypatch.setattr(pilot, "train_child", lambda *a: pytest.fail("generator started"))
    result = pilot.run_pilot(2, config_path=path, prepare=True, data_config=data, no_chat=True)
    assert result["status"] == status and result["failed_stage"] == stage and seen == [stage]


def test_capacity_failure_precedes_teacher_and_preserves_exclusions(tmp_path):
    path, data_path = local_configs(tmp_path)
    data = read(data_path)
    data["data"]["max_source_rows"] = 1
    write_json(data_path, data)
    result = pilot.run_pilot(2, config_path=path, prepare=True, data_config=data_path, no_chat=True)
    assert result["status"] == "blocked" and result["failed_stage"] == "data", result
    assert "data-capacity" in result["preparation"]["error"]
    assert "--data-config" in result["preparation"]["error"]
    assert not list((tmp_path / "runs/teacher").glob("*"))


def test_data_split_stability_prompt_capacity_and_censoring(tmp_path):
    path, data_path = local_configs(tmp_path)
    first = pipeline_configs(config(path), data_path)["data"]
    result8 = execute(first)
    assert result8["status"] == "completed", result8
    other = config(path)
    other["architecture"] = tiny_architecture(32).to_dict()
    result32 = execute(pipeline_configs(other, data_path)["data"])
    assert result32["status"] == "completed", result32
    d8, d32 = Dataset(result8["artifact"]), Dataset(result32["artifact"])
    small = {
        d8.metadata(i)["example"]["group_id"]: d8.metadata(i)["example"]["split"] for i in range(len(d8))
    }
    large = {
        d32.metadata(i)["example"]["group_id"]: d32.metadata(i)["example"]["split"] for i in range(len(d32))
    }
    assert all(large[g] == split for g, split in small.items())
    assert first["data"]["max_prompt_tokens"] == 512
    cfg = read(data_path)
    cfg["data"]["overlength"] = "censor"
    write_json(data_path, cfg)
    censored = execute(pipeline_configs(config(path), data_path)["data"])
    assert censored["status"] == "completed", censored
    ds = Dataset(censored["artifact"])
    prefixes = [ds[i][0] for i in range(len(ds)) if ds[i][0].censored]
    assert prefixes and all(not e.eos_genuine and 2 not in e.answer_ids for e in prefixes)


def test_missing_matching_inputs_has_exact_preparation_command(tmp_path):
    with pytest.raises(
        ValueError,
        match="medium/64.*Prepare with: python scripts/run_100_step_pilot.py --fresh --canvas-size medium --prepare --no-chat",
    ):
        pilot.run_pilot(100, fresh=True, no_chat=True)


def test_unreviewed_source_fails_before_model_allocation(native_artifacts, tmp_path, monkeypatch):
    from ncm_lab.training.checkpoints import CheckpointReader, save_checkpoint, native_identity

    # An independent synthetic checkpoint, never an edit to a real outcome.
    reader = CheckpointReader(native_artifacts["results"]["generator"]["checkpoint"])
    payload = reader.materialize(reader.manifest["tree"])
    payload["metadata"]["implementation_sha256"] = "f" * 64
    payload["identity"] = native_identity(payload["metadata"])
    folder = tmp_path / "runs/generator/synthetic"
    folder.mkdir(parents=True)
    original_run = reader.path.parent.parent.parent
    for filename in ("resolved_config.json", "launch.json"):
        (folder / filename).write_bytes((original_run / filename).read_bytes())
    saved = save_checkpoint(folder / "checkpoints", payload)
    monkeypatch.setattr(pilot, "train_child", lambda *a: pytest.fail("child started"))
    with pytest.raises(ValueError, match="unreviewed implementation transition"):
        resume_config(saved["manifest"])


def test_cancellation_stops_owned_tree_and_preserves_unrelated_child(tmp_path):
    import ctypes
    import os
    import subprocess
    import sys
    import time

    helper = Path(__file__).parents[1] / "fixtures/process_tree.py"
    pid_file = tmp_path / "leaf.pid"
    options = {
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        "start_new_session": os.name != "nt",
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(12)"], **options)
    owned = subprocess.Popen([sys.executable, str(helper), "parent", str(pid_file)], **options)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.is_file(), "owned grandchild must start to test tree cleanup"
        leaf = int(pid_file.read_text())
        pilot.stop_child(owned)
        assert owned.poll() is not None and unrelated.poll() is None
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x00100000, False, leaf)
            if handle:
                try:
                    assert kernel.WaitForSingleObject(handle, 5000) == 0
                finally:
                    kernel.CloseHandle(handle)
        else:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = Path(f"/proc/{leaf}/stat")
                try:
                    os.kill(leaf, 0)
                except ProcessLookupError:
                    break
                if status.exists() and status.read_text().split()[2] == "Z":
                    break
                time.sleep(0.05)
            else:
                pytest.fail("owned grandchild survived cancellation")
    finally:
        pilot.stop_child(owned)
        pilot.stop_child(unrelated)
