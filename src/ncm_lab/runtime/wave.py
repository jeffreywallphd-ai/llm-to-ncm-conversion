"""Sequential native CLI orchestration with exact wave lineage and fail-closed resume."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from importlib.metadata import version
from ncm_lab.config.waves import STAGES
from ncm_lab.runtime.io import (
    root,
    read,
    digest,
    canonical_hash,
    implementation_hash,
    write_json,
    atomic_json,
    reference,
)
from ncm_lab.runtime.prepare import prepare, selected_assets


def software():
    return {
        name: version(name) for name in ("ncm-lab", "torch", "transformers", "pyarrow", "langid", "numpy")
    }


def remaining(wave):
    folder = Path(wave["run_root"]) / "ledgers"
    child = wave["stages"]["generator"]["budget"]
    path = folder / (wave["phase_id"] + ".json")
    state = read(path) if path.exists() else {}
    result = {
        "native": {
            "ledger": str(path),
            "reserved_or_spent_wall_seconds": state.get("wall", 0),
            "remaining_wall_seconds": child["max_phase_wall_seconds"] - state.get("wall", 0),
            "charges": state.get("charges", {}),
            "caps": {k: v for k, v in child.items() if k.startswith("max_phase_")},
        }
    }
    for name in ("prepare_sources", "prepare_model"):
        settings = wave["stages"][name]["preparation"]
        budget = settings["budget"]
        path = (
            root() / settings["assets_root"] / "preparations" / (budget["phase_budget_id"] + ".ledger.json")
        )
        state = read(path) if path.exists() else {}
        result[name] = {
            "ledger": str(path),
            "spent_wall_seconds": state.get("elapsed_seconds", 0),
            "remaining_wall_seconds": budget["max_phase_wall_seconds"] - state.get("elapsed_seconds", 0),
            "charged_download_bytes": state.get("charged_download_bytes", 0),
            "remaining_download_bytes": budget["max_download_bytes"] - state.get("charged_download_bytes", 0),
        }
    path = folder / (wave["phase_id"] + "-controller.json")
    result["controller"] = read(path) if path.exists() else {"wall": 0, "attempts": 0}
    return result


def plan(wave):
    started = time.monotonic()

    def check():
        if time.monotonic() - started > wave["limits"]["max_controller_seconds"]:
            raise TimeoutError("offline plan read/verification time limit reached")

    acquisition = {
        name: prepare(stage, True) for name, stage in wave["stages"].items() if name.startswith("prepare_")
    }
    missing = []
    for name in acquisition:
        for asset in selected_assets(wave["stages"][name]["preparation"]):
            from ncm_lab.data.download import asset_path

            path = asset_path(root() / wave["stages"][name]["preparation"]["assets_root"], asset)
            if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
                missing.append(asset["id"])
    capacity = {
        "status": "unknown",
        "reason": "No explicit qualified extension artifact supplied; capacity is not measured.",
    }
    if wave.get("qualified_data"):
        from ncm_lab.runtime.io import resolve_reference

        path = resolve_reference(wave["qualified_data"])
        verify_reference(reference(path), check)
        report_path = path.parent / "capacity.json"
        report = read(report_path)
        if digest(report_path) != read(path.parent / "capacity.ref.json")["sha256"] or report[
            "data_manifest_sha256"
        ] != digest(path):
            raise ValueError("explicit capacity artifact identity changed")
        capacity = {
            "status": report["status"],
            "data": reference(path),
            "report": reference(report_path),
            "selected": report["selected"],
            "blockers": report["blockers"],
        }
    return {
        "schema": "native_wave_plan_v1",
        "status": "planned",
        "phase_id": wave["phase_id"],
        "config_sha256": canonical_hash(wave),
        "stages": list(STAGES),
        "acquisition": acquisition,
        "missing_asset_ids": missing,
        "capacity": capacity,
        "remaining_aggregate_capacity": remaining(wave),
        "limits": wave["limits"],
        "objectives": {
            name: stage["training"]
            for name, stage in wave["stages"].items()
            if name in {"codec", "generator"}
        },
        "storage_estimates_bytes": {
            "minimum_codec_teacher_states": 10000000 * 4096 * 2,
            "upper_pool_fine_codes": 100000000 * 384 * 2,
            "four_generator_recovery_versions": 4 * 8250600480 * 10,
            "selected_generator_export": 8250600480 * 2,
            "prepared_metadata_index_allowance": 1000000 * 16384,
            "free_space_reserve": wave["limits"]["reserve_free_bytes"],
        },
        "model_downloads_started": False,
        "training_started": False,
        "ledger_reserved": False,
        "fresh_initialization": "New extension codec and pinned-Qwen generator; no pilot recovery transfer.",
        "monitor_command": "ncm monitor serve --runs-dir "
        + str(wave["run_root"])
        + " --host 127.0.0.1 --port 6006",
    }


@contextmanager
def lock(path):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def verify_reference(ref, check):
    path = Path(ref["manifest"]).resolve()
    if digest(path, check) != ref["sha256"]:
        raise ValueError("wave artifact/reference changed: " + str(path))
    value = read(path)
    schema = value.get("schema", "")
    if schema in {"native_data_v1", "native_features_v1", "native_targets_v1"}:
        from ncm_lab.data.artifacts import Dataset
        from ncm_lab.data.records import Example

        data = Dataset(path)
        for _, row in data.iter_metadata():
            check()
            if Example(**row["example"]).identity != row["example_sha256"]:
                raise ValueError("wave example changed")
            if "tensor" in row:
                tensor = (path.parent / row["tensor"]["file"]).resolve()
                if not tensor.is_relative_to(path.parent) or digest(tensor, check) != row["tensor"]["sha256"]:
                    raise ValueError("wave tensor cache changed")
    elif schema in {"native_codec_v1", "native_generator_export_v1"}:
        from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE

        weights = (path.parent / value["weights_manifest"]).resolve()
        sha = value.get("weights_sha256", value.get("weights_manifest_sha256"))
        if not weights.is_relative_to(path.parent) or digest(weights, check) != sha:
            raise ValueError("wave selected weights changed")
        CheckpointReader(weights, expected_kind=INFERENCE, identity=value["identity"], check=check)
        for name, sha in value.get("tokenizer_files", {}).items():
            target = (path.parent / "tokenizer" / name).resolve()
            if target.parent != path.parent / "tokenizer" or digest(target, check) != sha:
                raise ValueError("wave export tokenizer changed")
    elif schema == "native_capacity_v1":
        if value["status"] != "qualified":
            raise ValueError("wave source is not qualified")
    elif schema == "native_asset_preparation_v1":
        for asset in value["assets"]:
            if digest(asset["path"], check) != asset["sha256"]:
                raise ValueError("wave acquired asset changed")
    else:
        raise ValueError("unsupported wave artifact schema")
    return path


def verify_completed(record, check, expected=None, wave_id=None, code=None):
    if record["result"]["status"] != "completed":
        raise ValueError("wave cannot reuse incomplete child")
    if record.get("receipt"):
        receipt = record["receipt"]
        if digest(receipt["manifest"], check) != receipt["sha256"] or read(receipt["manifest"]) != {
            k: v for k, v in record.items() if k != "receipt"
        }:
            raise ValueError("wave stage receipt changed")
    if record.get("run"):
        from ncm_lab.runtime.provenance import validate_run

        path = Path(record["run"]["manifest"])
        verify = validate_run(path.parent)
        if digest(path, check) != record["run"]["sha256"] or verify["status"] != "completed":
            raise ValueError("wave child run changed")
        if expected is not None:
            from ncm_lab.config.stages import bind_inputs

            actual = read(path.parent / "resolved_config.json")
            bindings, missing = bind_inputs(expected)
            if (
                missing
                or actual != {**expected, "resolved_inputs": bindings}
                or verify["implementation_sha256"] != code
            ):
                raise ValueError("cross-wave or changed child configuration/input/software identity")
            if actual.get("wave_context", {}).get("wave_id") != wave_id:
                raise ValueError("cross-wave child reuse is forbidden")
            summary = read(path.parent / "summary.json")
            if any(record["result"].get(k) != v for k, v in summary.items()):
                raise ValueError("child summary/result changed")
            if record["result"].get("artifact"):
                handoff = read(expected["output_ref"])
                if handoff != reference(record["result"]["artifact"]):
                    raise ValueError("wave output handoff changed")
    for ref in record.get("outputs", []):
        verify_reference(ref, check)
    if record.get("checkpoint"):
        from ncm_lab.training.checkpoints import CheckpointReader

        path = Path(record["checkpoint"]["manifest"])
        if digest(path, check) != record["checkpoint"]["sha256"]:
            raise ValueError("terminal recovery reference changed")
        CheckpointReader(path, check=check)


def execute_child(command, folder, timeout, max_log_bytes, environment, check):
    """Bounded files, never buffered child output; child owns all model allocations."""
    folder.mkdir()
    write_json(folder / "command.json", {"argv": command, "timeout_seconds": timeout}, exclusive=True)
    started = time.monotonic()
    with (folder / "stdout.json").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        write_json(
            folder / "process.json",
            {"pid": process.pid, "started_utc": datetime.now(timezone.utc).isoformat()},
            exclusive=True,
        )
        try:
            while process.poll() is None:
                check()
                if time.monotonic() - started > timeout:
                    raise TimeoutError("wave child timeout")
                if (
                    sum(p.stat().st_size for p in (folder / "stdout.json", folder / "stderr.log"))
                    > max_log_bytes
                ):
                    raise ValueError("wave child output limit exceeded")
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    # Windows venv launchers own a second Python process. Stop the
                    # complete task-owned tree, including that actual interpreter.
                    taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
                    stopped = subprocess.run(
                        [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if stopped.returncode and process.poll() is None:
                        raise RuntimeError(
                            "owned child tree could not be stopped; inspect retained process record"
                        )
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            write_json(
                folder / "exit.json",
                {"exit_status": process.returncode, "elapsed_seconds": time.monotonic() - started},
                exclusive=True,
            )
    if sum(p.stat().st_size for p in (folder / "stdout.json", folder / "stderr.log")) > max_log_bytes:
        raise ValueError("wave child output limit exceeded; retained files were not buffered")
    result = read(folder / "stdout.json")
    if not isinstance(result, dict):
        raise ValueError("malformed or missing child result; inspect retained stdout/stderr")
    if process.returncode or result.get("status") != "completed":
        return process.returncode or 2, result
    return 0, result


def snapshots(wave, folder, identifier):
    values = copy.deepcopy(wave["stages"])
    directory = folder / "resolved"
    directory.mkdir()
    dependencies = directory / "dependencies"
    dependencies.mkdir()
    for name, value in values.items():
        value["run_root"] = wave["run_root"]
        value["wave_context"] = {"wave_id": identifier, "phase_id": wave["phase_id"], "stage": name}
        for section, field in [
            ("data", "asset_lock"),
            ("data", "source_policy"),
            ("preparation", "asset_lock"),
        ]:
            if value.get(section, {}).get(field):
                path = root() / value[section][field]
                target = dependencies / (digest(path) + ".json")
                if not target.exists():
                    target.write_bytes(path.read_bytes())
                value[section][field] = str(target)
        if name.startswith("prepare_"):
            continue
        value["output_ref"] = str(folder / "handoffs" / (name + ".json"))
        inputs = {
            "teacher": {"data": "data"},
            "codec": {"features": "teacher"},
            "targets": {"data": "data", "codec": "codec"},
            "generator": {"targets": "targets"},
        }.get(name, {})
        value["inputs"] = {key: str(folder / "handoffs" / (stage + ".json")) for key, stage in inputs.items()}
    refs = {}
    for name, value in values.items():
        path = directory / (name + ".json")
        write_json(path, value, exclusive=True)
        refs[name] = reference(path)
    return refs


def objective_progress(value, state=None):
    options, state = value["training"], state or {}
    mode = options["objective"]
    if mode == "tokens":
        target, completed = options["max_processed_tokens"], state.get("processed_source_tokens", 0)
    elif mode == "epochs":
        cursor = state.get("cursor", {})
        target = options["epochs"]
        completed = cursor.get("epoch", 0) + cursor.get("position", 0) / max(1, len(cursor.get("order", [])))
    else:
        target, completed = options["steps"], state.get("optimizer_steps", 0)
    return {
        "mode": mode,
        "target": target,
        "completed": completed,
        "remaining": max(0, target - completed),
        "scope": "last controller-verified complete boundary; child canonical metrics provide live progress",
    }


def run_wave(wave, resume=None):
    run_root = Path(wave["run_root"])
    wave_root = run_root / "waves"
    wave_root.mkdir(parents=True, exist_ok=True)
    code, config_hash, runtime = implementation_hash(), canonical_hash(wave), software()
    if resume:
        manifest = Path(resume).resolve()
        meta = read(manifest)
        folder = manifest.parent
        if (
            meta.get("schema") != "native_wave_manifest_v1"
            or meta["config_sha256"] != config_hash
            or meta["implementation_sha256"] != code
            or meta["software"] != runtime
            or not folder.is_relative_to(wave_root.resolve())
        ):
            raise ValueError("wave resume configuration/software/implementation/namespace changed")
        if digest(folder / "resolved_config.json") != meta["resolved_config_file_sha256"]:
            raise ValueError("wave configuration snapshot changed")
    else:
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        folder = wave_root / identifier
        folder.mkdir()
        write_json(folder / "resolved_config.json", wave, exclusive=True)
        refs = snapshots(wave, folder, identifier)
        meta = {
            "schema": "native_wave_manifest_v1",
            "wave_id": identifier,
            "phase_id": wave["phase_id"],
            "config_sha256": config_hash,
            "implementation_sha256": code,
            "software": runtime,
            "resolved_config_file_sha256": digest(folder / "resolved_config.json"),
            "stages": refs,
        }
        write_json(folder / "manifest.json", meta, exclusive=True)
        atomic_json(folder / "status.json", {"status": "ready", "completed": {}, "current_stage": None})
    for path in (folder / "resolved/dependencies").iterdir():
        if digest(path) != path.stem:
            raise ValueError("wave dependency snapshot changed")
    started, waiting = time.monotonic(), 0.0
    limits = wave["limits"]
    ledger = run_root / "ledgers" / (wave["phase_id"] + "-controller.json")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    status = read(folder / "status.json")

    def event(kind, **fields):
        with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}) + "\n"
            )

    def check():
        if time.monotonic() - started > limits["max_wave_wall_seconds"]:
            raise TimeoutError("wave invocation wall ceiling reached")
        if time.monotonic() - started - waiting > limits["max_controller_seconds"] and not status.get(
            "child_running"
        ):
            raise TimeoutError("wave controller overhead ceiling reached")
        if status.get("child_running") and not status.get("active_child"):
            path = Path(status["last_attempt"]["folder"]) / "launch.json"
            if path.exists():
                try:
                    launch = read(path)
                    if (
                        isinstance(launch, dict)
                        and digest(launch["launch"]["manifest"]) == launch["launch"]["sha256"]
                    ):
                        status["active_child"] = launch
                        atomic_json(folder / "status.json", status)
                except (OSError, ValueError, KeyError):
                    pass  # Retry an in-progress small launch write; never select another run.

    with lock(folder / "execution.lock"), lock(ledger.with_suffix(".lock")):
        spending = read(ledger) if ledger.exists() else {"wall": 0, "attempts": 0, "caps": limits}
        if (
            spending["caps"] != limits
            or spending["wall"] + limits["max_controller_seconds"] > limits["max_phase_controller_seconds"]
            or spending["attempts"] >= limits["max_phase_attempts"]
        ):
            raise ValueError("persistent controller allowance exhausted or caps changed")
        spending["wall"] += limits["max_controller_seconds"]
        spending["attempts"] += 1
        atomic_json(ledger, spending)
        attempt = folder / "attempts" / str(spending["attempts"])
        attempt.mkdir(parents=True)
        if status.get("failure") or status.get("reason"):
            status.setdefault("failures", []).append(
                {
                    "status": status["status"],
                    "stage": status.get("current_stage"),
                    "failure": status.pop("failure", None),
                    "reason": status.pop("reason", None),
                }
            )
        event("attempt_started", attempt=spending["attempts"])
        try:
            for name in STAGES:
                check()
                status["current_stage"] = name
                config_name = "generator" if name == "preflight" else name
                config_ref = meta["stages"][config_name]
                if digest(config_ref["manifest"]) != config_ref["sha256"]:
                    raise ValueError("wave stage snapshot changed")
                value = read(config_ref["manifest"])
                if name in status["completed"]:
                    if status["completed"][name]["config"] != config_ref or not Path(
                        status["completed"][name]["child_folder"]
                    ).is_relative_to(folder / "attempts"):
                        raise ValueError("cross-wave stage receipt/configuration")
                    verify_completed(status["completed"][name], check, value, meta["wave_id"], code)
                    event("verified_reuse", stage=name)
                    continue
                config_name = "generator" if name == "preflight" else name
                config_ref = meta["stages"][config_name]
                if digest(config_ref["manifest"]) != config_ref["sha256"]:
                    raise ValueError("wave stage snapshot changed")
                value = read(config_ref["manifest"])
                if name == "prepare_model":
                    from ncm_lab.data.artifacts import Dataset
                    from ncm_lab.data.capacity import qualified_selection

                    data = Dataset(status["completed"]["data"]["result"]["artifact"])
                    teacher = read(meta["stages"]["teacher"]["manifest"])
                    qualified_selection(
                        data,
                        teacher["data"]["max_unique_answer_positions"],
                        teacher["data"]["max_dev_examples"],
                        check,
                    )
                    capacity = read(data.path / "capacity.json")
                    estimate = sum(
                        v for v in capacity["storage_estimates_bytes"].values() if isinstance(v, int)
                    )
                    estimate += prepare(value, True)["missing_payload_bytes"]
                    if (
                        not value["model"].get("fixture")
                        and shutil.disk_usage(run_root).free < estimate + limits["reserve_free_bytes"]
                    ):
                        raise ValueError(
                            "qualified extension storage estimates plus headroom exceed free disk"
                        )
                child_folder = attempt / name
                command = [sys.executable, "-m", "ncm_lab"]
                if name.startswith("prepare_"):
                    value["preparation"]["receipt_path"] = str(child_folder / "acquisition.json")
                    path = attempt / (name + "-acquisition-config.json")
                    write_json(path, value, exclusive=True)
                    command += ["prepare", "--config", str(path)]
                    timeout = value["preparation"]["budget"]["max_wall_seconds"] + 30
                elif name == "preflight":
                    command += ["preflight", "--config", config_ref["manifest"]]
                    timeout = 120
                else:
                    action = {
                        "data": "prepare",
                        "teacher": "extract",
                        "codec": "train",
                        "targets": "build",
                        "generator": "train",
                    }[name]
                    command += [name, action, "--config", config_ref["manifest"]]
                    timeout = value["budget"]["max_wall_seconds"] + 30
                    previous = status.get("last_attempt", {})
                    if name in {"codec", "generator"} and previous.get("stage") == name:
                        launch_path = Path(previous["folder"]) / "launch.json"
                        if launch_path.exists():
                            from ncm_lab.training.checkpoints import CheckpointReader

                            launch = read(launch_path)
                            if digest(launch["launch"]["manifest"], check) != launch["launch"]["sha256"]:
                                raise ValueError("child launch identity changed")
                            launched = read(Path(launch["path"]) / "resolved_config.json")
                            from ncm_lab.config.stages import bind_inputs

                            bindings, missing = bind_inputs(value)
                            if missing or launched != {**value, "resolved_inputs": bindings}:
                                raise ValueError("recovery belongs to another wave/configuration")
                            pointer = Path(launch["path"]) / "checkpoints/current.json"
                            if pointer.exists():
                                reader = CheckpointReader(pointer, check=check)
                                command += ["--resume", str(reader.path)]
                                status["last_valid_recovery"] = reference(reader.path)
                                status["objective_progress"] = objective_progress(
                                    value, reader.materialize(reader.field("training"))
                                )
                                event("exact_training_resume", stage=name, checkpoint=reference(reader.path))
                    elif previous.get("stage") == name:
                        event("preprocessing_restart_charged", stage=name, previous=previous)
                status.update(
                    status="running",
                    current_stage=name,
                    child_running=True,
                    active_child=None,
                    last_attempt={"stage": name, "folder": str(child_folder)},
                )
                if name in {"codec", "generator"} and "--resume" not in command:
                    status["objective_progress"] = objective_progress(value)
                atomic_json(folder / "status.json", status)
                event("stage_started", stage=name, command=command)
                child_started = time.monotonic()
                try:
                    exit_code, result = execute_child(
                        command,
                        child_folder,
                        min(timeout, limits["max_wave_wall_seconds"] - (child_started - started)),
                        limits["max_log_bytes"],
                        {**os.environ, "NCM_CHILD_LAUNCH_RECORD": str(child_folder / "launch.json")},
                        check,
                    )
                finally:
                    waiting += time.monotonic() - child_started
                    status["child_running"] = False
                record = {
                    "result": result,
                    "exit_status": exit_code,
                    "outputs": [],
                    "config": config_ref,
                    "child_folder": str(child_folder),
                }
                if exit_code or result.get("status") != "completed":
                    write_json(child_folder / "result.json", record, exclusive=True)
                    status.update(
                        status=result.get("status")
                        if result.get("status")
                        in {"failed", "blocked", "cancelled", "budget_stopped", "incomplete"}
                        else "failed",
                        failure={"stage": name, "result": result},
                    )
                    if result.get("qualification"):
                        status["qualification"] = result["qualification"]
                    if (
                        name in {"codec", "generator"}
                        and result.get("checkpoint")
                        and Path(result["checkpoint"]).is_file()
                    ):
                        from ncm_lab.training.checkpoints import CheckpointReader

                        recovery = CheckpointReader(result["checkpoint"], check=check)
                        status["last_valid_recovery"] = reference(recovery.path)
                        status["objective_progress"] = objective_progress(
                            value, recovery.materialize(recovery.field("training"))
                        )
                    event("stage_stopped", stage=name, result=result)
                    break
                if result.get("path") and result.get("run_id"):
                    record["run"] = reference(Path(result["path"]) / "manifest.json")
                if result.get("artifact"):
                    record["outputs"].append(reference(result["artifact"]))
                if result.get("capacity"):
                    record["outputs"].append(result["capacity"])
                if result.get("checkpoint"):
                    from ncm_lab.training.checkpoints import CheckpointReader

                    recovery = CheckpointReader(result["checkpoint"], check=check)
                    record["checkpoint"] = reference(recovery.path)
                    status["last_valid_recovery"] = record["checkpoint"]
                    status["objective_progress"] = objective_progress(
                        value, recovery.materialize(recovery.field("training"))
                    )
                verify_completed(record, check, value, meta["wave_id"], code)
                write_json(child_folder / "result.json", record, exclusive=True)
                record["receipt"] = reference(child_folder / "result.json")
                status["completed"][name] = record
                event("stage_completed", stage=name, record=record)
                atomic_json(folder / "status.json", status)
            else:
                status.update(status="completed", current_stage=None)
            status["qualification"] = (
                status["completed"]
                .get("data", {})
                .get("result", {})
                .get("qualification", status.get("qualification", "unknown"))
            )
        except TimeoutError as exc:
            status.update(status="budget_stopped", reason=str(exc))
            event("controller_limit", reason=str(exc))
        except KeyboardInterrupt:
            status.update(status="cancelled", reason="controller interrupted; complete recovery only")
            event("cancelled")
        except Exception as exc:
            status.update(status="failed", reason=type(exc).__name__ + ": " + str(exc))
            event("failed", reason=status["reason"])
        finally:
            spending["wall"] += (
                max(0, time.monotonic() - started - waiting) - limits["max_controller_seconds"]
            )
            atomic_json(ledger, spending)
            atomic_json(folder / "status.json", status)
            write_json(attempt / "result.json", status, exclusive=True)
        return {
            "schema": "native_wave_result_v1",
            "status": status["status"],
            "wave_id": meta["wave_id"],
            "manifest": str(folder / "manifest.json"),
            "summary": str(folder / "status.json"),
            "completed_stages": list(status["completed"]),
            "current_stage": status.get("current_stage"),
            "qualification": status.get("qualification", "unknown"),
            "reason": status.get("reason"),
            "attempt": spending["attempts"],
        }
