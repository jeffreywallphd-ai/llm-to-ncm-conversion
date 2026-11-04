"""Explicit two/100-step pilot, isolated training process, then a verified local chatbot."""

import argparse
from datetime import datetime, timezone
from importlib.util import find_spec
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
from ncm_lab.config.stages import config
from ncm_lab.config.architecture import Architecture
from ncm_lab.config.canvas import add_canvas_argument, canvas_metadata, resolve_canvas
from ncm_lab.runtime.canvas_pipeline import (
    pipeline_configs,
    prerequisite_plan,
    verify_stage,
    resume_config,
    exclusive_writer,
    STAGES,
    verify_generator_inputs,
)
from ncm_lab.runtime.budget import Budget, Meter
from ncm_lab.runtime.io import root, read, write_json, atomic_json, reference, digest, resolve_reference


def stop_child(child):
    """Only a child created by this launcher; cleanup timeout never limits training."""
    if child.poll() is not None:
        return
    if os.name == "nt":
        command = [
            str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"),
            "/PID",
            str(child.pid),
            "/T",
            "/F",
        ]
        result = subprocess.run(
            command, capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode and child.poll() is None:
            raise RuntimeError("could not stop owned training tree; inspect retained process.json")
    else:
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait(timeout=5)


def stage_child(value, folder, meter, resume=None, compatibility=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "resolved_config.json"
    write_json(path, value, exclusive=True)
    stage, action = STAGES[value.get("stage", "generator")]
    command = [sys.executable, "-m", "ncm_lab", stage, action, "--config", str(path)]
    if resume:
        command += ["--resume", str(Path(resume).resolve())]
        if compatibility:
            command += ["--recovery-compatibility", compatibility]
    with (folder / "stdout.json").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env={**os.environ, "NCM_CHILD_LAUNCH_RECORD": str(folder / "child-launch.json")},
            cwd=root(),
            shell=False,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            if os.name == "nt"
            else 0,
            start_new_session=os.name != "nt",
        )
        write_json(
            folder / "process.json",
            {"pid": child.pid, "command": command, "outer_training_deadline": None},
            exclusive=True,
        )
        try:
            while child.poll() is None:
                meter.check_disk()
                time.sleep(0.2)
        finally:
            stop_child(child)
        write_json(folder / "exit.json", {"exit_status": child.returncode}, exclusive=True)
    result = read(folder / "stdout.json")
    if not isinstance(result, dict):
        raise ValueError("training did not return its JSON result; inspect retained stdout/stderr")
    if child.returncode:
        status = result.get("status")
        return {
            **result,
            "status": status
            if status in {"failed", "blocked", "cancelled", "budget_stopped", "incomplete"}
            else "failed",
            "child_exit_status": child.returncode,
        }
    if result.get("status") != "completed":
        if result.get("status") in {"failed", "blocked", "cancelled", "budget_stopped", "incomplete"}:
            return result
        return {**result, "status": "failed", "error": "unrecognized child completion status"}
    run = Path(result["path"])
    manifest = read(run / "manifest.json")
    if (
        manifest.get("status") != "completed"
        or manifest.get("stage") != value.get("stage", "generator")
        or digest(run / "summary.json") != manifest["summary_sha256"]
        or digest(run / "resolved_config.json") != manifest["resolved_config_sha256"]
    ):
        raise ValueError("stage did not complete a sealed successful run")
    if any(result.get(key) != item for key, item in read(run / "summary.json").items()):
        raise ValueError("child output differs from its sealed summary")
    return result


def train_child(value, folder, meter, resume, compatibility):
    result = stage_child(value, folder, meter, resume, compatibility)
    if result.get("status") != "completed":
        return result
    if (
        result.get("completion", {}).get("code") != "objective_attained"
        or result.get("optimizer_steps") != value["training"]["steps"]
    ):
        raise ValueError("pilot did not complete the exact declared objective")
    artifact = Path(result["artifact"]).resolve()
    if (
        resolve_reference(value["output_ref"]) != artifact
        or read(artifact).get("schema") != "native_generator_export_v1"
    ):
        raise ValueError("training did not publish the expected native generator export")
    return result


def run_pilot(
    steps,
    *,
    config_path=None,
    resume=None,
    fresh=False,
    compatibility=None,
    port=7860,
    chat_seconds=7200,
    chat_requests=100,
    serve_fn=None,
    canvas_size=None,
    prepare=False,
    no_chat=False,
    plan_only=False,
    data_config=None,
):
    if steps not in {2, 100} or (resume and fresh) or (steps == 100 and not resume and not fresh):
        raise ValueError("100 step pilot requires --resume /ABS/checkpoints/current.json or explicit --fresh")
    if data_config and not prepare:
        raise ValueError("--data-config requires --prepare")
    if resume and prepare:
        raise ValueError("resume uses exact checkpoint-bound inputs; omit --prepare")
    if canvas_size is not None:
        resolve_canvas(canvas_size)  # Reject before model/child/chat allocation.
    explicit = (
        config(config_path, canvas_size if not resume else None, _bind_canvas=False) if config_path else None
    )
    if resume:
        value = resume_config(resume, explicit, canvas_size, target_steps=steps)
        value["training"].update(
            steps=steps,
            validation_interval=25 if steps == 100 else 1,
            checkpoint_interval=25 if steps == 100 else 1,
        )
        if not explicit:
            value["output_ref"] = str(Path(value["output_ref"]).parent / f"generator_{steps}_step.json")
            if steps == 100:
                value.setdefault("monitoring", {})["sample_interval"] = 25
        value["budget"]["enforce_wall_time_limits"] = False
        values = None
    else:
        value = explicit or config(f"configs/validation/generator_{steps}_step_pilot.yaml")
        value["budget"]["enforce_wall_time_limits"] = False
        if canvas_size is not None:
            value["architecture"] = (
                resolve_canvas(canvas_size).architecture(Architecture(**value["architecture"])).to_dict()
            )
        # Explicit fixture/legacy configurations retain their bound inputs unless preparation is requested.
        values = (
            pipeline_configs(value, data_config)
            if (not explicit or prepare or canvas_size is not None)
            else None
        )
        if values:
            value = values["generator"]
    if value["stage"] != "generator" or value["training"]["steps"] != steps or not value.get("output_ref"):
        raise ValueError("named pilot requires its exact generator step target and explicit handoff")
    if value["training"]["objective"] != "steps":
        raise ValueError(
            "named pilot requires a step objective, counted as completed generator optimizer updates"
        )
    from ncm_lab.config.canvas import validate_production

    if not value["model"].get("fixture"):
        validate_production(value["architecture"])
    geometry = canvas_metadata(value["architecture"])
    decisions = prerequisite_plan(values) if values else {}
    preparation_command = f"python scripts/run_{steps}_step_pilot.py --fresh --canvas-size {geometry['preset']} --prepare --no-chat"
    if data_config:
        preparation_command += f' --data-config "{Path(data_config).resolve()}"'
    plan = {
        "status": "planned",
        "canvas": geometry,
        "architecture": value["architecture"],
        "generator_steps": steps,
        "mode": "verified same-size resume" if resume else "fresh",
        "preparation": "requested" if prepare else "verified existing inputs",
        "reuse": decisions,
        "artifact_namespace": value.get("experiment_context", {}).get("namespace"),
        "clock_policy": {
            "outer_training_deadline": None,
            "stages": {
                name: v["budget"]["enforce_wall_time_limits"]
                for name, v in (values or {"generator": value}).items()
            },
        },
        "data_capacity": "unknown until actual reference qualification",
        "stage_objectives": {name: v["training"] for name, v in (values or {"generator": value}).items()},
        "stage_commands": {
            name: [
                sys.executable,
                "-m",
                "ncm_lab",
                *STAGES[name],
                "--config",
                f"<ATTEMPT>/{name}/resolved_config.json",
            ]
            for name in (values or {"generator": value})
        },
        "preparation_command": preparation_command,
    }
    print(
        f"Canvas preset: {geometry['preset']}\nAnswer canvas: {geometry['positions']} token positions, including a genuine EOS when present\n"
        f"Backbone: {value['model']['model_id']} (unchanged across canvas presets)\n"
        f"Hierarchy: {len(geometry['resolutions'])} updates, 4 -> {geometry['positions']} positions, latent width {value['architecture']['latent_width']}\n"
        f"Generator objective: {steps} total completed optimizer steps\n"
        f"Training clock limits: {'enabled' if value['budget']['enforce_wall_time_limits'] else 'disabled'}\n"
        f"Mode: {plan['mode']}\nPreparation: {plan['preparation']}",
        flush=True,
    )
    if plan_only:
        return plan
    if values and not prepare:
        for name, decision in decisions.items():
            if not decision["reuse"]:
                raise ValueError(
                    f"{geometry['preset']}/{geometry['positions']}: missing/incompatible {name}: {decision['reason']}. Prepare with: {preparation_command}"
                )
    if values is None and not resume:
        try:
            verify_generator_inputs(value)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(
                f"{geometry['preset']}/{geometry['positions']}: incompatible/missing targets: {exc}. Prepare with: {preparation_command}"
            ) from exc
    if not no_chat and find_spec("werkzeug") is None:
        raise ImportError(
            "install the chatbot extra first: python -m pip install -e '.[chat]' (preserve CUDA Torch)"
        )
    if not no_chat:
        from ncm_lab.chat.server import Limits

        Limits(max_seconds=chat_seconds, max_requests=chat_requests)
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("valid loopback chatbot port required")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    folder = (
        Path(value.get("run_root", root() / "runs"))
        / "pilots"
        / (geometry["preset"] if values else "explicit")
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    )
    folder.mkdir(parents=True, exist_ok=False)
    budget = Budget(
        "native-canvas-pilot-launcher-v2",
        enforce_wall_time_limits=False,
        max_steps=8,
        max_phase_steps=4000,
        max_storage_bytes=10_000_000,
        max_phase_storage_bytes=1_000_000_000,
    )
    meter = Meter(budget, folder, Path(value.get("run_root", root() / "runs")) / "ledgers")
    state = {
        "schema": "native_pilot_launcher_v1",
        "pilot": f"{steps} training step pilot",
        "status": "training",
        "resume": str(resume) if resume else None,
        "outer_training_deadline": None,
        "canvas": geometry,
    }
    atomic_json(folder / "status.json", state)
    try:
        write_json(folder / "plan.json", plan, exclusive=True)
        if values:
            with exclusive_writer(Path(value["experiment_context"]["namespace"]) / "pipeline.lock"):
                for name in ("data", "teacher", "codec", "targets"):
                    try:
                        verify_stage(values[name])  # Re-evaluate after any upstream rebuild.
                        continue
                    except (OSError, ValueError, KeyError):
                        if not prepare:
                            raise ValueError(
                                f"{name}: matching prerequisites changed; prepare with: {preparation_command}"
                            ) from None
                    meter.charge(steps=1)
                    result = stage_child(values[name], folder / name, meter)
                    if result.get("status") != "completed":
                        state.update(
                            status=result.get("status", "failed"),
                            failed_stage=name,
                            preparation=result,
                            chat_started=False,
                        )
                        atomic_json(folder / "status.json", state)
                        return state
                    verify_stage(values[name])
                # Pin to immutable manifests before releasing the preparation lock.
                from ncm_lab.config.stages import bind_inputs

                bound, missing = bind_inputs(value)
                if missing:
                    raise ValueError(str(missing))
                value["inputs"] = {key: item["manifest"] for key, item in bound.items()}
        meter.charge(steps=1)
        print(f"Starting {steps} training step pilot. Logs: {folder}", flush=True)
        result = train_child(value, folder / "generator", meter, resume, compatibility)
        state.update(training=result, training_status=result.get("status"))
        if result.get("status") != "completed":
            state.update(status=result.get("status", "failed"), chat_started=False)
            atomic_json(folder / "status.json", state)
            return state
        meter.charge(steps=1)
        state.update(status="starting_chat", export=reference(result["artifact"]))
        state["chat_command"] = (
            f'ncm chat serve --checkpoint "{Path(result["artifact"]).resolve()}" --device {value["device"]} --profile {value["hardware_profile"]}'
        )
        print("Export: " + str(Path(result["artifact"]).resolve()) + "\n" + state["chat_command"], flush=True)
        if no_chat:
            state.update(status="completed", chat_started=False)
            atomic_json(folder / "status.json", state)
            return state
        atomic_json(folder / "status.json", state)
        meter.check_disk()
        # The training interpreter has exited, freeing its optimizer/gradients.
        meter.close()
        if serve_fn is None:
            from ncm_lab.chat.server import serve as serve_fn

        def ready(chat):
            state.update(status="ready", chat_started=True, chat=chat)
            atomic_json(folder / "status.json", state)

        chat = serve_fn(
            checkpoint=result["artifact"],
            device=value["device"],
            profile=value["hardware_profile"],
            port=port,
            max_seconds=chat_seconds,
            max_requests=chat_requests,
            run_root=value.get("run_root"),
            on_ready=ready,
        )
        state.update(status="completed", chat=chat, training_status="completed")
        atomic_json(folder / "status.json", state)
        return state
    except KeyboardInterrupt:
        state.update(status="cancelled", chat_started=False)
        atomic_json(folder / "status.json", state)
        return state
    except Exception as exc:
        state.update(status="failed", error=str(exc), chat_started=False)
        atomic_json(folder / "status.json", state)
        raise
    finally:
        meter.close()


def script_main(steps):
    parser = argparse.ArgumentParser(description=f"{steps} training step pilot, then local chatbot")
    add_arguments(parser)
    args = parser.parse_args()
    try:
        result = run_pilot(steps, **vars(args))
        print(__import__("json").dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] in {"completed", "planned"} else 2
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print("Pilot: " + str(exc), file=sys.stderr)
        return 2


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--fresh", action="store_true")
    parser.add_argument("--config", dest="config_path", type=Path)
    parser.add_argument(
        "--recovery-compatibility",
        dest="compatibility",
        choices=["clock-policy-v1", "pilot-chat-v1"],
        default=None,
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--chat-seconds", type=float, default=7200)
    parser.add_argument("--chat-requests", type=int, default=100)
    add_canvas_argument(parser)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--data-config", type=Path)
