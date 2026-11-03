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


def train_child(value, folder, meter, resume, compatibility):
    path = folder / "resolved_config.json"
    write_json(path, value, exclusive=True)
    command = [sys.executable, "-m", "ncm_lab", "generator", "train", "--config", str(path)]
    if resume:
        command += ["--resume", str(Path(resume).resolve()), "--recovery-compatibility", compatibility]
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
        return {**result, "status": "failed", "child_exit_status": child.returncode}
    if result.get("status") != "completed":
        return result
    run = Path(result["path"])
    manifest = read(run / "manifest.json")
    if (
        manifest.get("status") != "completed"
        or digest(run / "summary.json") != manifest["summary_sha256"]
        or digest(run / "resolved_config.json") != manifest["resolved_config_sha256"]
        or read(run / "summary.json").get("completion", {}).get("code") != "objective_attained"
        or result.get("completion", {}).get("code") != "objective_attained"
        or result.get("optimizer_steps") != value["training"]["steps"]
    ):
        raise ValueError("pilot did not complete the exact declared objective and sealed run")
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
    compatibility="pilot-chat-v1",
    port=7860,
    chat_seconds=7200,
    chat_requests=100,
    serve_fn=None,
):
    if steps not in {2, 100} or (resume and fresh) or (steps == 100 and not resume and not fresh):
        raise ValueError("100 step pilot requires --resume /ABS/checkpoints/current.json or explicit --fresh")
    if find_spec("werkzeug") is None:
        raise ImportError(
            "install the chatbot extra first: python -m pip install -e '.[chat]' (preserve CUDA Torch)"
        )
    from ncm_lab.chat.server import Limits

    Limits(max_seconds=chat_seconds, max_requests=chat_requests)
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("valid loopback chatbot port required")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    value = config(config_path or f"configs/validation/generator_{steps}_step_pilot.yaml")
    if value["stage"] != "generator" or value["training"]["steps"] != steps or not value.get("output_ref"):
        raise ValueError("named pilot requires its exact generator step target and explicit handoff")
    folder = (
        Path(value.get("run_root", root() / "runs"))
        / "pilots"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    )
    folder.mkdir(parents=True, exist_ok=False)
    budget = Budget(
        "native-pilot-launcher-v1",
        enforce_wall_time_limits=False,
        max_steps=4,
        max_phase_steps=4000,
        max_storage_bytes=10_000_000,
        max_phase_storage_bytes=1_000_000_000,
    )
    meter = Meter(budget, folder, folder.parents[1] / "ledgers")
    state = {
        "schema": "native_pilot_launcher_v1",
        "pilot": f"{steps} training step pilot",
        "status": "training",
        "resume": str(resume) if resume else None,
        "outer_training_deadline": None,
    }
    atomic_json(folder / "status.json", state)
    try:
        meter.charge(steps=1)
        print(f"Starting {steps} training step pilot. Logs: {folder}", flush=True)
        result = train_child(value, folder, meter, resume, compatibility)
        state.update(training=result, training_status=result.get("status"))
        if result.get("status") != "completed":
            state.update(status=result.get("status", "failed"), chat_started=False)
            atomic_json(folder / "status.json", state)
            return state
        meter.charge(steps=1)
        state.update(status="starting_chat", export=reference(result["artifact"]))
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
        return 0 if result["status"] == "completed" else 2
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
        default="pilot-chat-v1",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--chat-seconds", type=float, default=7200)
    parser.add_argument("--chat-requests", type=int, default=100)
