"""Owned child processes, loopback HTTP identity checks, bounded cleanup."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import uuid
from ncm_lab.monitoring.projector import write_status, filesystem_path


def doctor(output=None):
    versions = {}
    for distribution in ("tensorboard", "Pillow", "setuptools"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    output = Path(output or "runs/_monitoring").resolve()
    parent = next((p for p in (output, *output.parents) if p.exists()), None)
    return {
        "dependencies": versions,
        "compatible": versions.get("tensorboard") == "2.20.0"
        and versions.get("Pillow") is not None
        and importlib.util.find_spec("pkg_resources") is not None
        and importlib.util.find_spec("torch") is not None,
        "tensorflow_installed": importlib.util.find_spec("tensorflow") is not None,
        "output_parent_writable": bool(parent and os.access(parent, os.W_OK)),
        "output": str(output),
        "cuda_initialization": "not performed",
        "training_started": False,
        "install": "uv sync --locked --extra cpu --extra prepare --extra monitoring --group dev (CPU); see Spark instructions to preserve CUDA Torch",
    }


def child_environment():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}


def spawn(command, log):
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=child_environment(),
        shell=False,
        **kwargs,
    )


def stop_owned(child):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


class Session:
    def __init__(
        self,
        runs_dir,
        host="127.0.0.1",
        port=6006,
        *,
        startup_seconds=30,
        max_seconds=86400,
        max_bytes=2147483648,
        headroom_bytes=2147483648,
        run_id=None,
        attempt_id=None,
    ):
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("monitor listener must be loopback; use an SSH tunnel for Spark")
        if not 0 < port < 65536 or not 0 < startup_seconds <= 60 or not 0 < max_seconds <= 604800:
            raise ValueError("invalid monitor port or finite lifecycle budget")
        self.root = Path(runs_dir).resolve()
        self.host, self.port = host, port
        self.startup, self.duration = startup_seconds, max_seconds
        self.max_bytes, self.headroom = max_bytes, headroom_bytes
        self.run_id, self.attempt_id = run_id, attempt_id
        self.path = self.root / "_monitoring" / ("view-" + uuid.uuid4().hex)
        self.output = self.path / "projection"
        self.projector = self.server = None
        self.log = None

    def start(self):
        if not doctor()["compatible"]:
            raise ImportError("install the pinned optional monitoring extra; TensorFlow is not required")
        family = socket.AF_INET6 if self.host == "::1" else socket.AF_INET
        with socket.socket(family) as test:
            try:
                test.bind((self.host, self.port))
            except OSError as exc:
                raise ValueError(
                    f"monitor port {self.port} is occupied or unavailable; no service was changed"
                ) from exc
        self.path.mkdir(parents=True, exist_ok=False)
        self.log = (self.path / "server.log").open("wb")
        command = [
            sys.executable,
            "-m",
            "ncm_lab.monitoring.projector",
            "--runs-dir",
            str(self.root),
            "--output-dir",
            str(self.output),
            "--follow",
            "--max-seconds",
            str(self.duration),
            "--max-bytes",
            str(self.max_bytes),
            "--headroom-bytes",
            str(self.headroom),
        ]
        if self.run_id:
            command += ["--run-id", self.run_id]
        if self.attempt_id:
            command += ["--attempt-id", self.attempt_id]
        try:
            self.projector = spawn(command, self.log)
            self.server = spawn(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    filesystem_path(self.output / "tensorboard"),
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                    "--reload_interval",
                    "5",
                    "--load_fast=false",
                    "--window_title",
                    "Native training | canonical JSONL view",
                ],
                self.log,
            )
            started = time.monotonic()
            address = "[::1]" if self.host == "::1" else self.host
            self.url = f"http://{address}:{self.port}"
            while time.monotonic() - started < self.startup:
                if self.server.poll() is not None:
                    raise RuntimeError(
                        "TensorBoard exited before HTTP readiness; see " + str(self.path / "server.log")
                    )
                if self.projector.poll() is not None:
                    raise RuntimeError(
                        "projector exited before readiness; see " + str(self.output / "status.json")
                    )
                try:
                    with urlopen(self.url + "/data/logdir", timeout=0.5) as response:
                        logdir = json.load(response)["logdir"]
                    with urlopen(self.url + "/data/environment", timeout=0.5) as response:
                        environment = json.load(response)
                    if (
                        filesystem_path(logdir) == filesystem_path(self.output / "tensorboard")
                        and environment["version"] == "2.20.0"
                    ):
                        status = self.status()
                        if status["projector"].get("state") not in {
                            "waiting_for_runs",
                            "catching_up",
                            "live",
                        }:
                            time.sleep(0.1)
                            continue
                        write_status(self.path / "launcher.json", status)
                        return status
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.1)
            raise TimeoutError("monitor HTTP/log-directory readiness timed out")
        except BaseException:
            self.close()
            raise

    def status(self):
        path = self.output / "status.json"
        projection = json.loads(path.read_text()) if path.exists() else {"state": "starting"}
        if self.projector and self.projector.poll() is not None:
            projection = {**projection, "state": "projector_error", "exit_code": self.projector.returncode}
        return {
            "http_ready": bool(self.server and self.server.poll() is None),
            "url": getattr(self, "url", None),
            "event_logdir": str(self.output / "tensorboard"),
            "session": str(self.path),
            "projector": projection,
            "canonical_runs": str(self.root),
            "read_only_training": True,
        }

    def close(self):
        stop_owned(self.server)
        stop_owned(self.projector)
        if self.log:
            self.log.close()


def serve(**kwargs):
    session = Session(**kwargs)
    try:
        print(json.dumps(session.start(), indent=2), flush=True)
        started = time.monotonic()
        while time.monotonic() - started < session.duration:
            status = session.status()
            if not status["http_ready"] or status["projector"]["state"] == "projector_error":
                write_status(session.path / "launcher.json", status)
                raise RuntimeError(
                    "optional monitoring degraded; training continues; see " + str(session.path)
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return {"status": "monitor_stopped", "session": str(session.path), "training_signalled": False}
