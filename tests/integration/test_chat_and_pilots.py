"""Actual tiny 100-step continuation, verified in-memory chat, and bounded live preview."""

import copy
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from urllib.request import urlopen, Request
import pytest
from werkzeug.test import Client
from ncm_lab.runtime.io import atomic_json, read, digest, resolve_reference
from ncm_lab.runtime.pilot import run_pilot, stop_child
from ncm_lab.chat.server import App
from ncm_lab.chat.backend import ModelBackend


def test_actual_100_step_resume_then_model_chat_after_child_exit(native_artifacts, tmp_path, monkeypatch):
    import ncm_lab.inference.export as exported

    value = copy.deepcopy(native_artifacts["configs"]["generator"])
    value["training"].update(steps=100, validation_interval=25, checkpoint_interval=25)
    value["monitoring"] = {"enabled": True, "sample_interval": 25, "sample_prompts": 1}
    value["output_ref"] = str(tmp_path / "100-step.json")
    source = tmp_path / "100-step-config.json"
    atomic_json(source, value)
    baseline = Path(native_artifacts["refs"]) / "generator.json"
    before = digest(baseline)
    parent = native_artifacts["results"]["generator"]["checkpoint"]
    calls = []
    loader = exported.load_export

    def counted(*args, **kwargs):
        calls.append(args[0])
        return loader(*args, **kwargs)

    monkeypatch.setattr(exported, "load_export", counted)

    def chat(*, checkpoint, device, profile, run_root, on_ready, **kwargs):
        runs = Path(value["run_root"])
        launch = max((runs / "pilots/explicit").glob("*"), key=lambda p: p.stat().st_mtime) / "generator"
        assert read(launch / "exit.json")["exit_status"] == 0
        result = read(launch / "stdout.json")
        assert result["optimizer_steps"] == 100 and result["charged_work"]["steps"] == 98
        assert result["completion"]["code"] == "objective_attained"
        assert resolve_reference(value["output_ref"]) == Path(checkpoint).resolve()
        backend = ModelBackend(checkpoint, device, profile, run_root=run_root, phase_id="chat-test")
        app = App(backend)
        client = Client(app)
        base = {"base_url": "http://127.0.0.1:7860"}
        token = client.get("/api/status", **base).json["token"]
        for text in ("token4", "token5"):
            response = client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": text}]},
                headers={"X-NCM-Token": token},
                **base,
            )
            assert response.status_code == 200, response.text
            assert response.json["mode"] == "model" and response.json["backbone_traversals"] == 1
        assert len(calls) == 1  # One model allocation/load for both requests.
        on_ready({"url": "http://127.0.0.1:7860", "mode": "model"})
        return {"status": "stopped", "requests": 2}

    result = run_pilot(100, config_path=source, resume=parent, port=0, serve_fn=chat)
    assert result["status"] == "completed" and result["training_status"] == "completed"
    assert result["chat_started"] is True and digest(baseline) == before
    training = Path(result["training"]["path"])
    rows = [json.loads(line) for line in (training / "monitor_samples.jsonl").read_text().splitlines()]
    assert {row["step"] for row in rows} == {25, 50, 75, 100}
    from ncm_lab.training.checkpoints import CheckpointReader

    reader = CheckpointReader(result["training"]["checkpoint"])
    state = reader.materialize(reader.field("training"))
    assert state["optimizer_steps"] == 100


def test_live_preview_starts_without_model_and_has_bounded_shutdown(tmp_path):
    ready = tmp_path / "ready.json"
    command = [
        sys.executable,
        "-m",
        "ncm_lab",
        "chat",
        "serve",
        "--preview",
        "--port",
        "0",
        "--max-seconds",
        "3",
        "--run-root",
        str(tmp_path / "runs"),
        "--ready-file",
        str(ready),
    ]
    with (tmp_path / "preview.log").open("wb") as stream:
        child = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert ready.exists(), (tmp_path / "preview.log").read_text()
            state = read(ready)
            assert state["mode"] == "preview" and state["model"] is None
            url = state["url"]
            with urlopen(url + "/api/status", timeout=3) as response:
                token = json.load(response)["token"]
            request = Request(
                url + "/api/chat",
                data=json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode(),
                headers={"Content-Type": "application/json", "X-NCM-Token": token},
            )
            with urlopen(request, timeout=3) as response:
                assert json.load(response)["mode"] == "preview"
            child.wait(timeout=10)
            assert child.returncode == 0
        finally:
            stop_child(child)
    assert read(Path(state["session"]) / "closed.json")["closed"] is True


def test_chat_context_and_export_integrity_fail_without_repair(native_artifacts, tmp_path):
    export = native_artifacts["results"]["generator"]["artifact"]
    backend = ModelBackend(export, run_root=tmp_path / "runs", phase_id="chat-limits")
    with pytest.raises(ValueError, match="512 prompt tokens"):
        backend.respond([{"role": "user", "content": "token4 " * 520}])
    # A new local manifest fixture with a wrong weights digest must fail loading.
    value = read(export)
    value["weights_sha256"] = "f" * 64
    value["weights_manifest"] = str((Path(export).parent / value["weights_manifest"]).resolve())
    path = tmp_path / "invalid-export.json"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="changed"):
        ModelBackend(path, run_root=tmp_path / "bad-runs", phase_id="bad-chat")
