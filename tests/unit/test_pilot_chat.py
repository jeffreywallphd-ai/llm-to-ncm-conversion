"""UI protocol, honest preview, exact pilot configuration and post-training gates."""

from types import SimpleNamespace
import subprocess
import sys
import pytest
from werkzeug.test import Client
from ncm_lab.chat.server import App, Limits
from ncm_lab.config.stages import config
from ncm_lab.runtime import pilot
from ncm_lab.runtime.budget import Budget, Meter
from ncm_lab.runtime.io import atomic_json, implementation_hash
from ncm_lab.training.checkpoints import save_checkpoint, CheckpointReader, native_identity
from ncm_lab.training.recovery_compatibility import verify_recovery, CHAT_PREDECESSOR, CHAT_POLICY


def client(app=None):
    app = app or App()
    result = Client(app)
    result.base = {"base_url": "http://127.0.0.1:7860"}
    result.headers = {"X-NCM-Token": result.get("/api/status", **result.base).json["token"]}
    return result


def send(web, messages=None, **kwargs):
    return web.post(
        "/api/chat",
        json={"messages": messages or [{"role": "user", "content": "hello"}]},
        headers=web.headers,
        **web.base,
        **kwargs,
    )


def test_named_pilots_preserve_training_contract_and_separate_handoffs():
    old = config("configs/validation/generator_pilot.yaml")
    two = config("configs/validation/generator_2_step_pilot.yaml")
    hundred = config("configs/validation/generator_100_step_pilot.yaml")
    assert old["training"] == two["training"]
    assert two["training"]["steps"] == 2 and hundred["training"]["steps"] == 100
    changed = {k for k in two["training"] if two["training"][k] != hundred["training"][k]}
    assert changed == {"steps", "validation_interval", "checkpoint_interval"}
    assert hundred["training"]["validation_interval"] == hundred["training"]["checkpoint_interval"] == 25
    assert hundred["training"]["max_validation_examples"] == 32
    assert hundred["budget"] == old["budget"] and not hundred["budget"]["enforce_wall_time_limits"]
    assert hundred["inputs"] == old["inputs"]
    assert hundred["output_ref"] != old["output_ref"]
    assert hundred["monitoring"]["sample_interval"] == 25 and hundred["monitoring"]["sample_prompts"] == 4


def test_preview_is_explicit_and_has_no_fake_model_metrics():
    web = client()
    status = web.get("/api/status", **web.base).json
    assert status["mode"] == "preview" and status["model"] is None
    response = send(web)
    assert response.status_code == 200 and response.json["mode"] == "preview"
    assert "no inference" in response.json["text"] and "returned_tokens" not in response.json
    html = web.get("/", **web.base)
    assert html.status_code == 200 and "No model attached" in html.text
    assert "script-src 'self'" in html.headers["Content-Security-Policy"]
    assert web.get("/assets/app.js", **web.base).status_code == 200
    assert "textContent" in web.get("/assets/app.js", **web.base).text


def test_preview_can_import_and_respond_without_torch_or_transformers():
    code = """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','transformers'}: raise AssertionError('preview imported a model dependency')
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from ncm_lab.chat.server import App
from werkzeug.test import Client
c=Client(App()); base={'base_url':'http://127.0.0.1:7860'}
token=c.get('/api/status',**base).json['token']
r=c.post('/api/chat',json={'messages':[{'role':'user','content':'test'}]},headers={'X-NCM-Token':token},**base)
assert r.status_code==200 and r.json['mode']=='preview'
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "wrong"}],
        [{"role": "user", "content": " "}],
        [{"role": "user", "content": "hello", "reference": "secret"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    ],
)
def test_invalid_conversation_is_rejected(messages):
    web = client()
    response = web.post("/api/chat", json={"messages": messages}, headers=web.headers, **web.base)
    assert response.status_code == 400


def test_local_origin_body_budget_and_busy_checks():
    app = App(limits=Limits(max_requests=1))
    web = client(app)
    assert web.get("/api/status", base_url="http://untrusted.example:7860").status_code == 403
    assert web.post("/api/chat", json={"messages": []}, **web.base).status_code == 403
    assert (
        web.post(
            "/api/chat",
            json={"messages": []},
            headers={**web.headers, "Origin": "https://untrusted.example"},
            **web.base,
        ).status_code
        == 403
    )
    assert (
        web.post(
            "/api/chat", data="x" * 140000, headers=web.headers, content_type="application/json", **web.base
        ).status_code
        == 413
    )
    app.lock.acquire()
    try:
        assert send(web).status_code == 409
    finally:
        app.lock.release()
    assert send(web).status_code == 200
    assert send(web).status_code == 429
    app.started -= app.limits.max_seconds + 1
    assert web.get("/api/status", **web.base).status_code == 429


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 86401, True])
def test_chat_has_finite_lifetime(value):
    with pytest.raises(ValueError):
        Limits(max_seconds=value)


def test_one_backend_is_reused_for_multiple_turns():
    backend = SimpleNamespace(calls=[])
    backend.describe = lambda: {"label": "fixture"}

    def respond(messages):
        backend.calls.append(messages)
        return {"mode": "model", "text": "fixture response"}

    backend.respond = respond
    web = client(App(backend))
    assert send(web).json["mode"] == "model"
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior output"},
        {"role": "user", "content": "next"},
    ]
    assert send(web, messages).status_code == 200
    assert backend.calls[-1] == messages and len(backend.calls) == 2


@pytest.mark.parametrize("status", ["failed", "blocked", "budget_stopped", "incomplete", "cancelled"])
def test_launcher_never_starts_chat_after_unsuccessful_training(tmp_path, monkeypatch, status):
    cfg = config("configs/validation/generator_2_step_pilot.yaml")
    cfg["run_root"] = str(tmp_path / "runs")
    path = tmp_path / "config.json"
    atomic_json(path, cfg)
    monkeypatch.setattr(pilot, "train_child", lambda *args: {"status": status})
    result = pilot.run_pilot(
        2, config_path=path, port=0, serve_fn=lambda **kwargs: pytest.fail("chat started")
    )
    assert result["status"] == status and result["chat_started"] is False


def test_nonzero_child_exit_cannot_masquerade_as_training_success(tmp_path, monkeypatch):
    folder = tmp_path / "launcher"
    folder.mkdir()
    meter = Meter(Budget("nonzero", max_storage_bytes=10000000), folder, tmp_path / "ledgers")

    class Child:
        pid = 12345
        returncode = 2

        def __init__(self, *args, stdout, **kwargs):
            stdout.write(b'{"status":"completed"}')

        def poll(self):
            return self.returncode

    monkeypatch.setattr(pilot.subprocess, "Popen", Child)
    try:
        result = pilot.train_child({}, folder, meter, None, CHAT_POLICY)
        assert result["status"] == "failed" and result["child_exit_status"] == 2
    finally:
        meter.close()


def test_hundred_steps_requires_deliberate_resume_or_fresh():
    with pytest.raises(ValueError, match="requires --resume"):
        pilot.run_pilot(100)


def test_reviewed_recent_predecessor_needs_new_explicit_policy(tmp_path):
    previous = {"implementation_sha256": CHAT_PREDECESSOR, "model": "fixture", "phase_id": "same"}
    identity = native_identity(previous)
    saved = save_checkpoint(
        tmp_path,
        {
            "identity": identity,
            "metadata": previous,
            "scheduler": {"schema": "constant_separate_rates_v1"},
            "scaler": None,
        },
    )
    reader = CheckpointReader(saved["manifest"])
    expected = {**previous, "implementation_sha256": implementation_hash()}
    with pytest.raises(ValueError, match="unreviewed implementation"):
        verify_recovery(reader, expected, "clock-policy-v1")
    result = verify_recovery(reader, expected, CHAT_POLICY)
    assert result["predecessor_implementation_sha256"] == CHAT_PREDECESSOR
    assert len(result["reviewed_chain"]) == 1
