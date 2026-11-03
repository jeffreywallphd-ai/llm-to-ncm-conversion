"""Bounded loopback-only Werkzeug UI. Preview imports no Torch and performs no inference."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import secrets
import threading
import time
import uuid
from werkzeug.wrappers import Request, Response
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server, WSGIRequestHandler
from ncm_lab.data.tokenization import validated_messages
from ncm_lab.runtime.budget import Budget, Meter, BudgetStop
from ncm_lab.runtime.io import root, write_json, atomic_json


@dataclass(frozen=True)
class Limits:
    max_seconds: float = 7200
    max_requests: int = 100
    max_http_requests: int = 10000
    max_body_bytes: int = 131072

    def __post_init__(self):
        if (
            isinstance(self.max_seconds, bool)
            or not math.isfinite(self.max_seconds)
            or not 0 < self.max_seconds <= 86400
        ):
            raise ValueError("chat lifetime must be finite, positive and at most one day")
        for name in ("max_requests", "max_http_requests", "max_body_bytes"):
            if type(getattr(self, name)) is not int or not 0 < getattr(self, name) <= 1_000_000:
                raise ValueError("positive bounded chat limit required: " + name)


class App:
    def __init__(self, backend=None, *, limits=None, port=7860):
        self.backend, self.limits, self.port = backend, limits or Limits(), port
        self.started = time.monotonic()
        self.requests = self.http_requests = 0
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.accounting_lock = threading.Lock()
        self.meter = None

    def status(self):
        return {
            "mode": "model" if self.backend is not None else "preview",
            "model": self.backend.describe() if self.backend is not None else None,
            "requests": self.requests,
            "max_requests": self.limits.max_requests,
            "max_seconds": self.limits.max_seconds,
            "token": self.token,
        }

    def dispatch(self, request):
        allowed = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if request.host not in allowed:
            return {"error": "This playground accepts only its local host."}, 403
        with self.accounting_lock:
            self.http_requests += 1
            if (
                self.http_requests > self.limits.max_http_requests
                or time.monotonic() - self.started >= self.limits.max_seconds
            ):
                return {"error": "This local server reached its session budget. Restart it to continue."}, 429
            if self.meter:
                self.meter.charge(steps=1)
                self.meter.check_disk()
        if request.method == "GET" and request.path == "/api/status":
            return self.status(), 200
        assets = {
            "/": ("index.html", "text/html"),
            "/assets/style.css": ("style.css", "text/css"),
            "/assets/app.js": ("app.js", "text/javascript"),
        }
        if request.method == "GET" and request.path in assets:
            name, mime = assets[request.path]
            return Response((Path(__file__).parent / "web" / name).read_text(encoding="utf-8"), mimetype=mime)
        if request.path != "/api/chat" or request.method != "POST":
            return {"error": "Not found"}, 404
        origin = request.headers.get("Origin")
        if (origin and origin != f"http://{request.host}") or not secrets.compare_digest(
            request.headers.get("X-NCM-Token", ""), self.token
        ):
            return {"error": "Refresh the local playground before sending a message."}, 403
        if (
            request.content_length is None
            or request.content_length > self.limits.max_body_bytes
            or request.mimetype != "application/json"
        ):
            return {"error": "Send a bounded JSON message."}, 413
        data = request.get_json()
        if not isinstance(data, dict) or set(data) != {"messages"}:
            return {"error": "Expected a messages list."}, 400
        messages = validated_messages(data["messages"], max_messages=41, max_chars=16384)
        if not self.lock.acquire(blocking=False):
            return {"error": "The model is answering another request. Try again shortly."}, 409
        try:
            if self.requests >= self.limits.max_requests:
                return {
                    "error": "The session request limit was reached. Restart the server to continue."
                }, 429
            self.requests += 1
            if self.backend is None:
                return {
                    "mode": "preview",
                    "text": "Your message reached the playground. No model is attached, so no inference was run. Connect a trained export to see real model responses.",
                }, 200
            return self.backend.respond(messages), 200
        finally:
            self.lock.release()

    @Request.application
    def __call__(self, request):
        request.max_content_length = self.limits.max_body_bytes
        try:
            result = self.dispatch(request)
            if isinstance(result, Response):
                response = result
            else:
                body, status = result
                response = Response(
                    json.dumps(body, allow_nan=False), status=status, mimetype="application/json"
                )
        except HTTPException as exc:
            response = Response(
                json.dumps({"error": exc.description}), status=exc.code, mimetype="application/json"
            )
        except (ValueError, BudgetStop) as exc:
            response = Response(
                json.dumps({"error": str(exc)[:600]}), status=400, mimetype="application/json"
            )
        except Exception:
            response = Response(
                json.dumps({"error": "Model inference failed. Inspect the retained chat run for details."}),
                status=500,
                mimetype="application/json",
            )
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
                "Referrer-Policy": "no-referrer",
            }
        )
        return response


def serve(
    *,
    checkpoint=None,
    preview=False,
    device="cpu",
    profile=None,
    host="127.0.0.1",
    port=7860,
    max_seconds=7200,
    max_requests=100,
    run_root=None,
    ready_file=None,
    on_ready=None,
):
    if bool(checkpoint) == bool(preview):
        raise ValueError("choose an explicit checkpoint or --preview")
    if host != "127.0.0.1" or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError(
            "chat serves on 127.0.0.1 with a valid local port only; use an SSH tunnel for remote access"
        )
    limits = Limits(max_seconds=max_seconds, max_requests=max_requests)
    folder = Path(run_root) if run_root else root() / "runs"
    session = (
        folder
        / "chat_servers"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    )
    session.mkdir(parents=True, exist_ok=False)
    write_json(
        session / "launch.json",
        {
            "preview": preview,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "limits": asdict(limits),
            "device": device,
        },
        exclusive=True,
    )
    backend = None

    # Bind before loading an expensive export; an occupied port must not disturb another service.
    class QuietHandler(WSGIRequestHandler):
        def log_request(self, *args, **kwargs):
            pass

    app = App(limits=limits, port=port)
    server = make_server(host, port, app, threaded=True, request_handler=QuietHandler)
    server.timeout = 0.25
    meter = None
    try:
        if checkpoint:
            from ncm_lab.chat.backend import ModelBackend

            backend = ModelBackend(checkpoint, device, profile, run_root=folder)
            app.backend = backend
        budget = Budget(
            "native-chat-server-v1",
            max_wall_seconds=max_seconds,
            max_phase_wall_seconds=86400,
            max_steps=limits.max_http_requests,
            max_phase_steps=100000,
            max_storage_bytes=1_000_000,
            max_phase_storage_bytes=100_000_000,
        )
        meter = Meter(budget, session, folder / "ledgers")
        app.meter = meter
        app.port = server.server_port
        app.started = time.monotonic()
        state = {
            "status": "ready",
            "url": f"http://127.0.0.1:{server.server_port}",
            "mode": "preview" if preview else "model",
            "model": backend.describe() if backend else None,
            "session": str(session),
            "limits": asdict(limits),
        }
        write_json(session / "ready.json", state, exclusive=True)
        if ready_file:
            write_json(ready_file, state, exclusive=True)
        print(json.dumps(state, allow_nan=False), flush=True)
        if on_ready:
            on_ready(state)
        while time.monotonic() - app.started < max_seconds and app.http_requests < limits.max_http_requests:
            server.handle_request()
        return {**state, "status": "stopped", "requests": app.requests}
    except KeyboardInterrupt:
        return {"status": "cancelled", "session": str(session), "requests": app.requests}
    except Exception as exc:
        write_json(session / "failure.json", {"error": str(exc)}, exclusive=True)
        raise
    finally:
        server.server_close()
        if meter:
            meter.close()
        atomic_json(
            session / "closed.json",
            {"requests": app.requests, "http_requests": app.http_requests, "closed": True},
        )
