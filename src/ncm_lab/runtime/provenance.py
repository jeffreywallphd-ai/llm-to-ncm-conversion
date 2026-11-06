"""Immutable native runs, append-only observations, and independent budget ledgers."""

from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
import os
from ncm_lab.runtime.io import root, write_json, read, digest, implementation_hash, atomic_json, reference
from ncm_lab.runtime.budget import Budget, Meter
from ncm_lab.runtime.doctor import doctor
from ncm_lab.config.canvas import canvas_metadata
from ncm_lab.runtime.io import disk_path


class Run:
    def __init__(self, value, bound=None):
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        self.value = value
        self.pending_artifact = None
        self.stage = value["stage"]
        self.started = time.perf_counter()
        if value.get("run_root"):
            run_root = Path(value["run_root"])
        else:
            try:
                run_root = root() / "runs"
            except ValueError:
                run_root = Path.cwd() / "runs"
        self.path = run_root / self.stage / self.id
        disk_path(self.path).mkdir(parents=True, exist_ok=False)
        self.resolved = {**value, "resolved_inputs": bound or {}}
        write_json(self.path / "resolved_config.json", self.resolved, exclusive=True)
        write_json(
            self.path / "environment.json",
            doctor(value.get("hardware_profile", "cpu" if value["device"] == "cpu" else "cuda")),
            exclusive=True,
        )
        self.meter = None
        self.code = implementation_hash()
        write_json(
            self.path / "launch.json",
            {
                "schema": "native_launch_v1",
                "implementation_sha256": self.code,
                "resolved_config_sha256": digest(self.path / "resolved_config.json"),
                "budget": value["budget"],
            },
            exclusive=True,
        )
        if os.environ.get("NCM_CHILD_LAUNCH_RECORD"):
            write_json(
                os.environ["NCM_CHILD_LAUNCH_RECORD"],
                {
                    "run_id": self.id,
                    "path": str(self.path.resolve()),
                    "launch": reference(self.path / "launch.json"),
                },
                exclusive=True,
            )
        from ncm_lab.monitoring.recorder import Recorder

        self.recorder = Recorder(self)

    def start(self):
        self.meter = Meter(Budget(**self.value["budget"]), self.path, self.path.parents[1] / "ledgers")
        if self.value.get("stage3"):
            self.meter.stage3_run = self
        if self.value.get("stage1_io"):
            from ncm_lab.stage1.io import WriteMeter

            self.meter.write_meter = WriteMeter(self.value["stage1_io"], self.id)
        self.event(
            "started",
            wave=self.value.get("wave_context"),
            wall_policy=self.meter.wall_report(),
            wall_policy_message="Wall-clock limits: "
            + (
                "enabled"
                if self.meter.budget.enforce_wall_time_limits
                else "disabled; completion is governed by the configured work target."
            ),
        )
        return self.meter

    def event(self, status, **fields):
        from json import dumps

        with disk_path(self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                dumps({"utc": datetime.now(timezone.utc).isoformat(), "status": status, **fields}) + "\n"
            )
        self.recorder.event(status, **fields)

    def metric(self, value):
        from json import dumps

        with disk_path(self.path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(dumps(value, allow_nan=False) + "\n")
            stream.flush()

    def close(self, status, summary):
        from ncm_lab.runtime.memory import observe

        self.recorder.phase = status
        self.recorder.durable()
        if self.meter:
            self.meter.close()
        summary = {
            **summary,
            "canvas": canvas_metadata(self.value["architecture"]),
            "experiment": self.value.get("experiment_context"),
            **({"stage1": self.value["stage1_context"]} if self.value.get("stage1_context") else {}),
            "status": status,
            "elapsed_seconds": time.perf_counter() - self.started,
            "charged_work": self.meter.used if self.meter else {},
            "wall_policy": self.meter.wall_report() if self.meter else None,
            "origin": "software_fixture" if self.value.get("model", {}).get("fixture") else "real_model",
            "memory": observe(),
        }
        write_json(self.path / "summary.json", summary, exclusive=True)
        self.event(status, reason=summary.get("error", summary.get("reason", summary.get("stop_reason"))))
        self.recorder.durable()
        write_json(
            self.path / "manifest.json",
            {
                "schema": "native_run_v1",
                "run_id": self.id,
                "stage": self.stage,
                "implementation_sha256": self.code,
                "resolved_config_sha256": digest(self.path / "resolved_config.json"),
                "summary_sha256": digest(self.path / "summary.json"),
                "status": status,
                "research_claims": [],
            },
            exclusive=True,
        )
        if status == "completed" and self.pending_artifact:
            atomic_json(
                root() / self.value["output_ref"],
                {**reference(self.pending_artifact), "run": reference(self.path / "manifest.json")},
            )
        return {"run_id": self.id, "path": str(self.path), "status": status, **summary}

    def publish(self, artifact_manifest):
        if self.value.get("output_ref"):
            if self.value.get("experiment_context") or self.value.get("stage1_context"):
                self.pending_artifact = artifact_manifest
                return  # Only publish after completed run and hashes have been sealed.
            atomic_json(root() / self.value["output_ref"], reference(artifact_manifest))


def validate_run(path):
    path = Path(path)
    manifest = read(path / "manifest.json")
    if (
        manifest.get("schema") != "native_run_v1"
        or manifest["run_id"] != path.name
        or digest(path / "resolved_config.json") != manifest["resolved_config_sha256"]
        or digest(path / "summary.json") != manifest["summary_sha256"]
    ):
        raise ValueError("native run integrity failure")
    return manifest
