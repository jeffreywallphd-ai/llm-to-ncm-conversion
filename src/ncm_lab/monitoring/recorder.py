"""Append-only canonical records through the run's provenance/budget boundary."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import time
import uuid
from ncm_lab.monitoring.records import Monitoring, Measure, SCHEMA, MAX_RECORD_BYTES, mappings
from ncm_lab.runtime.io import canonical_hash, digest, write_json, disk_path
from ncm_lab.config.canvas import canvas_metadata


class Recorder:
    def __init__(self, run):
        self.run = run
        from ncm_lab.monitoring.records import Stage1Monitoring

        self.options = (Stage1Monitoring if run.value.get("stage1") else Monitoring)(
            **run.value.get("monitoring", {})
        )
        self.attempt = uuid.uuid4().hex
        self.sequence = {s: 0 for s in ("metrics", "events", "samples")}
        self.step = 0
        self.phase = "initialization"
        self.last_emission = time.monotonic()
        self.config_sha = digest(run.path / "resolved_config.json")
        self.paths = {s: run.path / ("monitor_" + s + ".jsonl") for s in self.sequence}
        write_json(
            run.path / "attempt.pending",
            {
                "schema": "native_monitor_attempt_v1",
                "run_id": run.id,
                "attempt_id": self.attempt,
                "stage": run.stage,
                "config_sha256": self.config_sha,
                "architecture_sha256": canonical_hash(run.value.get("architecture", {})),
                "canvas": canvas_metadata(run.value["architecture"]),
                "experiment": run.value.get("experiment_context"),
                **(
                    {
                        "stage1": run.value["stage1_context"],
                        "stage1_policy": run.value["stage1"],
                        "precision": run.value["dtype"],
                        "training": run.value["training"],
                    }
                    if run.value.get("stage1_context")
                    else {}
                ),
                "mapping": mappings(run.value["architecture"]),
                "inputs": run.resolved.get("resolved_inputs", {}),
                "settings": asdict(self.options),
                "streams": {s: p.name for s, p in self.paths.items()},
                "origin": "software_fixture" if run.value.get("model", {}).get("fixture") else "real_model",
                "lineage_file": "attempt_lineage.json",
                "budget": run.value["budget"],
            },
            exclusive=True,
        )
        os.replace(disk_path(run.path / "attempt.pending"), disk_path(run.path / "attempt.json"))

    def append(self, stream, *, metrics=None, split="diagnostic", terminal=False, **fields):
        self.sequence[stream] += 1
        now = datetime.now(timezone.utc)
        row = {
            "schema": SCHEMA,
            "stream": stream,
            "run_id": self.run.id,
            "attempt_id": self.attempt,
            "stage": self.run.stage,
            "phase": self.phase,
            "step": self.step,
            "sequence": self.sequence[stream],
            "utc": now.isoformat(),
            "wall_time": now.timestamp(),
            "elapsed_seconds": time.perf_counter() - self.run.started,
            "split": split,
            "origin": "software_fixture" if self.run.value.get("model", {}).get("fixture") else "real_model",
            "canvas": canvas_metadata(self.run.value["architecture"]),
            "config_sha256": self.config_sha,
            **fields,
        }
        if metrics is not None:
            row["metrics"] = {k: v.json() for k, v in metrics.items()}
        encoded = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError("canonical monitoring record exceeds bounded size")
        if self.run.meter and stream != "events":
            from ncm_lab.runtime.budget import BudgetStop

            try:
                self.run.meter.check_disk(len(encoded))
            except BudgetStop as exc:
                if not terminal or "wall" not in str(exc):
                    raise
        with disk_path(self.paths[stream]).open("ab") as f:
            f.write(encoded)
            f.flush()
        if metrics and any(v["status"] == "nonfinite" for v in row["metrics"].values()):
            self.append(
                "events",
                status="nonfinite_observation",
                names=[k for k, v in row["metrics"].items() if v["status"] == "nonfinite"],
            )
        return row

    def event(self, status, **fields):
        return self.append("events", status=status, **fields)

    def due(self, step, final=False):
        return self.options.enabled and (
            step <= self.options.first_steps
            or final
            or step % self.options.scalar_interval == 0
            or time.monotonic() - self.last_emission >= self.options.scalar_seconds
        )

    def detail_due(self, step):
        return self.options.enabled and (
            step <= self.options.first_steps or step % self.options.detail_interval == 0
        )

    def metric(self, metrics, split="train", **fields):
        if split == "train":
            self.last_emission = time.monotonic()
        return self.append("metrics", metrics=metrics, split=split, **fields)

    def lineage(self, parent, checkpoint, state):
        write_json(
            self.run.path / "attempt_lineage.json",
            {
                "parent_attempt": parent,
                "checkpoint": checkpoint,
                "restored_step": state["optimizer_steps"],
                "restored_counters": {
                    k: v
                    for k, v in state.items()
                    if k
                    in {"processed_source_tokens", "answer_position_exposures", "allocated_canvas_positions"}
                },
                "budget_policy": "persistent phase ledger; spending is never restored from checkpoint",
            },
            exclusive=True,
        )
        self.event(
            "attempt_restored" if checkpoint else "attempt_initialized",
            parent_attempt=parent,
            checkpoint=checkpoint,
            restored_step=state["optimizer_steps"],
            restored_counters={
                k: state[k]
                for k in (
                    "processed_source_tokens",
                    "answer_position_exposures",
                    "allocated_canvas_positions",
                )
            },
        )

    def durable(self):
        for path in self.paths.values():
            if disk_path(path).exists():
                with disk_path(path).open("ab") as f:
                    f.flush()
                    os.fsync(f.fileno())


def scalar(value, unit="dimensionless", **kwargs):
    return Measure(value, unit=unit, **kwargs)
