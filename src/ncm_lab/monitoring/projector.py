"""Independent CPU-only incremental JSONL -> TensorBoard view. No training imports."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from ncm_lab.monitoring.records import MAX_RECORD_BYTES, validate_record
from ncm_lab.monitoring.layouts import LAYOUT, Heatmaps


def write_status(path, value):
    path = Path(path)
    pending = path.with_suffix(".pending")
    pending.write_text(
        json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **value}, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(pending, path)


def size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def filesystem_path(path):
    """TensorBoard's byte-path IO needs Windows extended paths for nested runs."""
    resolved = str(Path(path).resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


class Projector:
    def __init__(
        self,
        runs_dir,
        output,
        *,
        attempt_dir=None,
        run_id=None,
        attempt_id=None,
        max_bytes=2147483648,
        aggregate_bytes=8589934592,
        headroom_bytes=2147483648,
        history=2048,
        max_attempts=128,
        aggregate_root=None,
    ):
        # Production entry points set this before Python starts. Direct CPU tests
        # must also explicitly select an empty CUDA environment.
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("projector requires CUDA_VISIBLE_DEVICES='' before imports")
        self.root = Path(runs_dir).resolve()
        self.output = Path(output).resolve()
        self.source = Path(attempt_dir).resolve() if attempt_dir else None
        if self.source:
            if not self.source.is_relative_to(self.root) or not (self.source / "attempt.json").is_file():
                raise ValueError(
                    "explicit replay source must exist inside runs root and declare a compatible attempt"
                )
            source_meta = self.validate_attempt(self.source / "attempt.json")
            if (
                run_id
                and source_meta["run_id"] != run_id
                or attempt_id
                and source_meta["attempt_id"] != attempt_id
            ):
                raise ValueError("explicit replay source identity disagrees with requested filter")
            if self.output.is_relative_to(self.source) or self.source.is_relative_to(self.output):
                raise ValueError("derived projection must be separate from the canonical attempt")
        self.aggregate_root = (
            Path(aggregate_root).resolve()
            if aggregate_root
            else (
                self.root / "_monitoring"
                if self.output.is_relative_to(self.root / "_monitoring")
                else self.output.parent
            )
        )
        if not self.output.is_relative_to(self.aggregate_root):
            raise ValueError("projection must be inside its declared derived aggregate root")
        self.view_root = (
            self.output.parent
            if self.output.name == "projection"
            and self.output.parent.parent == self.aggregate_root
            and self.output.parent.name.startswith("view-")
            else self.output
        )
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = self.output / "tensorboard"
        self.events.mkdir()
        self.run_id, self.attempt_id = run_id, attempt_id
        self.max_bytes, self.aggregate_bytes, self.headroom = max_bytes, aggregate_bytes, headroom_bytes
        if min(max_bytes, aggregate_bytes) <= 0 or headroom_bytes < 0:
            raise ValueError("finite positive projection disk caps required")
        self.history, self.max_attempts = history, max_attempts
        self.writers = {}
        self.streams = {}
        self.maps = {}
        self.records = 0
        self.last_source_time = None
        self.degraded = None
        self.last_heatmap = {}
        self.snapshot_extents = None
        write_status(
            self.output / "projection.json",
            {
                "schema": "native_projection_v1",
                "runs_dir": str(self.root),
                "attempt_dir": str(self.source) if self.source else None,
                "max_bytes": max_bytes,
                "aggregate_bytes": aggregate_bytes,
                "headroom_bytes": headroom_bytes,
                "authoritative": False,
                "aggregate_root": str(self.aggregate_root),
            },
        )

    def check_disk(self):
        registered = {self.view_root}
        for pattern in ("*/projection.json", "*/projection/projection.json"):
            for marker in self.aggregate_root.glob(pattern):
                if marker.stat().st_size > MAX_RECORD_BYTES or not marker.resolve().is_relative_to(
                    self.aggregate_root
                ):
                    continue
                try:
                    meta = json.loads(marker.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                if meta.get("schema") == "native_projection_v1" and meta.get("authoritative") is False:
                    candidate = marker.parent.resolve()
                    if (
                        candidate.name == "projection"
                        and candidate.parent.parent == self.aggregate_root
                        and candidate.parent.name.startswith("view-")
                    ):
                        candidate = candidate.parent
                    registered.add(candidate)
        if size(self.view_root) >= self.max_bytes or sum(size(p) for p in registered) >= self.aggregate_bytes:
            raise RuntimeError("derived monitoring disk budget reached; canonical runs retained")
        if shutil.disk_usage(self.output).free < self.headroom:
            raise RuntimeError("monitor disk headroom reached; canonical/checkpoint space retained")

    @staticmethod
    def validate_attempt(path):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if (
            meta.get("schema") != "native_monitor_attempt_v1"
            or meta.get("run_id") != path.parent.name
            or meta.get("stage") != path.parent.parent.name
        ):
            raise ValueError("incompatible attempt metadata/path identity")
        for key in ("stage", "run_id", "attempt_id"):
            if (
                not isinstance(meta.get(key), str)
                or not meta[key]
                or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                    for c in meta[key]
                )
            ):
                raise ValueError("unsafe/missing attempt identity")
        if set(meta.get("streams", {})) != {"metrics", "events", "samples"}:
            raise ValueError("attempt must declare all canonical streams")
        return meta

    def discover(self):
        candidates = [self.source / "attempt.json"] if self.source else self.root.glob("*/*/attempt.json")
        for path in candidates:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if (
                not resolved.is_relative_to(self.root)
                or "_monitoring" in resolved.relative_to(self.root).parts
            ):
                continue
            meta = self.validate_attempt(path)
            key = meta["attempt_id"]
            if (
                key in self.writers
                or self.run_id
                and meta["run_id"] != self.run_id
                or self.attempt_id
                and key != self.attempt_id
            ):
                continue
            if len(self.writers) >= self.max_attempts:
                raise RuntimeError("active projection attempt cap reached; narrow run filter")
            # Validate paths before allocating a writer, never load model files.
            paths = {}
            for stream, name in meta["streams"].items():
                target = (path.parent / name).resolve()
                if (
                    stream not in {"metrics", "events", "samples"}
                    or target.parent != resolved.parent
                    or target.suffix != ".jsonl"
                ):
                    raise ValueError("unsafe declared monitoring stream")
                paths[stream] = target
            from torch.utils.tensorboard import SummaryWriter

            folder = self.events / meta["stage"] / meta["run_id"] / key
            writer = SummaryWriter(filesystem_path(folder), max_queue=128, flush_secs=5)
            writer.add_custom_scalars(LAYOUT)
            writer.add_text("overview/attempt", json.dumps(meta, indent=2), 0)
            self.writers[key] = writer
            self.maps[key] = Heatmaps(self.history)
            for stream, target in paths.items():
                self.streams[(key, stream)] = {"path": target, "offset": 0, "sequence": 0, "meta": meta}

    def consume(self, writer, row, key):
        step, wall = row["step"], row["wall_time"]
        self.last_source_time = max(self.last_source_time or wall, wall)
        for name, measure in row.get("metrics", {}).items():
            if measure["value"] is not None:
                writer.add_scalar(name, measure["value"], step, walltime=wall)
            writer.add_scalar(name + "/observation_count", measure["count"], step, walltime=wall)
            if measure["status"] not in {"valid", "finite_zero", "finite_nonzero"}:
                writer.add_text(name + "/status", measure["status"], step, walltime=wall)
        if row["stream"] == "events":
            writer.add_text("overview/events", json.dumps(row, indent=2), step, walltime=wall)
            if "duration_seconds" in row:
                writer.add_scalar(
                    "resources/" + row["status"] + "/wall_seconds",
                    row["duration_seconds"],
                    step,
                    walltime=wall,
                )
        elif row["stream"] == "samples":
            writer.add_text(
                "outputs/" + row.get("prompt_id", "sample"), json.dumps(row, indent=2), step, walltime=wall
            )
        self.maps[key].add(row)
        self.last_heatmap[key] = (step, wall)
        self.records += 1

    def poll(self, max_records=128):
        self.check_disk()
        self.discover()
        consumed = 0
        catching_up = False
        for (key, stream), state in list(self.streams.items()):
            path = state["path"]
            if not path.exists():
                if self.snapshot_extents is not None and self.snapshot_extents.get(str(path), 0) > 0:
                    raise ValueError("captured source stream disappeared during replay")
                continue
            if path.stat().st_size < state["offset"]:
                raise ValueError("canonical stream was truncated")
            with path.open("rb") as source:
                source.seek(state["offset"])
                extent = (
                    self.snapshot_extents.get(str(path), 0)
                    if self.snapshot_extents is not None
                    else path.stat().st_size
                )
                if path.stat().st_size < extent:
                    raise ValueError("captured source extent was truncated")
                while consumed < max_records:
                    available = extent - source.tell()
                    if available <= 0:
                        break
                    line = source.readline(min(MAX_RECORD_BYTES + 1, available))
                    if not line:
                        break
                    if len(line) > MAX_RECORD_BYTES:
                        raise ValueError("canonical record exceeds bounded parser buffer")
                    if not line.endswith(b"\n"):
                        state["partial_tail_bytes"] = len(line)
                        break
                    state["partial_tail_bytes"] = 0
                    row = validate_record(
                        json.loads(
                            line,
                            parse_constant=lambda x: (_ for _ in ()).throw(
                                ValueError("nonstandard JSON constant")
                            ),
                        )
                    )
                    if (
                        row["attempt_id"] != key
                        or row["run_id"] != state["meta"]["run_id"]
                        or row["stream"] != stream
                        or row["sequence"] != state["sequence"] + 1
                        or row["config_sha256"] != state["meta"]["config_sha256"]
                        or row["stage"] != state["meta"]["stage"]
                    ):
                        raise ValueError("canonical stream identity/sequence mismatch")
                    self.consume(self.writers[key], row, key)
                    if consumed % 16 == 0:
                        self.check_disk()
                    state["offset"] = source.tell()
                    state["sequence"] = row["sequence"]
                    consumed += 1
                catching_up |= state["offset"] < extent
        for key, (step, wall) in self.last_heatmap.items():
            for kind in ("aux", "gates"):
                pixels = self.maps[key].image(kind)
                if pixels is not None:
                    self.writers[key].add_image(
                        "heatmaps/" + kind, pixels, step, walltime=wall, dataformats="HWC"
                    )
        self.last_heatmap.clear()
        for writer in self.writers.values():
            writer.flush()
        self.check_disk()
        status = {
            "state": "catching_up" if catching_up else "live" if self.writers else "waiting_for_runs",
            "attempts": len(self.writers),
            "records": self.records,
            "source_last_wall_time": self.last_source_time,
            "data_age_seconds": time.time() - self.last_source_time if self.last_source_time else None,
            "offsets": {str(v["path"]): v["offset"] for v in self.streams.values()},
            **self.progress(),
        }
        write_status(self.output / "status.json", status)
        return consumed

    def capture_snapshot(self):
        if self.source is None:
            raise ValueError("non-follow replay requires an explicit declared attempt")
        self.discover()
        self.snapshot_extents = {
            str(s["path"]): s["path"].stat().st_size if s["path"].exists() else 0
            for s in self.streams.values()
        }

    def progress(self):
        extents = (
            self.snapshot_extents
            if self.snapshot_extents is not None
            else {
                str(s["path"]): s["path"].stat().st_size if s["path"].exists() else 0
                for s in self.streams.values()
            }
        )
        remaining = {
            str(s["path"]): max(0, extents.get(str(s["path"]), 0) - s["offset"])
            for s in self.streams.values()
        }
        return {
            "scope": "captured source byte extents"
            if self.snapshot_extents is not None
            else "live declared streams",
            "captured_extents": extents,
            "offsets": {str(s["path"]): s["offset"] for s in self.streams.values()},
            "partial_tail_bytes": {
                str(s["path"]): 0 if not remaining[str(s["path"])] else s.get("partial_tail_bytes")
                for s in self.streams.values()
            },
            "remaining_by_stream": remaining,
            "remaining_bytes": sum(remaining.values()),
        }

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--attempt-dir")
    p.add_argument("--run-id")
    p.add_argument("--attempt-id")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--max-seconds", type=float, default=86400)
    p.add_argument("--max-bytes", type=int, default=2147483648)
    p.add_argument("--headroom-bytes", type=int, default=2147483648)
    p.add_argument("--poll-seconds", type=float, default=1)
    args = p.parse_args(argv)
    if (
        not math.isfinite(args.max_seconds)
        or not 0 < args.max_seconds <= 604800
        or not math.isfinite(args.poll_seconds)
        or not 0 < args.poll_seconds <= 60
    ):
        p.error("finite positive time limits required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    projector = None
    started = time.perf_counter()
    try:
        projector = Projector(
            args.runs_dir,
            args.output_dir,
            attempt_dir=args.attempt_dir,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            max_bytes=args.max_bytes,
            headroom_bytes=args.headroom_bytes,
        )
        if not args.follow:
            projector.capture_snapshot()
        outcome = "follow_stopped" if args.follow else "incomplete"
        reason = "configured lifetime ended"
        while time.perf_counter() - started < args.max_seconds:
            n = projector.poll()
            if not args.follow:
                if projector.progress()["remaining_bytes"] == 0:
                    outcome = "complete" if projector.records else "empty"
                    reason = "all declared captured bytes consumed"
                    break
                if n == 0:
                    reason = "unresolved partial tail in captured extent; no incomplete JSON parsed"
                    break
            if args.follow or n:
                time.sleep(min(args.poll_seconds, max(0, args.max_seconds - (time.perf_counter() - started))))
        progress = projector.progress()
        if not args.follow and not any(progress["captured_extents"].values()):
            outcome = "empty"
        write_status(
            projector.output / "status.json",
            {
                "state": outcome,
                "status": outcome,
                "outcome": outcome,
                "reason": reason,
                "records": projector.records,
                **progress,
            },
        )
        return 0 if outcome in {"complete", "empty", "follow_stopped"} else 3
    except Exception as exc:
        output = Path(args.output_dir)
        if projector is not None:
            write_status(
                output / "status.json",
                {
                    "state": "projector_error",
                    "status": "incomplete" if isinstance(exc, RuntimeError) else "invalid_source",
                    "outcome": "incomplete" if isinstance(exc, RuntimeError) else "invalid_source",
                    "error": type(exc).__name__ + ": " + str(exc),
                    **projector.progress(),
                },
            )
        print("monitor projector: " + str(exc), file=sys.stderr)
        return 2
    finally:
        if projector is not None:
            projector.close()


if __name__ == "__main__":
    raise SystemExit(main())
