"""Independent CPU-only incremental JSONL -> TensorBoard view. No training imports."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
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
    ):
        # Production entry points set this before Python starts. Direct CPU tests
        # must also explicitly select an empty CUDA environment.
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("projector requires CUDA_VISIBLE_DEVICES='' before imports")
        self.root = Path(runs_dir).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = self.output / "tensorboard"
        self.events.mkdir()
        self.source = Path(attempt_dir).resolve() if attempt_dir else None
        if self.source and not self.source.is_relative_to(self.root):
            raise ValueError("attempt outside configured runs root")
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
            },
        )

    def check_disk(self):
        aggregate_root = (
            self.root / "_monitoring"
            if self.output.is_relative_to(self.root / "_monitoring")
            else self.output.parent
        )
        if size(self.output) >= self.max_bytes or size(aggregate_root) >= self.aggregate_bytes:
            raise RuntimeError("derived monitoring disk budget reached; canonical runs retained")
        if shutil.disk_usage(self.output).free < self.headroom:
            raise RuntimeError("monitor disk headroom reached; canonical/checkpoint space retained")

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
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("schema") != "native_monitor_attempt_v1":
                raise ValueError("incompatible declared attempt schema")
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
                continue
            if path.stat().st_size < state["offset"]:
                raise ValueError("canonical stream was truncated")
            with path.open("rb") as source:
                source.seek(state["offset"])
                while consumed < max_records:
                    line = source.readline(MAX_RECORD_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_RECORD_BYTES:
                        raise ValueError("canonical record exceeds bounded parser buffer")
                    if not line.endswith(b"\n"):
                        break
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
                    ):
                        raise ValueError("canonical stream identity/sequence mismatch")
                    self.consume(self.writers[key], row, key)
                    if consumed % 16 == 0:
                        self.check_disk()
                    state["offset"] = source.tell()
                    state["sequence"] = row["sequence"]
                    consumed += 1
                catching_up |= source.tell() < path.stat().st_size
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
        }
        write_status(self.output / "status.json", status)
        return consumed

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
        started = time.monotonic()
        while time.monotonic() - started < args.max_seconds:
            n = projector.poll()
            if not args.follow and n == 0:
                break
            time.sleep(args.poll_seconds)
        return 0
    except Exception as exc:
        output = Path(args.output_dir)
        if projector is not None:
            write_status(
                output / "status.json",
                {"state": "projector_error", "error": type(exc).__name__ + ": " + str(exc)},
            )
        return 2
    finally:
        if projector is not None:
            projector.close()


if __name__ == "__main__":
    raise SystemExit(main())
