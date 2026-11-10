"""Read-only, bounded follower for an exact block8 experiment, including older runs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from ncm_lab.monitoring.step_progress import StepEstimate, format_step


MAX_RECORD = 65536
TERMINAL = {"completed", "failed", "budget_stopped", "incomplete"}
PREPARATION = {
    "prepare": (
        "checking source data and tokenizer",
        "building complete-answer records and eight-token blocks",
        "finalizing the block dataset",
        "dataset/parents.sqlite",
        "dataset/manifest.json",
    ),
    "teacher": (
        "checking the block dataset and loading the frozen teacher",
        "caching frozen teacher features",
        "finalizing the teacher-feature cache",
        "cache/index.sqlite",
        "cache/manifest.json",
    ),
    "targets": (
        "checking codec/cache inputs and loading frozen models",
        "encoding frozen latent targets",
        "finalizing the target-code cache",
        "cache/index.sqlite",
        "cache/manifest.json",
    ),
}


def preparation_phase(stage, attempt):
    """Use producer artifact milestones; never inspect data rows or invent finer boundaries."""
    if stage == "source":
        artifacts = attempt / "artifacts"
        if (attempt / "qualification.json").is_file():
            return "finalizing qualified source data"
        if (artifacts / "candidates").is_dir() or (artifacts / "locked").is_dir():
            return "publishing qualified training and evaluation data"
        if (artifacts / "preparation.sqlite").is_file() or (artifacts / "candidates.sqlite").is_file():
            return "building the source index (overlap checks, filtering, tokenization, deduplication and grouping)"
        return "checking source inputs and tokenizer"
    setup, work, finish, index, manifest = PREPARATION[stage]
    if (attempt / manifest).is_file():
        return finish
    return work if (attempt / index).is_file() else setup


def small_json(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(2097153)
    if len(payload) > 2097152:
        raise ValueError("monitor metadata exceeds its byte allowance")
    return json.loads(payload), payload


class StepView:
    def __init__(self, manifest, *, emit=print, clock=time.perf_counter, max_bytes=268435456):
        self.manifest = Path(manifest).resolve()
        self.folder = self.manifest.parent
        state, _ = small_json(self.manifest)
        if state.get("schema") != "hierarchical_block8_experiment_v1":
            raise ValueError("exact hierarchical block8 experiment manifest required")
        settings_file = Path(state["settings"]["manifest"]).resolve()
        if settings_file.parent != self.folder:
            raise ValueError("monitor settings must belong to the selected experiment")
        settings, payload = small_json(settings_file)
        if hashlib.sha256(payload).hexdigest() != state["settings"]["sha256"]:
            raise ValueError("monitor settings receipt changed")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive finite monitor read allowance required")
        self.targets = {"codec": settings["codec"]["steps"], "generator": settings["training"]["steps"]}
        self.emit, self.clock, self.max_bytes = emit, clock, max_bytes
        self.read_bytes = 0
        self.streams, self.last_stages, self.last_recoveries = {}, {}, {}
        self.last_preparation = {}
        self.last_status = None

    def attempt_folders(self, stage):
        parent = self.folder / "attempts" / stage
        attempts = []
        if parent.exists():
            for folder in parent.iterdir():
                if folder.is_dir():
                    attempts.append(folder)
                if len(attempts) > 24:
                    raise ValueError("monitor attempt discovery allowance exhausted")
        return sorted(attempts)

    def preparation(self, stage, value, state):
        if value["status"] != "running" or stage not in {"source", *PREPARATION}:
            return
        attempts = self.attempt_folders(stage)
        if not attempts or len(attempts) < state.get("attempts", {}).get(stage, 0):
            return  # The controller can announce a retry before its new attempt directory exists.

        def created(folder):
            config = folder / "resolved_config.json"
            return (config if config.exists() else folder).stat().st_mtime_ns, folder.name

        attempt = max(attempts, key=created)
        phase = preparation_phase(stage, attempt)
        if self.last_preparation.get(stage) != (attempt.name, phase):
            self.emit(f"data prep {stage}: {phase} (attempt={attempt.name})")
            self.last_preparation[stage] = attempt.name, phase

    def rows(self, path, initial):
        cursor = self.streams[path]
        size = path.stat().st_size
        if size < cursor["offset"]:
            raise ValueError("monitor source shrank; attempt history must remain append-only")
        with path.open("rb") as stream:
            start = max(0, size - MAX_RECORD) if initial else cursor["offset"]
            stream.seek(start)
            if initial and start:
                # Discard a leading fragment when attaching to a long existing log.
                fragment = stream.readline(MAX_RECORD + 1)
                self.read_bytes += len(fragment)
                if len(fragment) > MAX_RECORD or not fragment.endswith(b"\n"):
                    raise ValueError("monitor step record exceeds its byte allowance")
            result = []
            while True:
                position = stream.tell()
                payload = stream.readline(MAX_RECORD + 1)
                self.read_bytes += len(payload)
                if self.read_bytes > self.max_bytes:
                    raise ValueError("monitor cumulative read allowance exhausted; training is unaffected")
                if len(payload) > MAX_RECORD:
                    raise ValueError("monitor step record exceeds its byte allowance")
                if not payload or not payload.endswith(b"\n"):
                    cursor["offset"] = position
                    break
                result.append(json.loads(payload))
        return result[-1:] if initial else result

    def poll(self):
        state, _ = small_json(self.manifest)
        for stage, value in state.get("stages", {}).items():
            status = value["status"]
            if self.last_stages.get(stage) != status:
                self.emit(f"stage {stage}: {status}")
                self.last_stages[stage] = status
            self.preparation(stage, value, state)
        for stage in self.targets:
            for folder in self.attempt_folders(stage):
                path = folder / "metrics.jsonl"
                if not path.is_file():
                    continue
                initial = path not in self.streams
                if initial:
                    if len(self.streams) >= 24:
                        raise ValueError("monitor aggregate attempt allowance exhausted")
                    self.streams[path] = dict(offset=0, estimator=None)
                cursor = self.streams[path]
                rows = self.rows(path, initial)
                if cursor["estimator"] is None:
                    # A new file may have filled between polls. Never time its first batch as replay work.
                    rows = rows[-1:]
                for metric in rows:
                    step = metric["step"]
                    estimator = cursor["estimator"]
                    if estimator is None:
                        estimator = cursor["estimator"] = StepEstimate(
                            self.targets[stage], step, clock=self.clock
                        )
                        observed = estimator.snapshot()
                    else:
                        observed = estimator.advance(step)
                    if "progress" not in metric:
                        # Old trainers recorded no timestamps; estimate only new work observed after attachment.
                        metric["progress"] = observed
                        source = " estimate_source=viewer_observations"
                    else:
                        source = ""
                    self.emit(format_step(stage, metric, self.targets[stage], attempt=folder.name) + source)
        for stage, step in state.get("completed_steps", {}).items():
            if self.last_recoveries.get(stage) != step:
                self.emit(f"{stage} recovery checkpoint: step {step}")
                self.last_recoveries[stage] = step
        if self.last_status != state["status"]:
            self.emit(f"experiment: {state['status']}")
            self.last_status = state["status"]
            if state.get("failure"):
                self.emit("failure: " + json.dumps(state["failure"], allow_nan=False))
        return state["status"] in TERMINAL


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only block8 step progress; no timer-only output")
    parser.add_argument("--experiment", required=True, type=Path, help="exact experiment manifest.json")
    parser.add_argument("--once", action="store_true", help="show the current snapshot and exit")
    parser.add_argument("--max-polls", type=int, default=2592000, help="finite read-work cap; default 30 days")
    parser.add_argument("--max-read-bytes", type=int, default=268435456, help="aggregate metrics read cap")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_polls <= 2592000:
            raise ValueError("monitor poll allowance must be between 1 and 2592000")
        view = StepView(args.experiment, max_bytes=args.max_read_bytes, emit=lambda line: print(line, flush=True))
        for _ in range(args.max_polls):
            if view.poll() or args.once:
                return 0
            time.sleep(1)  # File discovery only; passing time never emits a progress report.
        print("Monitor read-work allowance exhausted; training is unaffected.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Monitor stopped: {exc}; training is unaffected.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
