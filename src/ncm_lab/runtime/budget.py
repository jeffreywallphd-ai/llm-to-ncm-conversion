"""Persistent aggregate charges survive failed runs, old checkpoints, and new run IDs."""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import time
from ncm_lab.runtime.io import read, atomic_json


class BudgetStop(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    phase_id: str
    max_wall_seconds: float = 120
    max_steps: int = 200
    max_source_tokens: int = 1000000
    max_answer_positions: int = 1000000
    max_canvas_positions: int = 10000000
    max_storage_bytes: int = 512000000
    max_phase_wall_seconds: float = 10800
    max_phase_steps: int = 10000
    max_phase_source_tokens: int = 100000000
    max_phase_answer_positions: int = 100000000
    max_phase_canvas_positions: int = 1000000000
    max_phase_storage_bytes: int = 4000000000

    def __post_init__(self):
        if not self.phase_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in self.phase_id
        ):
            raise ValueError("safe phase ID required")
        for key, value in asdict(self).items():
            if key == "phase_id":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("positive finite budget required: " + key)
        for field in (
            "wall_seconds",
            "steps",
            "source_tokens",
            "answer_positions",
            "canvas_positions",
            "storage_bytes",
        ):
            if getattr(self, "max_phase_" + field) < getattr(self, "max_" + field):
                raise ValueError("phase cap below per-run limit")


class Meter:
    def __init__(self, budget, run_dir, ledger_root):
        self.budget = budget
        self.run_dir = Path(run_dir)
        self.started = time.perf_counter()
        self.path = Path(ledger_root) / (budget.phase_id + ".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock")
        self.used = {"steps": 0, "source_tokens": 0, "answer_positions": 0, "canvas_positions": 0}
        self.closed = False
        with self.transaction() as state:
            caps = {k: v for k, v in asdict(budget).items() if k.startswith("max_phase_")}
            if state and state["caps"] != caps:
                raise ValueError("aggregate budget caps cannot change under the same phase ID")
            if not state:
                state.update(caps=caps, wall=0.0, charges={k: 0 for k in self.used}, runs=[])
            if state["wall"] + budget.max_wall_seconds > budget.max_phase_wall_seconds:
                raise BudgetStop("aggregate wall reservation exhausted")
            # An unhandled crash retains this complete conservative reservation.
            state["wall"] += budget.max_wall_seconds
            state["runs"].append(str(self.run_dir.resolve()))

    @contextmanager
    def transaction(self):
        try:
            with self.lock.open("x"):
                pass
        except FileExistsError as exc:
            raise ValueError("phase ledger locked; inspect interrupted writer before recovery") from exc
        try:
            state = read(self.path) if self.path.exists() else {}
            yield state
            atomic_json(self.path, state)
        finally:
            self.lock.unlink(missing_ok=True)

    def check(self):
        if time.perf_counter() - self.started >= self.budget.max_wall_seconds:
            raise BudgetStop("per-run wall budget exhausted")

    def charge(self, source=0, answer=0, canvas=0, steps=0):
        self.check()
        additions = {
            "source_tokens": source,
            "answer_positions": answer,
            "canvas_positions": canvas,
            "steps": steps,
        }
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in additions.values()):
            raise ValueError("nonnegative integer work charges required")
        with self.transaction() as state:
            for key, value in additions.items():
                if self.used[key] + value > getattr(self.budget, "max_" + key) or state["charges"][
                    key
                ] + value > getattr(self.budget, "max_phase_" + key):
                    raise BudgetStop("finite " + key + " budget exhausted")
            for key, value in additions.items():
                state["charges"][key] += value
                self.used[key] += value

    def check_disk(self, additional=0):
        self.check()

        def size(folder):
            return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())

        if size(self.run_dir) + additional > self.budget.max_storage_bytes:
            raise BudgetStop("per-run storage budget exhausted")
        state = read(self.path)
        if sum(size(p) for p in set(state["runs"])) + additional > self.budget.max_phase_storage_bytes:
            raise BudgetStop("aggregate retained storage budget exhausted")

    def close(self):
        if not self.closed:
            elapsed = time.perf_counter() - self.started
            with self.transaction() as state:
                state["wall"] += elapsed - self.budget.max_wall_seconds
            self.closed = True
