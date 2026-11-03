"""Persistent aggregate charges survive failed runs, old checkpoints, and new run IDs."""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import time
import uuid
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
    enforce_wall_time_limits: bool = True

    def __post_init__(self):
        if not self.phase_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in self.phase_id
        ):
            raise ValueError("safe phase ID required")
        for key, value in asdict(self).items():
            if key == "phase_id":
                continue
            if key == "enforce_wall_time_limits":
                if type(value) is not bool:
                    raise ValueError("enforce_wall_time_limits must be boolean")
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

    def wall_policy(self):
        return {
            "enforce_wall_time_limits": self.enforce_wall_time_limits,
            "status": "enforced" if self.enforce_wall_time_limits else "disabled_by_policy",
            "numeric_wall_limits_active": self.enforce_wall_time_limits,
            "legacy_max_wall_seconds": self.max_wall_seconds,
            "legacy_max_phase_wall_seconds": self.max_phase_wall_seconds,
        }


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
        self.settled_elapsed = None
        self.attempt = uuid.uuid4().hex
        self.wall_reservation = budget.max_wall_seconds if budget.enforce_wall_time_limits else 0
        with self.transaction() as state:
            caps = {k: v for k, v in asdict(budget).items() if k.startswith("max_phase_")}
            if state and state["caps"] != caps:
                raise ValueError("aggregate budget caps cannot change under the same phase ID")
            if not state:
                state.update(caps=caps, wall=0.0, charges={k: 0 for k in self.used}, runs=[])
            if "legacy_wall_balance_seconds" not in state:
                rows = list(state.get("wall_attempts", {}).values())
                state["legacy_wall_balance_seconds"] = max(
                    0.0,
                    state["wall"]
                    - sum(
                        row["elapsed_seconds"] if row["status"] == "closed" else row["reserved_seconds"]
                        for row in rows
                    ),
                )
            if budget.enforce_wall_time_limits:
                if incomplete_unbounded(state):
                    raise BudgetStop(
                        "aggregate elapsed accounting incomplete: unresolved disabled-clock attempt; "
                        "finite wall admission cannot be established"
                    )
                if state["wall"] + self.wall_reservation > budget.max_phase_wall_seconds:
                    raise BudgetStop("aggregate wall reservation exhausted")
            # Finite crashes retain the reservation. Disabled-clock crashes have an
            # explicit unknown duration and prevent later finite-mode admission.
            state["wall"] += self.wall_reservation
            state["runs"].append(str(self.run_dir.resolve()))
            state.setdefault("wall_attempts", {})[self.attempt] = {
                "run": str(self.run_dir.resolve()),
                "enforce_wall_time_limits": budget.enforce_wall_time_limits,
                "reserved_seconds": self.wall_reservation,
                "elapsed_seconds": None,
                "status": "open",
            }

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
        if self.budget.enforce_wall_time_limits and self.elapsed_seconds() >= self.budget.max_wall_seconds:
            raise BudgetStop("per-run wall budget exhausted")

    def elapsed_seconds(self):
        if self.settled_elapsed is not None:
            return self.settled_elapsed
        return max(0.0, time.perf_counter() - self.started)

    def wall_report(self):
        return {
            **self.budget.wall_policy(),
            "attempt_reserved_seconds": self.wall_reservation,
            "attempt_observed_seconds": self.elapsed_seconds(),
            "phase": phase_wall_report(read(self.path)),
        }

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
            elapsed = self.elapsed_seconds()
            with self.transaction() as state:
                entry = state["wall_attempts"][self.attempt]
                if entry["status"] != "closed":
                    state["wall"] += elapsed - self.wall_reservation
                    entry.update(status="closed", elapsed_seconds=elapsed)
            self.closed = True
            self.settled_elapsed = elapsed


def incomplete_unbounded(state):
    return [
        key
        for key, row in state.get("wall_attempts", {}).items()
        if not row["enforce_wall_time_limits"] and row["status"] != "closed"
    ]


def phase_wall_report(state):
    """Do not pretend legacy combined charges can be split or open durations are known."""
    rows = list(state.get("wall_attempts", {}).values())
    observed = sum(row["elapsed_seconds"] for row in rows if row["status"] == "closed")
    reserved = sum(row["reserved_seconds"] for row in rows if row["status"] != "closed")
    legacy = state.get("legacy_wall_balance_seconds", max(0.0, state["wall"] - observed - reserved))
    incomplete = incomplete_unbounded(state)
    return {
        "charged_and_reserved_seconds": state["wall"],
        "new_attempts_closed_observed_seconds": observed,
        "new_attempts_active_reserved_seconds": reserved,
        "legacy_charges_and_reservations_seconds": legacy,
        "incomplete_unbounded_attempts": incomplete,
        "elapsed_accounting_status": "incomplete_open_disabled_clock"
        if incomplete
        else "settled_or_finite_reserved",
        "legacy_split_status": "unknown" if legacy else "no_legacy_balance",
    }
