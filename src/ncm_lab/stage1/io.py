"""Persistent checkpoint/cache write charges, independent of retained disk usage."""

from contextlib import contextmanager
from pathlib import Path
import uuid
from ncm_lab.runtime.io import read, atomic_json, disk_path
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.canvas_pipeline import exclusive_writer


class WriteMeter:
    def __init__(self, settings, attempt):
        if set(settings) != {"ledger", "max_run_bytes", "max_phase_bytes"}:
            raise ValueError("typed Stage 1 cumulative I/O allowance required")
        if any(
            type(settings[k]) is not int or settings[k] <= 0 for k in ("max_run_bytes", "max_phase_bytes")
        ):
            raise ValueError("finite positive cumulative I/O allowances required")
        self.settings, self.attempt = settings, attempt
        self.path = Path(settings["ledger"])
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def transaction(self):
        with exclusive_writer(self.path.with_suffix(".lock")):
            state = (
                read(self.path)
                if self.path.exists()
                else {
                    "schema": "stage1_write_ledger_v1",
                    "caps": self.settings,
                    "attempts": {},
                    "reservations": {},
                    "written_bytes": 0,
                }
            )
            if state["caps"] != self.settings:
                raise ValueError("Stage 1 cumulative write caps changed under the same ledger")
            yield state
            atomic_json(self.path, state)

    @contextmanager
    def account(self, path, estimate):
        ticket = uuid.uuid4().hex
        with self.transaction() as state:
            active = sum(v["bytes"] for v in state["reservations"].values())
            own = state["attempts"].get(self.attempt, 0) + sum(
                v["bytes"] for v in state["reservations"].values() if v["attempt"] == self.attempt
            )
            if (
                own + estimate > self.settings["max_run_bytes"]
                or state["written_bytes"] + active + estimate > self.settings["max_phase_bytes"]
            ):
                raise BudgetStop("Stage 1 cumulative checkpoint/cache write allowance exhausted")
            state["reservations"][ticket] = {"attempt": self.attempt, "bytes": estimate, "path": str(path)}
        # An interrupted/failed write retains its conservative reservation. No guessed zero.
        yield
        actual = disk_path(path).stat().st_size
        with self.transaction() as state:
            state["reservations"].pop(ticket)
            state["written_bytes"] += actual
            state["attempts"][self.attempt] = state["attempts"].get(self.attempt, 0) + actual
            exceeded = (
                state["attempts"][self.attempt] > self.settings["max_run_bytes"]
                or state["written_bytes"] > self.settings["max_phase_bytes"]
            )
        if exceeded:
            raise BudgetStop("actual Stage 1 write exceeded its finite allowance; charge retained")


@contextmanager
def account(meter, path, estimate):
    writer = getattr(meter, "write_meter", None)
    if writer:
        with writer.account(path, estimate):
            yield
    else:
        yield
