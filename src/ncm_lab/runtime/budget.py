from dataclasses import dataclass, asdict
import math
from pathlib import Path
import time


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    phase_budget_id: str
    max_steps: int
    max_wall_seconds: float
    max_training_source_token_passes: int
    max_supervised_future_positions: int
    max_storage_bytes: int
    max_download_bytes: int
    max_phase_steps: int
    max_phase_source_token_passes: int
    max_phase_wall_seconds: float
    max_phase_storage_bytes: int
    max_phase_download_bytes: int

    def __post_init__(self):
        for key, val in asdict(self).items():
            if key == "phase_budget_id":
                if not val or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in val):
                    raise ValueError("phase_budget_id must be a safe nonempty identifier")
            elif not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val < 0:
                raise ValueError(f"{key} requires a finite nonnegative limit")
        for key in ("max_steps", "max_wall_seconds", "max_storage_bytes", "max_phase_steps"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.max_phase_steps < self.max_steps or self.max_phase_source_token_passes < self.max_training_source_token_passes:
            raise ValueError("phase budget must cover the per-run limits")
        for phase, per_run in (("max_phase_wall_seconds", "max_wall_seconds"),
                               ("max_phase_storage_bytes", "max_storage_bytes"),
                               ("max_phase_download_bytes", "max_download_bytes")):
            if getattr(self,phase) < getattr(self,per_run):
                raise ValueError(f"{phase} must cover the per-run limit")


class Meter:
    def __init__(self, budget: Budget, artifact_dir: Path):
        self.budget, self.artifact_dir = budget, artifact_dir
        self.accounting_dir = artifact_dir
        self.started = time.monotonic()
        self.steps = self.source_tokens = self.supervised_positions = 0
        self.inherited_steps = self.inherited_source_tokens = self.inherited_supervised_positions = 0

    @property
    def new_steps(self):
        return self.steps-self.inherited_steps

    @property
    def new_source_tokens(self):
        return self.source_tokens-self.inherited_source_tokens

    @property
    def new_supervised_positions(self):
        return self.supervised_positions-self.inherited_supervised_positions

    def check(self, source_tokens=0, supervised_positions=0, step=True):
        b = self.budget
        size = sum(p.stat().st_size for p in self.accounting_dir.rglob("*") if p.is_file())
        if (step and self.steps >= b.max_steps or time.monotonic() - self.started >= b.max_wall_seconds
                or self.source_tokens + source_tokens > b.max_training_source_token_passes
                or self.supervised_positions + supervised_positions > b.max_supervised_future_positions
                or size > b.max_storage_bytes):
            raise BudgetExceeded("finite step/time/token/storage budget reached")

    def record(self, source_tokens, supervised_positions):
        self.steps += 1
        self.source_tokens += source_tokens
        self.supervised_positions += supervised_positions
