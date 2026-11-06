"""Analytic rates indexed by completed updates, independent of wall clocks."""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Schedule:
    total: int
    warmup: int
    minimum_ratio: float = 0.1
    schema: str = "linear_warmup_cosine_v1"

    def __post_init__(self):
        if (
            self.schema != "linear_warmup_cosine_v1"
            or type(self.total) is not int
            or type(self.warmup) is not int
            or not 0 <= self.warmup < self.total
            or not math.isfinite(self.minimum_ratio)
            or not 0 < self.minimum_ratio <= 1
        ):
            raise ValueError("invalid versioned update schedule/horizon")

    def factor(self, update):
        if type(update) is not int or not 1 <= update <= self.total:
            raise ValueError("update outside immutable scheduler horizon")
        if update <= self.warmup:
            return update / self.warmup
        # Zero warmup: first update is peak; terminal is minimum (T=1 uses peak).
        progress = (
            (update - self.warmup) / (self.total - self.warmup)
            if self.warmup
            else (update - 1) / max(1, self.total - 1)
        )
        return self.minimum_ratio + (1 - self.minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def apply(self, optimizer, peaks, update):
        factor = self.factor(update)
        if len(peaks) != len(optimizer.groups):
            raise ValueError("scheduler group identity changed")
        for group, peak in zip(optimizer.groups, peaks, strict=True):
            group["lr"] = peak * factor

    def state(self, peaks, completed):
        return {
            **asdict(self),
            "peak_rates": list(peaks),
            "completed_updates": completed,
            "current_rates": [p * self.factor(completed) for p in peaks] if completed else list(peaks),
        }
