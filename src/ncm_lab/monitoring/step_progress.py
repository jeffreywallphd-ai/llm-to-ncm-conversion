"""Step-triggered console reporting; clocks measure work and never trigger output."""

import math
import time


class StepEstimate:
    """Average observed wall cost since attachment; refresh only at ten-step boundaries."""

    def __init__(self, total, initial_step=0, *, clock=time.perf_counter):
        if type(total) is not int or total <= 0 or type(initial_step) is not int or not 0 <= initial_step <= total:
            raise ValueError("valid completed/target step counts required")
        self.total, self.initial_step, self.step = total, initial_step, initial_step
        self.clock, self.started = clock, clock()
        self.average = self.remaining = self.estimate_step = None
        self.sampled_steps = 0

    def snapshot(self):
        return dict(
            target_steps=self.total,
            average_seconds_per_step=self.average,
            remaining_seconds=0.0 if self.step == self.total else self.remaining,
            estimate_step=self.estimate_step,
            sampled_steps=self.sampled_steps,
            timing_scope="wall cost of observed completed updates in this attempt; initialization excluded",
        )

    def advance(self, step):
        if type(step) is not int or not self.step < step <= self.total:
            raise ValueError("completed optimizer steps must increase within the declared target")
        self.step = step
        if step % 10 == 0:
            elapsed = self.clock() - self.started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("finite nonnegative step observation required")
            self.sampled_steps = step - self.initial_step
            self.average = elapsed / self.sampled_steps
            self.remaining = self.average * (self.total - step)
            self.estimate_step = step
        return self.snapshot()


def duration_minutes(seconds):
    """Round a positive remaining duration up to a whole minute."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("finite nonnegative remaining duration required")
    minutes = math.ceil(seconds / 60)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f"{days}d {hours}h {minutes}m"


def format_step(stage, metric, total, *, attempt=None):
    step = metric["step"]
    fields = [f"{stage} step {step}/{total} ({100 * step / total:.2f}%)"]
    if attempt:
        fields.append(f"attempt={attempt}")
    fields.extend(f"{name}={value:.6g}" for name, value in metric.get("losses", {}).items())
    if "dev_selection_loss" in metric:
        fields.append(f"dev={metric['dev_selection_loss']:.6g}")
    if "gradient_norm" in metric:
        fields.append(f"grad={metric['gradient_norm']:.5g}")
    rates = metric.get("scheduler", {}).get("current_rates")
    if rates is not None:
        fields.append("lr=" + ",".join(f"{rate:.6g}" for rate in rates))
    estimate = metric["progress"]
    average, remaining = estimate["average_seconds_per_step"], estimate["remaining_seconds"]
    fields.append(f"avg_step={average:.2f}s" if average is not None else "avg_step=pending")
    fields.append("eta=" + (duration_minutes(remaining) if remaining is not None else "pending"))
    if estimate["estimate_step"] is not None:
        fields.append(f"estimate_step={estimate['estimate_step']}")
    return " ".join(fields)
