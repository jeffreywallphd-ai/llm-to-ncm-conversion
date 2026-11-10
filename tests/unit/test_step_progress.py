import math
import pytest
from ncm_lab.monitoring.step_progress import StepEstimate, duration_minutes, format_step


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def test_estimate_refreshes_at_ten_steps_and_reuses_the_entire_previous_estimate():
    clock = Clock()
    estimate = StepEstimate(25, clock=clock)
    for step in range(1, 10):
        clock.value += 2
        assert estimate.advance(step)["remaining_seconds"] is None
    clock.value += 2
    first = estimate.advance(10)
    assert first["average_seconds_per_step"] == 2
    assert first["remaining_seconds"] == 30
    for step in range(11, 20):
        clock.value += 4
        assert estimate.advance(step) == first
    clock.value += 4
    second = estimate.advance(20)
    assert second["average_seconds_per_step"] == 3
    assert second["remaining_seconds"] == 15
    assert second["estimate_step"] == 20
    for step in range(21, 25):
        clock.value += 500
        assert estimate.advance(step) == second
    final = estimate.advance(25)
    assert final["remaining_seconds"] == 0
    assert final["estimate_step"] == 20


def test_restored_step_does_not_invent_elapsed_work_before_the_attempt():
    clock = Clock()
    clock.value = 10000
    estimate = StepEstimate(100, 50, clock=clock)
    for step in range(51, 61):
        clock.value += 30
        result = estimate.advance(step)
    assert result["sampled_steps"] == 10
    assert result["average_seconds_per_step"] == 30
    assert result["remaining_seconds"] == 1200
    assert result["estimate_step"] == 60
    with pytest.raises(ValueError, match="increase"):
        estimate.advance(60)


@pytest.mark.parametrize("elapsed", [-1, math.nan, math.inf])
def test_invalid_elapsed_observations_fail_without_nonfinite_metrics(elapsed):
    clock = Clock()
    estimate = StepEstimate(100, clock=clock)
    clock.value = elapsed
    with pytest.raises(ValueError, match="observation"):
        estimate.advance(10)


def test_duration_and_output_express_days_hours_minutes_and_cached_estimate_step():
    assert duration_minutes(0) == "0d 0h 0m"
    assert duration_minutes(60) == "0d 0h 1m"
    assert duration_minutes(86400 + 2 * 3600 + 3 * 60 + 1) == "1d 2h 4m"
    clock = Clock()
    estimate = StepEstimate(30, clock=clock)
    clock.value = 1000
    progress = estimate.advance(10)
    row = dict(
        step=11,
        losses=dict(text_ce=1.5, latent_loss=0.2),
        gradient_norm=0.5,
        scheduler=dict(current_rates=[1e-5, 1e-4]),
        progress=progress,
    )
    text = format_step("generator", row, 30)
    assert "step 11/30" in text and "text_ce=1.5" in text
    assert "avg_step=100.00s eta=0d 0h 34m estimate_step=10" in text
