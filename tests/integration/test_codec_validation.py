import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from ncm_lab.training.validation_plan import codec_validation_plan
from ncm_lab.training.losses import codec_loss
from ncm_lab.codec.model import ContinuousCodec
from ncm_lab.config.architecture import Architecture, tiny_architecture
from ncm_lab.runtime.runner import execute


@pytest.mark.parametrize("count", [1, 2, 3, 12, 31, 45])
def test_fixed_plan_spans_coarse_indices_and_reports_limited_coverage(count):
    dataset = SimpleNamespace(metadata=lambda i: {"example": {"original_id": "dev" + str(i)}})
    resolutions = Architecture().resolutions
    state = random.getstate()
    plan = codec_validation_plan(dataset, list(range(count)), resolutions, 64, 91)
    assert plan == codec_validation_plan(dataset, list(range(count)), resolutions, 64, 91)
    assert random.getstate() == state
    assigned = {row["cutoff_index"] for row in plan["assignments"]}
    assert len(assigned) == min(count, 31)
    assert plan["complete_coarse_coverage"] == (count >= 31)
    if count >= 2:
        assert min(assigned) == 0 and max(assigned) == 30
    assert all(row["resolution"] == resolutions[row["cutoff_index"]] for row in plan["assignments"])


def test_repeated_fixed_model_evaluation_reconstructs_score_and_preserves_rng():
    architecture = tiny_architecture()
    codec = ContinuousCodec(16, architecture)
    head = torch.nn.Linear(16, 32, bias=False).requires_grad_(False)
    states = torch.randn(1, 16, 16)
    valid = torch.arange(16)[None] < 5
    global_state = random.getstate()
    torch_state = torch.get_rng_state()
    sampler = random.Random(12)
    sampling_state = sampler.getstate()
    values = []
    for _ in range(2):
        with torch.no_grad():
            loss, row = codec_loss(codec, head, states, valid, 1.0, random.Random(99), cutoff_index=0)
        assert float(loss) == pytest.approx(
            0.5 * row["mse_full"] + 0.5 * row["mse_cutoff"] + 0.1 * row["head_kl"], abs=1e-6
        )
        values.append((float(loss), row))
    assert values[0] == values[1]
    assert (
        global_state == random.getstate()
        and torch.equal(torch_state, torch.get_rng_state())
        and sampler.getstate() == sampling_state
    )


def test_selection_score_matches_logged_components_and_versioned_plan(native_artifacts):
    cfg = copy.deepcopy(native_artifacts["configs"]["codec"])
    cfg["output_ref"] = None
    result = execute(cfg)
    assert result["status"] == "completed", result
    folder = Path(result["path"])
    events = [json.loads(x) for x in (folder / "monitor_events.jsonl").read_text().splitlines()]
    scored = [r for r in events if r["status"] == "selection_evaluated"]
    metrics = [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()]
    for row, step in zip(scored, metrics):
        assert row["policy"] == "codec_stratified_coarse_indices_v1"
        components = row["per_example_components"]
        reconstructed = sum(
            0.5 * x["mse_full"] + 0.5 * x["mse_cutoff"] + 0.1 * x["head_kl"] for x in components
        ) / len(components)
        assert row["score"] == pytest.approx(reconstructed, abs=1e-6) and step["dev_loss"] == row["score"]
        assert [x["cutoff"] for x in components] == [0, 30]
