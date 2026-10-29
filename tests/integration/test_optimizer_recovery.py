import copy
import json
from pathlib import Path
import torch
import pytest
from ncm_lab.runtime.runner import execute
from ncm_lab.training.checkpoints import CheckpointReader
from ncm_lab.training.engine import architecture_from_metadata, make_optimizer
from ncm_lab.teacher.stages import load_codec


def terminal(path):
    reader = CheckpointReader(path)
    metadata = reader.materialize(reader.field("metadata"))
    model = architecture_from_metadata(metadata, "cpu")
    reader.load_model(model)
    options = metadata["training_contract"]
    opt = make_optimizer(model, options, True)
    reader.load_optimizer(opt)
    return model, opt, reader.materialize(reader.field("training")), reader.materialize(reader.field("rng"))


@pytest.mark.parametrize("warmup", [1, 11])
def test_bf16_production_generator_private_rng_resume_warmup_and_monitor_invariance(native_artifacts, warmup):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    cfg["dtype"] = "bfloat16"
    cfg["training"].update(
        steps=3,
        batch_size=1,
        accumulation=2,
        warmup_processed_tokens=warmup,
        learning_rate=1e-5,
        new_learning_rate=1e-4,
        optimizer_tile_elements=256,
    )
    cfg["monitoring"] = {"enabled": False}
    full = execute(cfg)
    assert full["status"] == "completed", full
    stopped = execute(cfg, stop_after=1)
    assert stopped["status"] == "completed", stopped
    cfg["monitoring"] = {
        "enabled": True,
        "scalar_interval": 100,
        "first_steps": 1,
        "detail_interval": 1,
        "sample_interval": 1,
        "sample_prompts": 1,
    }
    resumed = execute(cfg, stopped["checkpoint"])
    assert resumed["status"] == "completed", resumed
    a, ao, at, ar = terminal(full["checkpoint"])
    b, bo, bt, br = terminal(resumed["checkpoint"])
    assert all(torch.equal(p, b.state_dict()[n]) for n, p in a.state_dict().items())
    assert (
        torch.equal(ao.generator.get_state(), bo.generator.get_state())
        and ao.rounding_draws == bo.rounding_draws
    )
    assert (
        at["cursor"] == bt["cursor"]
        and ar["sampler"] == br["sampler"]
        and torch.equal(ar["torch"], br["torch"])
    )
    for name, state in ao.state.items():
        assert state["step"] == bo.state[name]["step"]
        assert torch.equal(state["exp_avg"], bo.state[name]["exp_avg"]) and torch.equal(
            state["exp_avg_sq"], bo.state[name]["exp_avg_sq"]
        )
    rows = [json.loads(line) for line in (Path(stopped["path"]) / "metrics.jsonl").read_text().splitlines()]
    assert (
        rows[0]["phase"] == "warmup"
        and rows[0]["next_phase"] == ("full" if warmup == 1 else "warmup")
        and stopped["last_executed_phase"] == "warmup"
    )
    rows = [json.loads(line) for line in (Path(resumed["path"]) / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == ("full" if warmup == 1 else "warmup") and rows[0]["step"] == 2
    assert rows[-1]["phase"] == "full" and rows[-1]["trainable_parameters"] == sum(
        p.numel() for p in a.parameters()
    )
    assert (
        resumed["optimizer_rounding_draws"] > 0 and resumed["last_optimizer_update"]["intended_fp32_l2"] > 0
    )


def test_bf16_codec_training_export_honors_trained_precision(native_artifacts):
    cfg = copy.deepcopy(native_artifacts["configs"]["codec"])
    cfg["output_ref"] = None
    cfg["dtype"] = "bfloat16"
    cfg["training"].update(steps=3, learning_rate=1e-5, new_learning_rate=1e-4, optimizer_tile_elements=256)
    result = execute(cfg)
    assert result["status"] == "completed", result
    codec, manifest = load_codec(result["artifact"], "cpu", torch.float32)
    assert next(codec.parameters()).dtype == torch.bfloat16 and manifest["trained_dtype"] == "bfloat16"
    assert manifest["selection"]["validation_plan"]["policy"] == "codec_stratified_coarse_indices_v1"
    assert result["last_optimizer_update"]["intended_fp32_l2"] > 0 and result["optimizer_rounding_draws"] > 0


@pytest.mark.parametrize("accumulation,warmup,stop", [(1, 1, 1), (2, 1, 1), (1, 6, 3)])
def test_executed_phase_counts_and_forced_boundary_interval(native_artifacts, accumulation, warmup, stop):
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    cfg["training"].update(
        steps=stop, batch_size=1, accumulation=accumulation, warmup_processed_tokens=warmup
    )
    cfg["monitoring"] = {"first_steps": 1, "scalar_interval": 100, "detail_interval": 1, "sample_prompts": 0}
    result = execute(cfg)
    assert result["status"] == "completed", result
    folder = Path(result["path"])
    legacy = [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()]
    boundary = next(r for r in legacy if r["phase"] == "warmup" and r["next_phase"] == "full")
    canonical = [json.loads(x) for x in (folder / "monitor_metrics.jsonl").read_text().splitlines()]
    at_boundary = next(r for r in canonical if r["split"] == "train" and r["step"] == boundary["step"])
    assert at_boundary["phase"] == "warmup"
    assert (
        at_boundary["metrics"]["resources/trainable_parameters"]["value"] == boundary["trainable_parameters"]
    )
    events = [json.loads(x) for x in (folder / "monitor_events.jsonl").read_text().splitlines()]
    transition = next(r for r in events if r.get("status") == "phase_transition")
    assert (
        transition["effective_from_step"] == boundary["step"] + 1 and transition["previous_phase"] == "warmup"
    )
    assert at_boundary["metrics"]["gradients/backbone_block_01/combined_l2"]["status"] == "frozen"
    full_records = [r for r in canonical if r["split"] == "train" and r["phase"] == "full"]
    if full_records:
        assert (
            full_records[0]["metrics"]["gradients/backbone_block_01/combined_l2"]["status"]
            == "finite_nonzero"
        )
