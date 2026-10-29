import copy
import json
import weakref
from pathlib import Path
import torch
import pytest
from ncm_lab.runtime.runner import execute
from ncm_lab.training.checkpoints import CheckpointReader
from ncm_lab.training.engine import architecture_from_metadata, make_optimizer
from ncm_lab.teacher.stages import load_codec
from ncm_lab.training.optimizer import TiledAdamW


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


def assert_same_recovery_state(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_recovery_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_recovery_state(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_in_process_recovery_releases_old_tensors_and_matches_uninterrupted(
    native_artifacts, monkeypatch, tmp_path, dtype
):
    import ncm_lab.training.engine as engine

    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg.update(stage="profile", dtype=dtype, output_ref=None)
    cfg["training"].update(steps=3, batch_size=1, warmup_processed_tokens=0, recovery_probe=False)
    uninterrupted = execute(cfg)
    assert uninterrupted["status"] == "completed", uninterrupted

    # Shared fixture preparation and the reference run finish before observation.
    # Keep only weak references and numeric metadata about the old allocation.
    references = {key: [] for key in ("parameters", "gradients", "moments")}
    captured = {key: {"tensors": 0, "elements": 0, "bytes": 0} for key in references}
    observations = {
        "dtype": dtype,
        "captured": captured,
        "before_allocation": None,
        "replacement_forwards": [],
        "exact_continuation": False,
    }
    handles = []
    original_step = TiledAdamW.step  # Unbound functions do not own an optimizer/model.
    original_constructor = engine.architecture_from_metadata

    def capture(key, tensor):
        references[key].append(weakref.ref(tensor))
        captured[key]["tensors"] += 1
        captured[key]["elements"] += tensor.numel()
        captured[key]["bytes"] += tensor.numel() * tensor.element_size()

    def observe_step(optimizer, gradient_report=None):
        result = original_step(optimizer, gradient_report)
        if not references["parameters"] and max(s["step"] for s in optimizer.state.values()) == 2:
            for parameter in optimizer.parameters.values():
                assert parameter.requires_grad and parameter.grad is not None
                capture("parameters", parameter)
                capture("gradients", parameter.grad)
            for state in optimizer.state.values():
                for field in ("exp_avg", "exp_avg_sq"):
                    assert state[field].dtype == torch.float32
                    capture("moments", state[field])
            assert captured["parameters"]["tensors"] == len(optimizer.parameters)
        return result

    def alive():
        return {key: sum(reference() is not None for reference in refs) for key, refs in references.items()}

    def observe_forward(module, inputs):
        snapshot = alive()
        observations["replacement_forwards"].append(snapshot)
        assert not any(snapshot.values()), ("old tensors survived into replacement forward", snapshot)

    def observe_allocation(metadata, device):
        assert all(info["tensors"] > 0 and info["bytes"] > 0 for info in captured.values())
        snapshot = alive()
        observations["before_allocation"] = snapshot
        # No clearing, GC, tensor copies, or other repair in this observer.
        assert not any(snapshot.values()), ("old tensors alive before replacement allocation", snapshot)
        replacement = original_constructor(metadata, device)
        handles.append(replacement.register_forward_pre_hook(observe_forward))
        return replacement

    cfg["training"]["recovery_probe"] = True
    try:
        with monkeypatch.context() as instrumentation:
            instrumentation.setattr(TiledAdamW, "step", observe_step)
            instrumentation.setattr(engine, "architecture_from_metadata", observe_allocation)
            probe = execute(cfg)
        observations["probe"] = {key: probe[key] for key in ("run_id", "path", "status")}
        observations["uninterrupted"] = {key: uninterrupted[key] for key in ("run_id", "path", "status")}
        assert probe["status"] == "completed" and probe["optimizer_steps"] == 3, (probe, observations)
        assert observations["before_allocation"] is not None and observations["replacement_forwards"]
        assert captured["parameters"]["tensors"] == captured["gradients"]["tensors"]
        assert captured["moments"]["tensors"] == 2 * captured["parameters"]["tensors"]
        assert captured["parameters"]["elements"] == probe["parameter_count"]
        events = [
            json.loads(line) for line in (Path(probe["path"]) / "events.jsonl").read_text().splitlines()
        ]
        assert any(e["status"] == "recovery_probe_restored" and e["step"] == 2 for e in events)

        # Compare terminal recovery state, never a potentially earlier selected export.
        left, left_optimizer, left_state, left_rng = terminal(uninterrupted["checkpoint"])
        right, right_optimizer, right_state, right_rng = terminal(probe["checkpoint"])
        assert_same_recovery_state(left.state_dict(), right.state_dict())
        assert_same_recovery_state(left_optimizer.state_dict(), right_optimizer.state_dict())
        assert_same_recovery_state(left_rng, right_rng)
        for key in (
            "optimizer_steps",
            "processed_source_tokens",
            "answer_position_exposures",
            "allocated_canvas_positions",
            "backbone_answer_positions",
            "phase",
            "last_executed_phase",
            "cursor",
            "best_dev_loss",
            "selected_this_step",
        ):
            assert_same_recovery_state(left_state[key], right_state[key])

        def last_metric(result):
            return json.loads((Path(result["path"]) / "metrics.jsonl").read_text().splitlines()[-1])

        expected, actual = last_metric(uninterrupted), last_metric(probe)
        assert expected["step"] == actual["step"] == 3 and expected["loss"] == actual["loss"]
        observations.update(
            exact_continuation=True,
            third_step_loss=actual["loss"],
            optimizer_rounding_draws=right_optimizer.rounding_draws,
        )
    finally:
        for handle in handles:
            handle.remove()
        # Test evidence is separate from immutable production run artifacts.
        (tmp_path / "lifetime_observations.json").write_text(json.dumps(observations, indent=2) + "\n")
