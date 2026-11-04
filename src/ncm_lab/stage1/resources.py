"""Finite allowances derived after source qualification; elapsed clocks are observations."""

from copy import deepcopy
import math
from pathlib import Path
from ncm_lab.runtime.io import read, digest, write_json, reference
from ncm_lab.runtime.doctor import parameter_inventory
from ncm_lab.runtime.budget import Budget
from ncm_lab.data.artifacts import Dataset


def derive(settings, preparation, experiment, *, persist=True):
    data = Dataset(preparation["artifact"], "native_data_v1")
    qualification = read(preparation["qualification"]["manifest"])
    ntrain = qualification["selected"]["train"]["examples"]
    ndev = qualification["selected"]["dev"]["examples"]
    ntest = qualification["selected"]["test"]["examples"]
    canvas = settings["canvas"]["positions"]
    inventory = parameter_inventory(
        settings["stages"]["generator"]["model"], settings["stages"]["generator"]["architecture"]
    )
    hidden = 16 if settings["stages"]["generator"]["model"].get("fixture") else 4096
    if settings["stages"]["generator"]["model"].get("fixture"):
        hidden = settings["stages"]["generator"]["model"]["config"]["hidden_size"]
    width = settings["stages"]["generator"]["architecture"]["latent_width"]
    resources = settings["resources"]
    max_pair = max(
        len(data.metadata(i)["example"]["prompt_ids"]) + len(data.metadata(i)["example"]["answer_ids"])
        for i in data.indices("train")
    )
    stage_plans = {}
    for name in ("teacher", "codec", "targets", "generator", "evaluate"):
        value = deepcopy(settings["stages"][name])
        options = value["training"]
        target = options["steps"]
        ceiling = resources["max_generator_steps"] if name == "generator" else target
        effective = options["batch_size"] * options["accumulation"]
        presentations = ceiling * effective if name in {"codec", "generator"} else 0
        validations = math.ceil(ceiling / options["validation_interval"]) + 2 if presentations else 0
        assignments = (
            max(31, min(ndev, options["max_validation_examples"]))
            if name == "codec"
            else min(ndev, options["max_validation_examples"])
        )
        samples = (validations * value["monitoring"]["sample_prompts"]) if name == "generator" else 0
        auxiliary_passes = 2 * ntrain if name == "codec" else 0  # normalization and selected-state scales
        examples = presentations + validations * assignments + samples + auxiliary_passes + ndev
        if name in {"teacher", "targets"}:
            examples = ntrain + ndev
        if name == "evaluate":
            examples = 2 * (ndev + ntest)  # CE and generation, independently one traversal each.
        source_bound = max(1, math.ceil(examples * (512 + canvas) * resources["work_margin"]))
        answer_bound = max(1, math.ceil(examples * canvas * resources["work_margin"]))
        checkpoints = (
            math.ceil(ceiling / options["validation_interval"])
            + math.ceil(ceiling / options["checkpoint_interval"])
            + 2
            if presentations
            else 0
        )
        # Worst-case improvements at every validation; shared boundaries save once in execution.
        parameter_bytes = (
            inventory["generator"] * 10
            if name == "generator"
            else inventory["codec"] * 12
            if name == "codec"
            else 0
        )
        checkpoint_bytes = int(parameter_bytes * 1.05) + 1048576 if presentations else 0
        payload = (ntrain + ndev) * canvas * (hidden * 2 if name == "teacher" else width * 4)
        write_allowance = max(10_000_000, (checkpoints + 2) * checkpoint_bytes + payload * 2 + 10_000_000)
        budget = Budget(
            "s1-" + settings["contract_sha256"][:40] + "-" + name,
            enforce_wall_time_limits=False,
            max_steps=max(1, ceiling if presentations else examples),
            max_phase_steps=max(1, (ceiling if presentations else examples) * resources["retry_allowance"]),
            max_source_tokens=source_bound,
            max_phase_source_tokens=source_bound * resources["retry_allowance"],
            max_answer_positions=answer_bound,
            max_phase_answer_positions=answer_bound * resources["retry_allowance"],
            max_canvas_positions=answer_bound,
            max_phase_canvas_positions=answer_bound * resources["retry_allowance"],
            max_storage_bytes=resources["max_retained_bytes"],
            max_phase_storage_bytes=resources["max_phase_retained_bytes"],
        )
        from dataclasses import asdict

        value["budget"] = asdict(budget)
        value["stage1_io"] = {
            "ledger": str(Path(experiment) / "ledgers" / f"{name}-writes.json"),
            "max_run_bytes": write_allowance,
            "max_phase_bytes": write_allowance * resources["retry_allowance"],
        }
        options["epochs"] = max(1, math.ceil(ceiling * effective / ntrain) + 1)
        options["max_processed_tokens"] = max(1, ceiling * effective * max_pair + 1)
        stage_plans[name] = {
            "config": value,
            "target_completed_updates": target if presentations else None,
            "nominal_training_presentations": target * effective if presentations else 0,
            "finite_target_ceiling": ceiling if presentations else None,
            "allowance_example_forwards": examples,
            "routine_validation_assignments": assignments,
            "scheduled_validation_allowance": validations,
            "sample_allowance": samples,
            "normalization_and_scales_example_passes": auxiliary_passes,
            "worst_case_recovery_writes": checkpoints,
            "one_recovery_tensor_bytes": parameter_bytes,
            "cumulative_write_allowance_bytes": write_allowance,
        }
    model_cache = inventory["original"] * 2
    features = (ntrain + ndev) * canvas * hidden * 2
    targets = (ntrain + ndev) * canvas * width * 4
    peak = (
        4 * inventory["generator"] * 10
        + inventory["generator"] * 2
        + features
        + targets
        + model_cache
        + resources["reserve_free_bytes"]
    )
    report = {
        "schema": "stage1_resource_derivation_v1",
        "contract_sha256": settings["contract_sha256"],
        "qualified_data_sha256": digest(preparation["artifact"]),
        "qualified_counts": {"train": ntrain, "dev": ndev, "test": ntest},
        "inventory": inventory,
        "stages": stage_plans,
        "margin": resources["work_margin"],
        "peak_disk_estimate_including_shared_model_cache_and_headroom_bytes": peak,
        "peak_disk_scope": "one active attempt plus shared artifacts/headroom; preserved prior attempts consume current free space and the separate aggregate retained-storage allowance; every new stage admission checks current free space",
        "shared_model_cache_estimate_bytes": model_cache,
        "teacher_cache_upper_bytes": features,
        "fp32_fine_code_cache_upper_bytes": targets,
        "reserve_free_bytes": resources["reserve_free_bytes"],
        "generator_baseline_weights_gradients_fp32_moments_bytes": inventory["generator"] * 12,
        "codec_fp32_head_and_parameters_gradients_moments_bytes": inventory["head"] * 4
        + inventory["codec"] * 16,
        "activation_and_kernel_peak": "unmeasured; baseline state admission does not establish training feasibility",
        "cumulative_writes_are_not_peak_retained_disk": True,
        "clock_policy": "disabled for all training/preparation/evaluation/controller stages; elapsed measured",
    }
    path = Path(experiment) / "qualification/resources.json"
    if persist:
        write_json(path, report, exclusive=True)
    return report, reference(path) if persist else None
