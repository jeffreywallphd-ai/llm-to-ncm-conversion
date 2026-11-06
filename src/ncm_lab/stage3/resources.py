"""Finite work, retained disk, and cumulative writes are separate allowances."""

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from ncm_lab.stage1.resources import derive as legacy_derive
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.io import write_json, reference


def derive(settings, preparation, experiment, *, persist=True):
    report, _ = legacy_derive(settings, preparation, experiment, persist=False)
    inventory = report["inventory"]
    canvas = settings["canvas"]["positions"]
    q = report["qualified_counts"]
    for name, stage in report["stages"].items():
        value = stage["config"]
        t = value["training"]
        if name in {"codec", "generator"}:
            recoveries = math.ceil(t["steps"] / t["checkpoint_interval"]) + 2
            selections = math.ceil(t["steps"] / t["validation_interval"]) + 1
            parameters = inventory[name]
            element = 2 if name == "generator" else 4
            recover_bytes = parameters * (element + 8)
            selection_bytes = parameters * element
            writes = (
                int((recoveries * recover_bytes + selections * selection_bytes + 2 * selection_bytes) * 1.1)
                + 10000000
            )
            value["stage1_io"].update(
                max_run_bytes=writes, max_phase_bytes=writes * settings["resources"]["retry_allowance"]
            )
            stage.update(
                worst_case_recovery_writes=recoveries,
                worst_case_selected_weights_writes=selections,
                one_selected_tensor_bytes=selection_bytes,
                cumulative_write_allowance_bytes=writes,
            )
        if name == "evaluate":
            evaluation = value["stage3"]["evaluation"]
            count = q["dev"] + q["test"]
            repeats = evaluation["repeats"]
            warmup = evaluation["warmup"]
            ar = q["test"] * repeats + warmup
            forwards = count * (repeats + 1) + warmup
            b = deepcopy(value["budget"])
            for key, amount in {
                "steps": forwards + ar,
                "source_tokens": (forwards + ar * canvas) * (512 + canvas),
                "answer_positions": (forwards + ar) * canvas,
                "canvas_positions": (forwards + ar) * canvas,
            }.items():
                b["max_" + key] = max(1, math.ceil(amount * settings["resources"]["work_margin"]))
                b["max_phase_" + key] = b["max_" + key] * settings["resources"]["retry_allowance"]
            value["budget"] = asdict(Budget(**b))
            stage.update(
                original_ar_decode_forwards=ar * canvas,
                student_forwards=forwards,
                latency_warmup=warmup,
                latency_repeats=repeats,
            )
    report.update(
        schema="stage3_resource_derivation_v1",
        candidate_allowance=settings["stages"]["distill"]["stage3"]["distillation"],
        rejected_distillation_work="charged before every attempt under independent source/distill phase caps",
        selected_weights_separate_from_recovery=True,
    )
    # Two recent recoveries, in-progress recovery, best selected weights, export; conservative legacy fourth slot remains.
    path = Path(experiment) / "qualification/resources.json"
    if persist:
        write_json(path, report, exclusive=True)
    return report, reference(path) if persist else None
