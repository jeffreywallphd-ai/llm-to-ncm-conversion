"""Bounded immutable generation records from standalone native exports."""

from pathlib import Path
import torch
from ncm_lab.runtime.io import root, read, resolve_reference, digest
from ncm_lab.runtime.provenance import Run
from ncm_lab.runtime.doctor import preflight
from ncm_lab.runtime.budget import BudgetStop, Budget
from dataclasses import asdict


def run_generation(path, prompt_file, device="cpu", inference=None):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    path = resolve_reference(path)
    export = read(path)
    if export.get("schema") != "native_generator_export_v1":
        raise ValueError("native standalone generator export required")
    spec = {**export["model"]}
    if spec.get("fixture"):
        spec["config"] = export["backbone_config"]
    try:
        budget = read(root() / "configs/budgets/generation.yaml")
    except ValueError:
        budget = asdict(
            Budget(
                "native-generation-v1",
                max_wall_seconds=120,
                max_steps=1,
                max_source_tokens=512,
                max_answer_positions=1024,
                max_canvas_positions=1024,
                max_storage_bytes=10000000,
                max_phase_steps=10000,
                max_phase_storage_bytes=1000000000,
            )
        )
    value = {
        "schema": "native_stage_v1",
        "stage": "generation",
        "device": device,
        "dtype": export.get("dtype") or export["backbone_config"].get("dtype") or "float32",
        "model": spec,
        "architecture": export["architecture"],
        "budget": budget,
        "inputs": {},
        "threads": 1,
        "inference": inference or {},
    }
    run = Run(
        value,
        {
            "export": {"manifest": str(path), "sha256": digest(path)},
            "prompt": {"path": str(Path(prompt_file).resolve()), "sha256": digest(prompt_file)},
        },
    )
    try:
        run.start()
        torch.set_num_threads(1)
        report = preflight(value, run.path)
        if report["blockers"]:
            return run.close("blocked", {"blockers": report["blockers"]})
        from ncm_lab.inference.export import generate

        result = generate(path, prompt_file, device, run.meter, inference)
        return run.close("completed", result)
    except BudgetStop as exc:
        return run.close("budget_stopped", {"error": str(exc)})
    except Exception as exc:
        return run.close("failed", {"error": type(exc).__name__ + ": " + str(exc)})
