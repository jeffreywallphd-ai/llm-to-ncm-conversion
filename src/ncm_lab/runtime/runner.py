"""Stage execution with immutable handoffs, no historical-run guessing or hidden teachers."""

from pathlib import Path
import traceback
import torch
from ncm_lab.config.stages import config, bind_inputs
from ncm_lab.runtime.provenance import Run
from ncm_lab.runtime.io import write_json, reference
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.doctor import preflight
from ncm_lab.data.prepare import DataCapacityError


def execute(value, resume=None, stop_after=None, *, recovery_compatibility=None):
    if value.get("experiment_context"):
        from ncm_lab.runtime.canvas_pipeline import exclusive_writer

        with exclusive_writer(str(value["output_ref"]) + ".writer.lock"):
            return _execute(value, resume, stop_after, recovery_compatibility=recovery_compatibility)
    return _execute(value, resume, stop_after, recovery_compatibility=recovery_compatibility)


def _execute(value, resume=None, stop_after=None, *, recovery_compatibility=None):
    torch.set_num_threads(value["threads"])
    bound, missing = bind_inputs(value)
    run = Run(value, bound)
    if missing:
        return run.close("blocked", {"blockers": missing, "model_loaded": False})
    try:
        run.start()
        if value.get("experiment_context") and value["stage"] != "data":
            from ncm_lab.runtime.canvas_pipeline import verify_bound_inputs

            verify_bound_inputs(value)
        if recovery_compatibility and (not resume or value["stage"] not in {"codec", "generator", "profile"}):
            raise ValueError("recovery compatibility requires an explicit training resume")
        if value.get("stage3") and (value["stage3"].get("schema") != "stage3_single_pass_v1"
                                   or value.get("stage1_context", {}).get("experiment_kind") != "stage3"):
            raise ValueError("Stage 3 policies require their own manifest context; historical gates/rates cannot be overridden")
        if value.get("stage1") and not value.get("stage1_context"):
            raise ValueError(
                "Stage 1 execution requires an explicit experiment manifest, not a standalone template"
            )
        if value["stage"] not in {"data", "validation", "stage1_prepare", "stage3_source", "stage3_qualify"}:
            report = preflight(value, run.path, training=value["stage"] in {"generator", "profile"})
            write_json(run.path / "preflight.json", report, exclusive=True)
            if report["blockers"]:
                return run.close("blocked", {"blockers": report["blockers"], "model_loaded": False})
        if value["stage"].startswith("stage3_"):
            from ncm_lab.stage3.stages import execute_stage

            result = execute_stage(run, value, bound)
        elif value["stage"] == "data":
            from ncm_lab.data.prepare import prepare_data

            result = prepare_data(run, value)
        elif value["stage"] == "stage1_prepare":
            from ncm_lab.stage1.data import prepare

            result = prepare(run, value)
        elif value["stage"] == "stage1_evaluate":
            if value["stage1_context"].get("action") == "ar_reference":
                from ncm_lab.stage1.ar_reference import evaluate
            else:
                from ncm_lab.stage1.evaluation import evaluate
            result = evaluate(run, value, bound)
        elif value["stage"] == "teacher":
            from ncm_lab.teacher.stages import extract

            result = extract(run, value, bound)
        elif value["stage"] == "targets":
            from ncm_lab.teacher.stages import targets

            result = targets(run, value, bound)
        elif value["stage"] in {"codec", "generator", "profile"}:
            from ncm_lab.training.engine import fit

            if resume:
                from ncm_lab.runtime.io import digest

                recovery_path = Path(resume)
                if recovery_path.is_dir():
                    recovery_path = recovery_path / "manifest.json"
                run.event(
                    "explicit_recovery", path=str(recovery_path.resolve()), sha256=digest(recovery_path)
                )
            result = fit(
                run,
                value,
                bound,
                resume=resume,
                stop_after=stop_after,
                recovery_compatibility=recovery_compatibility,
            )
        elif value["stage"] == "validation":
            from ncm_lab.validation.workflow import validate

            result = validate(run, value)
        else:
            raise ValueError("unsupported executable native stage")
        run.meter.check_disk()
        return run.close(result.get("status", "completed"), result)
    except BudgetStop as exc:
        return run.close(
            "budget_stopped",
            {
                "reason": str(exc),
                "checkpoint": str(run.path / "checkpoints/current.json")
                if (run.path / "checkpoints/current.json").is_file()
                else None,
                "artifact_published": False,
            },
        )
    except DataCapacityError as exc:
        qualification = run.path / (
            "qualification.json" if value.get("stage1") else "data_qualification.json"
        )
        return run.close(
            "blocked",
            {
                "error": str(exc),
                "model_loaded": False,
                "qualification": reference(qualification)
                if value.get("stage1") and qualification.exists()
                else str(qualification),
            },
        )
    except KeyboardInterrupt:
        return run.close(
            "cancelled", {"reason": "user interruption; resume last complete optimizer boundary"}
        )
    except Exception as exc:
        primary = type(exc).__name__ + ": " + str(exc)
        trace = traceback.format_exc()
        try:
            write_json(run.path / "failure.json", {"error": primary, "traceback": trace}, exclusive=True)
            return run.close("failed", {"error": primary})
        except Exception as cleanup:
            # Preserve the actual training error even when canonical storage fails.
            return {
                "run_id": run.id,
                "path": str(run.path),
                "status": "failed",
                "error": primary,
                "canonical_storage_error": type(cleanup).__name__ + ": " + str(cleanup),
                "sealed_manifest": False,
            }


def run_file(path, resume=None):
    return execute(config(path), resume)
