"""Stage execution with immutable handoffs, no historical-run guessing or hidden teachers."""

from pathlib import Path
import traceback
import torch
from ncm_lab.config.stages import config, bind_inputs
from ncm_lab.runtime.provenance import Run
from ncm_lab.runtime.io import write_json
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.doctor import preflight


def execute(value, resume=None, stop_after=None):
    torch.set_num_threads(value["threads"])
    bound, missing = bind_inputs(value)
    run = Run(value, bound)
    if missing:
        return run.close("blocked", {"blockers": missing, "model_loaded": False})
    try:
        run.start()
        if value["stage"] not in {"data", "validation"}:
            report = preflight(value, run.path, training=value["stage"] in {"generator", "profile"})
            write_json(run.path / "preflight.json", report, exclusive=True)
            if report["blockers"]:
                return run.close("blocked", {"blockers": report["blockers"], "model_loaded": False})
        if value["stage"] == "data":
            from ncm_lab.data.prepare import prepare_data

            result = prepare_data(run, value)
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
            result = fit(run, value, bound, resume=resume, stop_after=stop_after)
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
                "checkpoint": str(run.path / "checkpoints/current.json"),
                "artifact_published": False,
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
