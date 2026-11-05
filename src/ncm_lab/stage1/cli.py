"""Public Stage 1 script interface. Planning never imports model libraries."""

import argparse
import json
from pathlib import Path
import sys
from ncm_lab.config.canvas import add_canvas_argument, resolve_canvas
from ncm_lab.stage1.config import resolve, STAGES
from ncm_lab.stage1.policy import SCHEMA
from ncm_lab.stage1.workflow import plan, load, run


def parser(fixed_stage=None):
    p = argparse.ArgumentParser(description="Manifest-bound foreground Stage 1 single-pass experiment")
    source = p.add_mutually_exclusive_group(required=fixed_stage is not None)
    source.add_argument("--experiment", type=Path)
    source.add_argument("--resume", type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    p.add_argument("--experiment-dir", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--data-config", type=Path)
    add_canvas_argument(p)
    p.add_argument("--seed", type=int)
    p.add_argument("--codec-steps", type=int)
    p.add_argument("--generator-steps", type=int)
    p.add_argument("--stage", choices=[*STAGES, "all"])
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--allow-codec-review", action="store_true")
    p.add_argument("--reason")
    p.add_argument("--extend-generator-steps", type=int)
    p.add_argument("--split", choices=["test"], default="test")
    p.add_argument("--ar-reference", action="store_true")
    p.add_argument("--ar-max-new-tokens", type=int)
    return p


def main(argv=None, *, fixed_stage=None):
    args = parser(fixed_stage).parse_args(argv)
    try:
        if args.ar_max_new_tokens is not None and not args.ar_reference:
            raise ValueError("--ar-max-new-tokens requires the explicit --ar-reference action")
        if args.ar_reference and (
            args.resume_checkpoint or args.allow_codec_review or args.extend_generator_steps is not None
        ):
            raise ValueError(
                "optional AR inference cannot be combined with training recovery/review/extension"
            )
        experiment = args.experiment or args.resume
        stage = fixed_stage or args.stage or "all"
        if fixed_stage and args.stage and args.stage != fixed_stage:
            raise ValueError("this wrapper executes only " + fixed_stage)
        if experiment:
            if (
                args.experiment_dir
                or args.config
                or args.data_config
                or args.codec_steps is not None
                or args.generator_steps is not None
            ):
                raise ValueError(
                    "configuration/data/step overrides apply only when planning a NEW experiment; existing manifests retain their settings"
                )
            path, state, settings = load(experiment)
            if settings["schema"] != SCHEMA:
                raise ValueError("use scripts/run_stage2.py for a Stage 2 manifest")
            if (
                args.canvas_size is not None
                and resolve_canvas(args.canvas_size).positions != settings["canvas"]["positions"]
            ):
                raise ValueError("Stage 1 resume canvas conflicts with stored geometry")
            if args.seed is not None and args.seed != settings["seed"]:
                raise ValueError("Stage 1 resume seed conflicts with stored initialization")
            if args.execute:
                if args.ar_reference:
                    if fixed_stage != "evaluate":
                        raise ValueError("optional AR reference belongs to evaluate_stage1.py")
                    from ncm_lab.stage1.workflow import run_ar_reference

                    result = run_ar_reference(path, args.ar_max_new_tokens)
                else:
                    result = run(
                        path,
                        stage=stage,
                        resume_checkpoint=args.resume_checkpoint,
                        allow_codec_review=args.allow_codec_review,
                        reason=args.reason,
                        extend_generator_steps=args.extend_generator_steps,
                    )
            else:
                result = {
                    "status": "planned",
                    "experiment": str(path),
                    "stage_statuses": {n: e["status"] for n, e in state["stages"].items()},
                    "canvas": settings["canvas"],
                    "execute_required": True,
                    "research_work_started": False,
                }
        else:
            if args.ar_reference or args.ar_max_new_tokens:
                raise ValueError("AR reference requires an existing frozen experiment")
            if args.resume_checkpoint or args.allow_codec_review or args.extend_generator_steps:
                raise ValueError("recovery/review/extension requires an existing explicit experiment")
            if args.execute and not args.experiment_dir:
                raise ValueError(
                    "new execution requires --experiment-dir, or select an existing --experiment"
                )
            settings = resolve(
                args.config,
                canvas_size=args.canvas_size,
                seed=args.seed,
                data_config=args.data_config,
                codec_steps=args.codec_steps,
                generator_steps=args.generator_steps,
            )
            if settings["schema"] != SCHEMA:
                raise ValueError("use scripts/run_stage2.py for a Stage 2 recipe")
            proposal = plan(settings, args.experiment_dir)
            if args.execute:
                result = run(proposal["experiment"], stage=stage)
            else:
                result = {
                    "status": "planned",
                    "experiment": proposal.get("experiment"),
                    "canvas": settings["canvas"],
                    "stages": {
                        n: {
                            "dtype": c["dtype"],
                            "updates": c["training"]["steps"] if n in {"codec", "generator"} else None,
                            "batch_size": c["training"]["batch_size"]
                            if n in {"codec", "generator"}
                            else None,
                            "accumulation": c["training"]["accumulation"]
                            if n in {"codec", "generator"}
                            else None,
                        }
                        for n, c in settings["stages"].items()
                    },
                    "qualification": proposal["plan"]["qualification"],
                    "source": proposal["plan"]["source"],
                    "training_clocks": "disabled",
                    "research_work_started": False,
                    "acquisition_is_separate": proposal["plan"]["missing_asset_instruction"],
                }
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0 if result["status"] in {"planned", "ready", "completed", "already_completed"} else 2
    except (ValueError, TypeError, OSError, KeyError, ImportError, RuntimeError) as exc:
        print("Stage 1: " + str(exc), file=sys.stderr, flush=True)
        return 2
