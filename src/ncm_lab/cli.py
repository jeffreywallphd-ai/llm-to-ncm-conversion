"""Stage-oriented native CLI. Help and planning never load research models."""

import argparse
import json
from pathlib import Path
import sys


def parser():
    p = argparse.ArgumentParser(
        prog="ncm", description="Native embedded Qwen hierarchy and bounded initial training"
    )
    commands = p.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--profile", choices=["cpu", "cuda", "spark"], default="cpu")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--plan-only", action="store_true")
    for name, action in [
        ("data", "prepare"),
        ("teacher", "extract"),
        ("codec", "train"),
        ("targets", "build"),
        ("generator", "train"),
        ("profile", "train"),
    ]:
        group = commands.add_parser(name).add_subparsers(dest="action", required=True)
        stage = group.add_parser(action)
        stage.add_argument("--config", type=Path, required=True)
        if name in {"codec", "generator", "profile"}:
            stage.add_argument("--resume", type=Path)
    validation = commands.add_parser("validate")
    validation.add_argument("--config", type=Path, required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--prompt-file", type=Path, required=True)
    generation.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command == "doctor":
            from ncm_lab.runtime.doctor import doctor

            result = doctor(args.profile)
        elif args.command == "prepare":
            from ncm_lab.runtime.prepare import prepare
            from ncm_lab.config.stages import config

            result = prepare(config(args.config), args.plan_only)
        elif args.command == "generate":
            from ncm_lab.runtime.generation import run_generation

            result = run_generation(args.checkpoint, args.prompt_file, args.device)
        else:
            from ncm_lab.config.stages import config
            from ncm_lab.runtime.runner import execute

            value = config(args.config)
            expected = {
                "teacher": "teacher",
                "targets": "targets",
                "data": "data",
                "generator": "generator",
                "codec": "codec",
                "profile": "profile",
                "validate": "validation",
            }[args.command]
            if value["stage"] != expected:
                raise ValueError("CLI stage and resolved configuration disagree")
            result = execute(value, getattr(args, "resume", None))
        print(json.dumps(result, indent=2, allow_nan=False))
        return (
            2
            if isinstance(result, dict) and result.get("status") in {"failed", "blocked", "budget_stopped"}
            else 0
        )
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print("ncm: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
