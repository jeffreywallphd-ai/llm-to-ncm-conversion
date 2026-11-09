"""Thin block8 command; execution always requires an explicit flag."""

import argparse
import json
from pathlib import Path


def add_arguments(parser):
    parser.add_argument("--config", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="block8 inference export; opens generation mode")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt-file", type=Path)
    prompts.add_argument("--messages-file", type=Path)


def dispatch(args):
    from ncm_lab.block8.config import resolve
    from ncm_lab.block8.workflow import run_workflow

    if args.resume and args.config:
        raise ValueError("resume uses its immutable configuration")
    if args.evaluation_manifest and not args.data_manifest:
        raise ValueError("explicit evaluation artifact requires explicit training artifact")
    prompt = messages = None
    for name in ("prompt_file", "messages_file"):
        file = getattr(args, name)
        if file:
            if file.stat().st_size > 524288:
                raise ValueError("block8 prompt file exceeds the finite input-byte cap")
            text = file.read_text(encoding="utf-8")
            if name == "prompt_file":
                prompt = text
            else:
                messages = json.loads(text)
    if (prompt is not None or messages is not None) and not (args.checkpoint or args.resume):
        raise ValueError("prompt inputs require an explicit block8 export")
    return run_workflow(
        None if args.resume else resolve(args.config),
        resume=args.resume,
        experiment_dir=args.experiment_dir,
        execute=args.execute,
        prepare_only=args.prepare_only,
        data_manifest=args.data_manifest,
        evaluation_manifest=args.evaluation_manifest,
        checkpoint=args.checkpoint,
        prompt=prompt,
        messages=messages,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Native hierarchical eight-token block-autoregressive workflow"
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] in {"planned", "completed"} else 2
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(json.dumps(dict(status="failed", error=str(exc))))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
