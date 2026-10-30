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
    wave = commands.add_parser('wave').add_subparsers(dest='wave_action', required=True)
    for action in ('plan', 'run'):
        operation = wave.add_parser(action)
        operation.add_argument('--config', type=Path, required=True)
        if action == 'run':
            operation.add_argument('--resume', type=Path)
    preflight = commands.add_parser('preflight')
    preflight.add_argument('--config', type=Path, required=True)
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
    generation_input = generation.add_mutually_exclusive_group(required=True)
    generation_input.add_argument("--prompt-file", type=Path)
    generation_input.add_argument("--messages-file", type=Path)
    generation.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    generation.add_argument("--head-chunk-positions", type=int, default=128)
    generation.add_argument("--max-head-temp-bytes", type=int, default=268435456)
    monitor = commands.add_parser(
        "monitor", help="read-only canonical JSONL / TensorBoard monitoring"
    ).add_subparsers(dest="monitor_action", required=True)
    monitor.add_parser("doctor")
    serve = monitor.add_parser("serve")
    serve.add_argument("--runs-dir", type=Path, default=Path("runs"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=6006)
    serve.add_argument("--max-seconds", type=float, default=86400)
    serve.add_argument("--max-bytes", type=int, default=2147483648)
    serve.add_argument("--headroom-bytes", type=int, default=2147483648)
    serve.add_argument("--run-id")
    serve.add_argument("--attempt-id")
    replay = monitor.add_parser("replay")
    replay.add_argument("--attempt-dir", type=Path, required=True)
    replay.add_argument("--runs-dir", type=Path, default=Path("runs"))
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--follow", action="store_true")
    replay.add_argument("--max-seconds", type=float, default=3600)
    smoke = monitor.add_parser("smoke")
    smoke.add_argument("--config", type=Path, default=Path("configs/monitoring/smoke.yaml"))
    return p


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command == 'wave':
            from ncm_lab.runtime.wave import plan, run_wave
            from ncm_lab.config.waves import wave_config

            wave = wave_config(args.config)
            result = plan(wave) if args.wave_action == 'plan' else run_wave(wave, args.resume)
        elif args.command == 'preflight':
            from ncm_lab.runtime.doctor import preflight
            from ncm_lab.config.stages import config
            from ncm_lab.runtime.io import root

            value = config(args.config)
            result = preflight(value, root(), training=value['stage'] in {'generator','profile'})
            result['status'] = 'blocked' if result['blockers'] else 'completed'
        elif args.command == "monitor":
            if args.monitor_action == "doctor":
                from ncm_lab.monitoring.server import doctor

                result = doctor()
            elif args.monitor_action == "serve":
                from ncm_lab.monitoring.server import serve

                result = serve(
                    runs_dir=args.runs_dir,
                    host=args.host,
                    port=args.port,
                    max_seconds=args.max_seconds,
                    max_bytes=args.max_bytes,
                    headroom_bytes=args.headroom_bytes,
                    run_id=args.run_id,
                    attempt_id=args.attempt_id,
                )
            elif args.monitor_action == "replay":
                import subprocess
                import os
                import math

                if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 604800:
                    raise ValueError("replay needs a finite positive lifetime of at most seven days")
                if args.output_dir.exists():
                    raise ValueError("replay refuses to overwrite an existing projection directory")

                command = [
                    sys.executable,
                    "-m",
                    "ncm_lab.monitoring.projector",
                    "--runs-dir",
                    str(args.runs_dir),
                    "--attempt-dir",
                    str(args.attempt_dir),
                    "--output-dir",
                    str(args.output_dir),
                    "--max-seconds",
                    str(args.max_seconds),
                ]
                if args.follow:
                    command += ["--follow"]
                child = subprocess.run(
                    command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=args.max_seconds + 30
                )
                status_path = args.output_dir / "status.json"
                if not status_path.is_file():
                    raise ValueError(
                        "projection failed or output already exists; inspect status.json / use a new directory"
                    )
                result = {
                    **json.loads(status_path.read_text()),
                    "output": str(args.output_dir),
                    "canonical_modified": False,
                }
                if child.returncode and result.get("status") not in {"incomplete", "invalid_source"}:
                    raise ValueError("projector failed; inspect preserved projection status")
            else:
                from ncm_lab.config.stages import config
                from ncm_lab.runtime.runner import execute

                value = config(args.config)
                if (
                    not value["model"].get("fixture")
                    or value["device"] != "cpu"
                    or value["stage"] != "validation"
                ):
                    raise ValueError("monitor smoke requires a bounded CPU fixture validation config")
                result = execute(value)
        elif args.command == "doctor":
            from ncm_lab.runtime.doctor import doctor

            result = doctor(args.profile)
        elif args.command == "prepare":
            from ncm_lab.runtime.prepare import prepare
            from ncm_lab.config.stages import config

            result = prepare(config(args.config), args.plan_only)
        elif args.command == "generate":
            from ncm_lab.runtime.generation import run_generation

            result = run_generation(
                args.checkpoint,
                args.prompt_file,
                args.device,
                {
                    "head_chunk_positions": args.head_chunk_positions,
                    "max_head_temp_bytes": args.max_head_temp_bytes,
                },
                messages_file=args.messages_file,
            )
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
            if isinstance(result, dict)
            and result.get("status")
            in {"failed", "blocked", "budget_stopped", "incomplete", "invalid_source", "cancelled"}
            else 0
        )
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print("ncm: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
