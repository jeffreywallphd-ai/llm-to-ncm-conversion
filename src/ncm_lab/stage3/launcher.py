"""One-command foreground Stage 3, explicit offline planning, and saved-recipe resume."""

import argparse
import json
from pathlib import Path
import shlex
import sys
from ncm_lab.runtime.io import root, read, atomic_json
from ncm_lab.runtime.canvas_pipeline import exclusive_writer
from ncm_lab.stage1.workflow import plan, run, load, checked, identifier, verify_record
from ncm_lab.stage3.config import resolve, SCHEMA


def show(value):
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)


def parser():
    p = argparse.ArgumentParser(description="Isolated Stage 3 single-pass research-candidate workflow")
    p.add_argument("--resume", type=Path, metavar="MANIFEST")
    p.add_argument("--config", type=Path)
    p.add_argument("--data-config", type=Path)
    p.add_argument("--source", choices=["tulu"])
    p.add_argument("--canvas-size")
    p.add_argument("--seed", type=int)
    p.add_argument("--target-policy", choices=["teacher-consistent", "reference-only"])
    p.add_argument("--experiment-dir", type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--qualify-only", action="store_true")
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--chat-port", type=int, default=7860)
    p.add_argument("--chat-seconds", type=float, default=7200)
    p.add_argument("--chat-requests", type=int, default=100)
    return p


def manual_chat(export, generator, args):
    return shlex.join(
        [
            sys.executable,
            "-m",
            "ncm_lab",
            "chat",
            "serve",
            "--checkpoint",
            export,
            "--device",
            generator["device"],
            "--profile",
            generator["hardware_profile"],
            "--port",
            str(args.chat_port),
            "--max-seconds",
            str(args.chat_seconds),
            "--max-requests",
            str(args.chat_requests),
        ]
    )


def launch_chat(path, args):
    _, state, settings = load(path)
    if state["status"] != "completed":
        raise ValueError("chat requires sealed final evaluation and reports")
    for name, entry in state["stages"].items():
        verify_record(name, read(checked(entry["receipt"])), path)
    for ref in state["reports"].values():
        checked(ref)
    result = read(checked(state["stages"]["generator"]["receipt"]))["result"]
    export = result["artifact"]
    generator = settings["stages"]["generator"]
    command = manual_chat(export, generator, args)
    show(
        dict(
            experiment=str(path),
            selected_export=export,
            canvas=settings["canvas"],
            selected_optimizer_step=result["selected_optimizer_step"],
            terminal_optimizer_step=result["optimizer_steps"],
            manual_chat=command,
            chat_url=f"http://127.0.0.1:{args.chat_port}"
            if args.chat_port
            else "assigned free loopback port printed by server",
        )
    )
    if args.no_chat:
        return 0
    ui = path.parent / "chat_status.json"
    try:
        from ncm_lab.chat.server import serve

        with exclusive_writer(path.parent / "chat.lock"):
            output = serve(
                checkpoint=export,
                device=generator["device"],
                profile=generator["hardware_profile"],
                port=args.chat_port,
                max_seconds=args.chat_seconds,
                max_requests=args.chat_requests,
                run_root=path.parent / "chat",
            )
        atomic_json(ui, {"status": "closed", "result": output})
    except KeyboardInterrupt:
        atomic_json(ui, {"status": "closed_by_operator", "training_status": "completed"})
    except (SystemExit, OSError, ValueError, RuntimeError, ImportError) as exc:
        retry = shlex.join(
            [
                sys.executable,
                "-u",
                "scripts/run_stage3.py",
                "--resume",
                str(path),
                "--execute",
                "--chat-port",
                "0",
            ]
        )
        atomic_json(
            ui, {"status": "failed", "error": str(exc), "training_status": "completed", "retry": retry}
        )
        print(
            f"Training/evaluation completed; UI failure: {exc}\nRetry: {retry}", file=sys.stderr, flush=True
        )
        return 3
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    path = None
    try:
        if args.resume:
            if any(
                getattr(args, k) is not None
                for k in (
                    "config",
                    "data_config",
                    "source",
                    "canvas_size",
                    "seed",
                    "target_policy",
                    "experiment_dir",
                )
            ):
                raise ValueError(
                    "resume uses the saved scientific recipe; scientific overrides require a new experiment"
                )
            path, state, settings = load(args.resume)
            if settings["schema"] != SCHEMA:
                raise ValueError("resume requires an explicit Stage 3 manifest")
            if not args.execute and not args.qualify_only:
                show({"experiment": str(path), "status": state["status"], "research_work_started": False})
                return 0
        else:
            settings = resolve(
                args.config,
                canvas_size=args.canvas_size,
                seed=args.seed,
                source=args.source,
                data_config=args.data_config,
                target_policy=args.target_policy,
            )
            executing = args.execute or args.qualify_only
            folder = args.experiment_dir
            if executing and folder is None:
                folder = root() / "runs/stage3" / settings["canvas"]["preset"] / identifier()
            proposal = plan(settings, folder)
            if not executing:
                show(proposal)
                return 0
            path = Path(proposal["experiment"])
        show(
            {
                "experiment": str(path),
                "action": "all_stages_then_chat"
                if args.execute and not args.no_chat
                else "all_stages"
                if args.execute
                else "source_capacity_only",
            }
        )
        result = run(path, stage="all" if args.execute else "source")
        show(result)
        if args.qualify_only:
            show({"final_target_capacity": "pending_distillation", "experiment": str(path)})
            return 0 if result["status"] in {"ready", "already_completed"} else 2
        if result["status"] not in {"completed", "already_completed"}:
            return 2
        return launch_chat(path, args)
    except (ValueError, TypeError, OSError, KeyError, ImportError, RuntimeError) as exc:
        print(f"Stage 3: {exc}\nManifest: {path}", file=sys.stderr, flush=True)
        return 2


def stage_main(name, argv=None):
    p = argparse.ArgumentParser(
        description=f"Run only the owned Stage 3 {name} stage; prerequisites must be sealed"
    )
    p.add_argument("--experiment", type=Path, required=True)
    args = p.parse_args(argv)
    _, _, settings = load(args.experiment)
    if settings["schema"] != SCHEMA:
        raise ValueError("Stage 3 experiment required")
    result = run(args.experiment, stage=name)
    show(result)
    return 0 if result["status"] in {"ready", "completed", "already_completed"} else 2
