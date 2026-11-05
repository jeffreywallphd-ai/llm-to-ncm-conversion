"""Offline planning or an uninterrupted Stage 2 workflow followed by local chat."""

import argparse
import json
from pathlib import Path
import sys
from ncm_lab.runtime.canvas_pipeline import exclusive_writer
from ncm_lab.runtime.io import read, root
from ncm_lab.stage1.config import ACQUIRE, STAGE2_SCHEMA, resolve
from ncm_lab.stage1.workflow import checked, identifier, load, plan, run


def parser():
    p = argparse.ArgumentParser(description="Fixed 64-position Stage 2 codec, generator, evaluation and chat")
    p.add_argument("--resume", type=Path, metavar="MANIFEST")
    p.add_argument("--experiment-dir", type=Path, help="new directory; otherwise generated under runs/stage2")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--source", choices=("oasst", "tulu"), help="pinned OASST (default) or bounded Tülu")
    source.add_argument("--config", type=Path, help="explicit Stage 2 recipe")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true", help="default; no data scan or model allocation")
    action.add_argument(
        "--qualify-only", action="store_true", help="optional source capacity check without training"
    )
    action.add_argument("--execute", action="store_true", help="complete all stages and launch chat")
    p.add_argument("--chat-port", type=int, default=7860, help="loopback port; 0 selects an available port")
    p.add_argument("--chat-seconds", type=float, default=7200)
    p.add_argument("--chat-requests", type=int, default=100)
    return p


def show(value):
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)


def launch_chat(path, args):
    from ncm_lab.chat.server import serve

    _, state, settings = load(path)
    if state["status"] != "completed" or any(e["status"] != "completed" for e in state["stages"].values()):
        raise ValueError("chat requires completed training and locked evaluation")
    export = read(checked(state["stages"]["generator"]["receipt"]))["result"]["artifact"]
    generator = settings["stages"]["generator"]
    show({"stage2_selected_export": export, "launching_chat": True})
    try:
        with exclusive_writer(path.parent / "chat.lock"):
            result = serve(
                checkpoint=export,
                device=generator["device"],
                profile=generator["hardware_profile"],
                port=args.chat_port,
                max_seconds=args.chat_seconds,
                max_requests=args.chat_requests,
                run_root=path.parent / "chat",
            )
    except (SystemExit, OSError, ValueError, RuntimeError, ImportError) as exc:
        raise RuntimeError(
            f"training and evaluation completed, but chat did not start or stopped with an error: {exc}. "
            f'Retry with --resume "{path}" --execute --chat-port 0.'
        ) from exc
    show({"chat": result})


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.execute:
            from ncm_lab.chat.server import Limits

            Limits(max_seconds=args.chat_seconds, max_requests=args.chat_requests)
            if not 0 <= args.chat_port <= 65535:
                raise ValueError("chat port must be in 0..65535")
        executing = args.execute or args.qualify_only
        if args.resume:
            if args.source or args.config or args.experiment_dir:
                raise ValueError(
                    "--resume keeps the existing experiment's configuration, source and directory"
                )
            path, state, settings = load(args.resume)
            if settings["schema"] != STAGE2_SCHEMA:
                raise ValueError("resume requires a Stage 2 manifest")
            if not executing:
                show({"experiment": str(path), "status": state["status"], "research_work_started": False})
                return 0
        else:
            settings = resolve(
                args.config or root() / "configs/stage2/experiment.yaml", stage2_source=args.source
            )
            if settings["schema"] != STAGE2_SCHEMA:
                raise ValueError("--config must name a Stage 2 recipe")
            folder = args.experiment_dir
            if executing and folder is None:
                folder = root() / "runs" / "stage2" / identifier()
            proposal = plan(settings, folder)
            if not executing:
                proposal["plan"]["model_asset_acquisition"] = ACQUIRE
                show(proposal)
                return 0
            path = Path(proposal["experiment"])
        show({"experiment": str(path), "action": "train_and_chat" if args.execute else "qualify_source"})
        result = run(path, stage="all" if args.execute else "prepare")
        show(result)
        if args.qualify_only:
            return 0 if result["status"] in {"ready", "completed", "already_completed"} else 2
        if result["status"] not in {"completed", "already_completed"}:
            return 2
        launch_chat(path, args)
        return 0
    except (ValueError, TypeError, OSError, KeyError, ImportError, RuntimeError) as exc:
        print("Stage 2: " + str(exc), file=sys.stderr, flush=True)
        return 2
