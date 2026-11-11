"""Plan or execute Block8, waiting for an active writer before resuming."""

import argparse
import json
from pathlib import Path
import sys
from time import sleep

from ncm_lab.block8.cli import add_arguments, dispatch
from ncm_lab.runtime.io import digest, read


def wait_for_writer(manifest: Path):
    manifest = manifest.resolve()
    lock = manifest.with_name("writer.lock")
    if not lock.exists():
        return

    print(f"Waiting for the current Block8 writer to release {lock}", file=sys.stderr, flush=True)
    watchdog = manifest.parent / "watchdog"
    while lock.exists():
        if watchdog.is_dir():
            report = watchdog / f"{digest(manifest)}.json"
            if report.is_file() and lock.exists():
                raise ValueError(read(report)["error"])
        sleep(5)

    state = read(manifest)
    prepare = state.get("stages", {}).get("prepare", {})
    if not state.get("generation_export") and prepare.get("status") != "completed":
        raise ValueError("the previous writer released its lock before preparation completed; check the experiment")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native hierarchical eight-token block-autoregressive workflow")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.resume and args.execute and not any(
            (args.config, args.experiment_dir, args.data_manifest, args.evaluation_manifest, args.checkpoint)
        ):
            wait_for_writer(args.resume)
        result = dispatch(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(json.dumps(dict(status="failed", error=str(exc))))
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in {"planned", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
