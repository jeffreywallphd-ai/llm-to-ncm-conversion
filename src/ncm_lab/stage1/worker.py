"""Internal synchronous heavy-stage entry; public scripts manage the experiment."""

import argparse
import json
from ncm_lab.config.stages import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    from ncm_lab.runtime.runner import execute

    result = execute(config(args.config, _bind_canvas=False), args.resume_checkpoint)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
