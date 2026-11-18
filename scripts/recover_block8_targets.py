"""Back up or recover a Block8 target cache without starting generator training."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ncm_lab.block8.cache_recovery import backup_targets, recover_targets  # noqa: E402
from ncm_lab.runtime.budget import BudgetStop  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "recover"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    operation = backup_targets if args.action == "backup" else recover_targets
    try:
        result = operation(args.experiment.resolve(), args.backup.resolve())
    except (ValueError, OSError, KeyError, sqlite3.Error, BudgetStop) as exc:
        print(json.dumps(dict(status="failed", error=str(exc))))
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
