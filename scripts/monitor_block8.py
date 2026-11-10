"""Read-only follower; can run from another checkout without installing or changing the trainer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ncm_lab.monitoring.block8 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
