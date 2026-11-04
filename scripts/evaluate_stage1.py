"""Evaluate the selected export; never train, select, repair, or start a server."""

from ncm_lab.stage1.cli import main

if __name__ == "__main__":
    raise SystemExit(main(fixed_stage="evaluate"))
