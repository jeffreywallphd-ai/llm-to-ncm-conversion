"""Plan by default; --execute explicitly starts the separate block8 workflow."""

from ncm_lab.block8.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
