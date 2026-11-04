"""Train/resume only the selected experiment's prompt-only generator."""

from ncm_lab.stage1.cli import main

if __name__ == "__main__":
    raise SystemExit(main(fixed_stage="generator"))
