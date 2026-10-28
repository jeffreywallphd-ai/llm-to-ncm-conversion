"""Thin explicit-input analysis CLI for SHR-003."""
import sys
from ncm_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["results", "analyze", "--experiment", "SHR-003", *sys.argv[1:]]))
