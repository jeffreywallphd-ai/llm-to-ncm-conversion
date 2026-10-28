"""Thin explicit-input analysis CLI for SHR-001."""
import sys
from ncm_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["results", "analyze", "--experiment", "SHR-001", *sys.argv[1:]]))
