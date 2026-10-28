"""Thin explicit-input analysis CLI for IS-001."""
import sys
from ncm_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["results", "analyze", "--experiment", "IS-001", *sys.argv[1:]]))
