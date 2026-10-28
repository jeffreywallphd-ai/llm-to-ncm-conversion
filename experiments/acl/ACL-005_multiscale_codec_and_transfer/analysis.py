"""Thin explicit-input analysis CLI for ACL-005."""
import sys
from ncm_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["results", "analyze", "--experiment", "ACL-005", *sys.argv[1:]]))
