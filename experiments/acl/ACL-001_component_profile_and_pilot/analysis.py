"""Thin explicit-input analysis CLI for ACL-001."""
import sys
from ncm_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["results", "analyze", "--experiment", "ACL-001", *sys.argv[1:]]))
