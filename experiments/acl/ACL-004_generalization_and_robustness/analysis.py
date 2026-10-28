"""Thin explicit-input analysis CLI for ACL-004."""
import sys
from ncm_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["results", "analyze", "--experiment", "ACL-004", *sys.argv[1:]]))
