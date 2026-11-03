"""Thin launcher: the 100 step pilot, followed by its local chatbot."""

from ncm_lab.runtime.pilot import script_main

if __name__ == "__main__":
    raise SystemExit(script_main(100))
