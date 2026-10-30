"""Bounded authored software fixture, never model-generated or dataset Python."""

from pathlib import Path
import os
import subprocess
import sys
import time


if __name__ == "__main__":
    if sys.argv[1] == "leaf":
        Path(sys.argv[2]).write_text(str(os.getpid()))
        time.sleep(12)
    else:
        child = subprocess.Popen(
            [sys.executable, __file__, "leaf", sys.argv[2]],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            child.wait(timeout=12)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)
