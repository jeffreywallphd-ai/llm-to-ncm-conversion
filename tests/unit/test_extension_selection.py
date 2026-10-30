import sqlite3
from collections import Counter
from ncm_lab.data.selection import fair_rows, SHARES, rank


def test_training_enrichment_without_replacement_and_representative_dev(tmp_path):
    connection = sqlite3.connect(tmp_path / "selection.sqlite")
    connection.execute("CREATE TABLE rows(key TEXT,split TEXT,band INTEGER,source TEXT,rank TEXT)")
    for split in ("train", "dev"):
        for band, total in ((0, 150), (4, 30)):
            for i in range(total):
                key = f"{split}-{band}-{i}"
                connection.execute(
                    "INSERT INTO rows VALUES (?,?,?,?,?)",
                    (key, split, band, "A" if i % 3 else "B", rank(key)),
                )
    train = list(fair_rows(connection, "train", SHARES))[:60]
    dev = list(fair_rows(connection, "dev"))[:60]
    assert len(train) == len(set(train)) == len(dev) == len(set(dev)) == 60
    assert sum("-4-" in key for key in train) > sum("-4-" in key for key in dev) * 2
    sources = Counter(
        connection.execute("SELECT source FROM rows WHERE key=?", (key,)).fetchone()[0] for key in train
    )
    assert abs(sources["A"] - 2 * sources["B"]) <= 3
    connection.close()


def test_acquisition_crash_reservation_survives_new_wave(tmp_path):
    from ncm_lab.data.download import AcquisitionMeter, PreparationBudget
    from ncm_lab.runtime.io import read

    budget = PreparationBudget(1, "fixture-acquisition", 1000, 10000, 30, 300, 0, 10000, 10, 1000, 0.15)
    first = AcquisitionMeter(tmp_path, budget, "selection")
    first.charge(100)
    assert read(first.path)["elapsed_seconds"] >= 30
    # Explicit operator inspection simulated only in this crash-accounting fixture.
    first.lock.unlink()
    second = AcquisitionMeter(tmp_path, budget, "selection")
    assert second.state["elapsed_seconds"] >= 60 and second.state["charged_download_bytes"] == 100
    second.close()
    assert read(first.path)["elapsed_seconds"] >= 30


def test_fast_child_output_is_bounded_before_parsing(tmp_path):
    import os
    import sys
    import pytest
    from ncm_lab.runtime.wave import execute_child

    folder = tmp_path / "child"
    with pytest.raises(ValueError, match="output limit"):
        execute_child(
            [sys.executable, "-m", "ncm_lab", "--help"], folder, 10, 16, dict(os.environ), lambda: None
        )
    assert (folder / "exit.json").is_file()


def test_timeout_stops_actual_interpreter_and_grandchild(tmp_path):
    import ctypes
    import os
    import sys
    import time
    from pathlib import Path
    import pytest
    from ncm_lab.runtime.wave import execute_child

    pid_file = tmp_path / "leaf.pid"
    helper = Path(__file__).parents[1] / "fixtures/process_tree.py"
    with pytest.raises(TimeoutError, match="child timeout"):
        execute_child(
            [sys.executable, str(helper), "parent", str(pid_file)],
            tmp_path / "tree",
            3,
            10000,
            dict(os.environ),
            lambda: None,
        )
    assert pid_file.is_file(), "nested child must have started for a meaningful cleanup check"
    process_id = int(pid_file.read_text())
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, process_id)
        if handle:
            try:
                assert kernel.WaitForSingleObject(handle, 5000) == 0
            finally:
                kernel.CloseHandle(handle)
    else:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                break
            status = Path(f"/proc/{process_id}/stat")
            if status.exists() and status.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("nested interpreter survived task-owned group termination")
