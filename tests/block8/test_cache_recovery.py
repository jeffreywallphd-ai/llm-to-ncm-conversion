from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from ncm_lab.block8 import cache_recovery
from ncm_lab.block8.runtime import Attempt, checked
from ncm_lab.block8.targets import TensorWriter
from ncm_lab.block8.workflow import run_workflow
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.io import digest, read

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="DGX target recovery uses Linux boot identity")


def test_interrupted_targets_reuse_live_rows_and_independent_backup(block8_fixture, monkeypatch):
    f = block8_fixture
    original_append = TensorWriter.append

    def interrupt(writer, key, parent, tensor):
        if writer.run.action == "targets" and sum(writer.count.values()) == 3:
            raise RuntimeError("interrupted target cache fixture")
        original_append(writer, key, parent, tensor)

    monkeypatch.setattr(TensorWriter, "append", interrupt)
    failed = run_workflow(
        f["settings"], execute=True, experiment_dir=f["folder"] / "interrupted run",
        data_manifest=f["data"], evaluation_manifest=f["evaluation"],
    )
    assert failed["status"] == "failed", failed
    manifest = Path(failed["experiment"])
    old = read(manifest)
    old_config = digest(manifest.parent / "resolved_config.json")
    original_receipt = checked(old["stages"]["targets"]["receipt"])
    old_cache = original_receipt.parent / "cache"
    backup = f["folder"] / "cache backup"
    monkeypatch.setattr(cache_recovery.os, "sync", lambda: None)
    result = cache_recovery.backup_targets(manifest, backup)
    assert result["parents"] == 3
    with sqlite3.connect(checked(result["index"])) as db:
        files = db.execute("SELECT file,sha FROM tensors ORDER BY parent").fetchall()
    for name, sha in files:
        assert digest(backup / "cache" / name) == sha
        assert (backup / "cache" / name).stat().st_ino != (old_cache / name).stat().st_ino
    for number in range(f["settings"]["resources"]["max_attempts_per_stage"] - 1):
        (backup / f"index-previous-{number}.sqlite").touch()
    with pytest.raises(BudgetStop, match="finite snapshot allowance"):
        cache_recovery.backup_targets(manifest, backup)
    assert read(backup / "manifest.json") == result
    (old_cache / files[0][0]).unlink()
    (old_cache / files[1][0]).write_bytes(b"damaged original file")
    monkeypatch.setattr(TensorWriter, "append", original_append)

    restored = cache_recovery.recover_targets(manifest, backup)
    assert restored["status"] == "completed", restored
    receipt = read(checked(restored["receipt"]))
    assert receipt["charged_work"]["steps"] == 5
    summary = read(checked(receipt["result"]["recovery"]))["restored"]
    assert summary[str(old_cache / "index.sqlite")] == 1
    assert summary[result["index"]["manifest"]] == 2
    assert digest(manifest.parent / "resolved_config.json") == old_config
    assert checked(old["stages"]["targets"]["receipt"]) == original_receipt
    assert "generator" not in read(manifest)["stages"]

    completed = run_workflow(resume=manifest, execute=True)
    assert completed["status"] == "completed", completed
    state = read(manifest)
    assert state["attempts"]["targets"] == 2
    assert state["stages"]["targets"]["receipt"] == restored["receipt"]


def test_recovery_preserves_a_live_writer_lock(tmp_path):
    lock = tmp_path / "writer.lock"
    lock.write_text("active owner")
    archive = tmp_path / "archived"
    with pytest.raises(ValueError, match="writer may still be active"):
        cache_recovery._archive_stale_lock(lock, {"boot_id": cache_recovery._boot_id()}, archive)
    assert lock.read_text() == "active owner"
    assert not archive.exists()


def test_recovery_archives_a_lock_from_before_the_reboot(tmp_path):
    lock = tmp_path / "writer.lock"
    lock.write_text("previous owner")
    cache_recovery.os.utime(lock, (1, 1))
    archive = tmp_path / "archived"
    cache_recovery._archive_stale_lock(lock, {"boot_id": "different boot"}, archive)
    assert not lock.exists()
    assert (archive / lock.name).read_text() == "previous owner"


def test_hot_sqlite_journal_is_recovered_on_a_private_copy(block8_fixture):
    f = block8_fixture
    index = f["folder"] / "index.sqlite"
    subprocess.run(
        [sys.executable, "-c", """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('CREATE TABLE tensors(parent,split,length,file,sha,dtype,width,parent_sha)')
db.executemany('INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
               [(i, 'train', 1, 'p', 'a' * 4000, 'float32', 8, 'sha') for i in range(1, 33)])
db.commit()
db.execute('PRAGMA cache_size=2')
db.execute('DELETE FROM tensors')
os._exit(3)
""", str(index)],
        timeout=15, check=False,
    )
    journal = index.with_suffix(".sqlite-journal")
    originals = digest(index), digest(journal)
    with pytest.raises(sqlite3.OperationalError), sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("SELECT count(*) FROM tensors").fetchone()
    run = Attempt(f["folder"] / "hot journal", "targets", f["settings"])
    rows = cache_recovery._recovery_rows(index, 32, run, backup=False)
    assert [row[0] for row in rows] == list(range(1, 33))
    assert (digest(index), digest(journal)) == originals
    assert not list(run.path.glob("recovery-index-*"))
    run.close("completed", {})
