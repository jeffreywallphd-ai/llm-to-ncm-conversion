"""Bounded storage-policy fixtures; no production corpus scan or model allocation."""

from contextlib import closing
from dataclasses import asdict
import json
import sqlite3
from types import SimpleNamespace
import zlib
import pytest
from ncm_lab.runtime.budget import Budget, BudgetStop
from ncm_lab.runtime.io import read
from ncm_lab.stage3.config import resolve
from ncm_lab.stage3.source_index import SourceIndex, SourceStorage


def storage(tmp_path, **changes):
    policy = SourceIndex(max_bytes=65536, candidate_bytes=4096, reserve_free_bytes=1, **changes)
    value = dict(
        data={"format": "tulu_parquet"},
        stage3={
            "source_index": asdict(policy),
            "payload_write_caps": {"per_run": 10000000, "aggregate": 30000000},
        },
    )
    run = SimpleNamespace(
        path=tmp_path,
        value=value,
        meter=SimpleNamespace(budget=Budget("storage-test", max_storage_bytes=1000000)),
    )
    return SourceStorage(run)


def test_defaults_keep_scan_candidate_and_write_caps_separate():
    s = resolve()
    source = s["stages"]["source"]["stage3"]
    assert s["qualification"]["max_source_index_bytes"] == 32000000000
    assert source["source_index"]["max_bytes"] == 32000000000
    assert source["source_index"]["candidate_bytes"] == source["cache_retained_bytes"] == 4000000000
    assert source["payload_write_caps"] == {"per_run": 128000000000, "aggregate": 384000000000}
    assert s["stages"]["distill"]["stage3"]["payload_write_caps"]["per_run"] == 40000000000
    assert s["recipe_label"] == "standard_research_candidate"


def test_compressed_payload_retains_all_ids_reference_and_provenance():
    p = SourceIndex()
    row = dict(
        prompt_ids=[1, 14, 6],
        answer_ids=list(range(32)) * 100,
        provenance={
            "reference_text": "Long original reference 日本語. " * 1000,
            "license": "fixture-license",
            "answer_origin": "mixed",
            "component_terms": "retained",
        },
    )
    encoded, raw_bytes = p.encode(row)
    assert isinstance(encoded, bytes) and len(encoded) < raw_bytes / 4
    assert p.decode(encoded) == p.decode(json.dumps(row)) == row


@pytest.mark.parametrize("bad", ["truncated", "trailing", "oversize", "not_object"])
def test_corrupt_or_oversized_payload_refused(bad):
    p = SourceIndex(max_payload_bytes=32)
    raw = b'{"x":1}'
    encoded = zlib.compress(raw)
    if bad == "truncated":
        encoded = encoded[:-1]
    elif bad == "trailing":
        encoded += b"extra"
    elif bad == "oversize":
        encoded = zlib.compress(b"x" * 33)
    else:
        encoded = zlib.compress(b"[]")
    with pytest.raises(ValueError):
        p.decode(encoded)
    with pytest.raises(BudgetStop, match="payload-byte"):
        p.encode({"x": "y" * 40})


def test_sqlite_really_enforces_page_limit_and_reports_exhaustion(tmp_path):
    guard = storage(tmp_path)
    guard.preflight()
    guard.path.parent.mkdir()
    with pytest.raises(sqlite3.OperationalError) as stopped, closing(sqlite3.connect(guard.path)) as db:
        guard.configure(db)
        assert db.execute("PRAGMA max_page_count").fetchone()[0] == 16
        db.execute("CREATE TABLE rows(payload BLOB)")
        for _ in range(100):
            db.execute("INSERT INTO rows VALUES(?)", (bytes(range(256)) * 8,))
            db.commit()
    assert stopped.value.sqlite_errorcode == sqlite3.SQLITE_FULL
    with pytest.raises(BudgetStop, match="NEW recipe/experiment"):
        guard.sqlite_failure(stopped.value)
    report = read(guard.report)
    assert report["category"] == "sqlite_or_filesystem_full" and report["index_bytes"] <= 65536
    assert report["model_loaded"] is False
    # The failed index is preserved and its connection is closed/reopenable.
    with closing(sqlite3.connect(guard.path)) as db:
        assert db.execute("SELECT count(*) FROM rows").fetchone()[0] > 0


def test_candidate_limit_is_independent_and_still_enforced(tmp_path):
    guard = storage(tmp_path)
    guard.preflight()
    folder = tmp_path / "artifacts/candidates"
    folder.mkdir(parents=True)
    (folder / "records.jsonl").write_bytes(b"x" * 4097)
    with pytest.raises(BudgetStop, match="candidate_byte_limit"):
        guard.check(phase="publish_candidates", publish=True)
    report = read(guard.report)
    assert report["candidate_bytes"] == 4097 and report["index_bytes"] == 0


def test_insufficient_disk_or_run_allowance_blocks_before_scan(tmp_path, monkeypatch):
    guard = storage(tmp_path)
    monkeypatch.setattr("ncm_lab.stage3.source_index.shutil.disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(BudgetStop, match="insufficient_free_disk"):
        guard.preflight()
    assert not guard.path.exists()
    assert read(guard.report)["required_free_bytes"] == 2 * 65536 + 4096 + 1
    guard.run.meter.budget = Budget("smaller-test", max_storage_bytes=65536)
    with pytest.raises(BudgetStop, match="retained_allowance_inconsistent"):
        guard.preflight()


@pytest.mark.parametrize("cap", [0, -1, True, 65535])
def test_invalid_index_allowance_rejected(cap):
    with pytest.raises(ValueError):
        SourceIndex(max_bytes=cap)


def test_offline_plan_exposes_full_scan_disk_allowance():
    from ncm_lab.stage1.workflow import plan

    proposal = plan(resolve())["plan"]
    assert proposal["source_storage"]["required_free_bytes"] == 88000000000
    assert proposal["source_storage"]["peak_source_allowance_bytes"] == 68000000000
    assert not proposal["research_work_started"]
