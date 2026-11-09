"""Bounded temporary source indexes, separate from the selected candidate cache."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import zlib
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.io import atomic_json, reference


@dataclass(frozen=True)
class SourceIndex:
    schema: str = "stage3_source_index_v1"
    encoding: str = "zlib_json_v1"
    max_bytes: int = 32000000000
    candidate_bytes: int = 4000000000
    max_payload_bytes: int = 16777216
    reserve_free_bytes: int = 20000000000

    def __post_init__(self):
        if self.schema != "stage3_source_index_v1" or self.encoding != "zlib_json_v1":
            raise ValueError("unsupported Stage 3 source index policy")
        for key in ("max_bytes", "candidate_bytes", "max_payload_bytes", "reserve_free_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (
                0 if key == "reserve_free_bytes" else 1
            ):
                raise ValueError("finite source storage allowance required: " + key)
        if self.max_bytes < 65536:
            raise ValueError("source index allowance must cover at least 64 KiB of SQLite structure")

    def encode(self, payload):
        raw = json.dumps(payload, allow_nan=False).encode("utf-8")
        if len(raw) > self.max_payload_bytes:
            raise BudgetStop("Stage 3 source row exceeds the declared uncompressed payload-byte limit")
        return zlib.compress(raw, level=3), len(raw)

    def decode(self, payload):
        if isinstance(payload, str):
            raw = payload.encode("utf-8")
        else:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, self.max_payload_bytes + 1)
            if len(raw) > self.max_payload_bytes or not decoder.eof or decoder.unused_data:
                raise ValueError("invalid/oversized compressed Stage 3 source row")
        if len(raw) > self.max_payload_bytes:
            raise ValueError("oversized Stage 3 source row")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Stage 3 source row must be an object")
        return value


class SourceStorage:
    def __init__(self, run):
        self.run = run
        self.policy = SourceIndex(**run.value["stage3"]["source_index"])
        name = (
            "preparation.sqlite" if run.value["data"].get("format") == "tulu_parquet" else "candidates.sqlite"
        )
        self.path = run.path / "artifacts" / name
        self.report = run.path / "source_storage.json"
        self.phase = "admission"
        self.scan = {}

    def snapshot(self, *, status="observed", category=None, **details):
        files = [self.path, *(Path(str(self.path) + s) for s in ("-journal", "-wal", "-shm"))]
        row = dict(
            schema="stage3_source_storage_v1",
            status=status,
            category=category,
            policy=asdict(self.policy),
            phase=self.phase,
            index_path=str(self.path),
            index_bytes=self.path.stat().st_size if self.path.is_file() else 0,
            index_and_journal_bytes=sum(p.stat().st_size for p in files if p.is_file()),
            candidate_bytes=self.output_bytes(),
            scan=dict(self.scan),
            scan_observation_scope="last completed storage check; later attempted work remains in native charges and artifacts/scan.json",
            free_bytes=shutil.disk_usage(self.run.path).free,
            peak_source_allowance_bytes=2 * self.policy.max_bytes + self.policy.candidate_bytes,
            peak_scope="main index plus one index-sized temporary workspace and selected caches; all run files also obey the native retained-storage budget",
            cumulative_write_caps=self.run.value["stage3"]["payload_write_caps"],
            model_loaded=False,
            **details,
        )
        atomic_json(self.report, row)
        return row

    def stop(self, category, **details):
        row = self.snapshot(status="blocked", category=category, **details)
        raise BudgetStop(
            f"Stage 3 source storage blocked ({category}): index_bytes={row['index_bytes']}, "
            f"index_limit_bytes={self.policy.max_bytes}, candidate_bytes={row['candidate_bytes']}, "
            f"candidate_limit_bytes={self.policy.candidate_bytes}; report: {self.report}. "
            "Keep this failed run. Revise explicit qualification.max_source_index_bytes or "
            "max_candidate_bytes in a NEW recipe/experiment if needed; do not edit old manifests or reset ledgers."
        )

    def preflight(self):
        required = 2 * self.policy.max_bytes + self.policy.candidate_bytes
        if required > self.run.meter.budget.max_storage_bytes:
            self.stop(
                "retained_allowance_inconsistent",
                required_retained_bytes=required,
                native_run_limit_bytes=self.run.meter.budget.max_storage_bytes,
            )
        if shutil.disk_usage(self.run.path).free < required + self.policy.reserve_free_bytes:
            self.stop("insufficient_free_disk", required_free_bytes=required + self.policy.reserve_free_bytes)
        self.snapshot(status="admitted")

    def configure(self, db):
        pagesize = db.execute("PRAGMA page_size").fetchone()[0]
        pages = self.policy.max_bytes // pagesize
        if db.execute(f"PRAGMA max_page_count={pages}").fetchone()[0] > pages:
            self.stop("existing_index_exceeds_limit")

    def output_bytes(self):
        return sum(
            p.stat().st_size
            for name in ("candidates", "locked")
            for p in (self.run.path / "artifacts" / name).rglob("*")
            if p.is_file()
        )

    def check(self, *, phase, scan=None, publish=False):
        self.phase = phase
        if scan is not None:
            self.scan = {k: scan.get(k) for k in ("rows_read", "release_rows", "indexed_rows", "complete")}
        if self.path.is_file() and self.path.stat().st_size > self.policy.max_bytes:
            self.stop("index_byte_limit")
        if publish and self.output_bytes() > self.policy.candidate_bytes:
            self.stop("candidate_byte_limit")
        if shutil.disk_usage(self.run.path).free < self.policy.reserve_free_bytes:
            self.stop("free_disk_reserve")

    def sqlite_failure(self, exc):
        if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
            self.stop("sqlite_or_filesystem_full", sqlite_error=str(exc))
        raise exc

    def finish(self):
        self.check(phase="completed", publish=True)
        self.snapshot(status="completed")
        return reference(self.report)
