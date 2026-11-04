"""Offline SQLite lifecycle regressions; fixtures establish software behavior only."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import pytest
from ncm_lab.data.prepare import raw_examples
from ncm_lab.runtime.io import digest


@pytest.mark.parametrize("path_type", ["path", "string", "relative"])
@pytest.mark.parametrize("existing_parent", [False, True])
def test_fresh_nested_index_and_existing_parent(
    tmp_path, monkeypatch, oasst_source, path_type, existing_parent
):
    index = tmp_path / "new run/stage attempt/artifacts/source-index.sqlite"
    if existing_parent:
        index.parent.mkdir(parents=True)
    else:
        assert not index.parent.exists()
    supplied = index
    if path_type == "string":
        supplied = str(index)
    elif path_type == "relative":
        monkeypatch.chdir(tmp_path)
        supplied = index.relative_to(tmp_path)
    before = oasst_source["path"].read_bytes()
    observations = {}
    assert list(raw_examples(oasst_source["settings"], supplied, observations=observations)) == oasst_source["expected"]
    assert index.is_file() and index.parent.is_dir()
    assert observations["source_rows_inspected"] == len(oasst_source["rows"])
    assert oasst_source["path"].read_bytes() == before
    assert digest(oasst_source["path"]) == oasst_source["settings"]["source_sha256"]


def test_row_limit_is_applied_before_parent_join(tmp_path, oasst_source):
    observations = {}
    settings = {**oasst_source["settings"], "max_source_rows": 3}
    index = tmp_path / "bounded scan/index.sqlite"
    assert list(raw_examples(settings, index, observations=observations)) == oasst_source["expected"][:1]
    assert observations == {"source_rows_inspected": 3}
    with closing(sqlite3.connect(index)) as database:
        assert database.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3


def tracked_connections(monkeypatch, *, fail_at=None):
    """Spy on actual SQLite connections, including their protected setup lifetime."""
    connections = []
    original = sqlite3.connect
    failure = sqlite3.OperationalError(f"injected {fail_at} failure")

    class Connection(sqlite3.Connection):
        close_calls = 0

        def execute(self, sql, *args):
            operation = (
                "pragma" if sql.startswith("PRAGMA") else "schema" if sql.startswith("CREATE") else "query"
            )
            if fail_at == operation:
                raise failure
            return super().execute(sql, *args)

        def executemany(self, *args):
            if fail_at == "insert":
                raise failure
            return super().executemany(*args)

        def commit(self):
            if fail_at == "commit":
                raise failure
            return super().commit()

        def close(self):
            self.close_calls += 1
            return super().close()

    def connect(path):
        database = original(path, factory=Connection)
        connections.append(database)
        return database

    monkeypatch.setattr(sqlite3, "connect", connect)
    return connections, failure


@pytest.mark.parametrize("early_close", [False, True])
def test_connection_closed_after_exhaustion_or_explicit_generator_close(
    tmp_path, monkeypatch, oasst_source, early_close
):
    connections, _ = tracked_connections(monkeypatch)
    reader = raw_examples(oasst_source["settings"], tmp_path / "index.sqlite")
    if early_close:
        assert next(reader) == oasst_source["expected"][0]
        assert connections[0].close_calls == 0
        reader.close()
    else:
        assert list(reader) == oasst_source["expected"]
    assert len(connections) == 1 and connections[0].close_calls == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("operation", ["pragma", "schema", "parquet", "insert", "commit", "query", "check"])
def test_connection_closed_for_setup_ingestion_and_query_errors(
    tmp_path, monkeypatch, oasst_source, operation
):
    import pyarrow.parquet as pq

    connections, failure = tracked_connections(monkeypatch, fail_at=operation)

    def fail(*args, **kwargs):
        raise failure

    if operation == "parquet":
        monkeypatch.setattr(pq, "ParquetFile", fail)
    with pytest.raises(sqlite3.OperationalError) as caught:
        list(raw_examples(
            oasst_source["settings"], tmp_path / "index.sqlite", check=fail if operation == "check" else None
        ))
    assert caught.value is failure  # SQL/ingestion errors are not recast as directory failures.
    assert len(connections) == 1 and connections[0].close_calls == 1


def test_parent_file_fails_without_replacement(tmp_path, oasst_source):
    parent = tmp_path / "artifacts"
    parent.write_text("retained diagnostic", encoding="utf-8")
    index = parent / "source-index.sqlite"
    with pytest.raises(RuntimeError, match="parent directory creation") as caught:
        list(raw_examples(oasst_source["settings"], index))
    assert isinstance(caught.value.__cause__, OSError)
    assert str(index) in str(caught.value)
    assert parent.read_text(encoding="utf-8") == "retained diagnostic"


@pytest.mark.parametrize("operation", ["mkdir", "connect"])
def test_io_errors_preserve_original_cause_and_intended_path(tmp_path, monkeypatch, oasst_source, operation):
    index = tmp_path / "new run/artifacts/source-index.sqlite"
    failure = PermissionError("injected filesystem denial")

    def fail(*args, **kwargs):
        raise failure

    if operation == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail)
    else:
        monkeypatch.setattr(sqlite3, "connect", fail)
    with pytest.raises(RuntimeError) as caught:
        list(raw_examples(oasst_source["settings"], index))
    assert caught.value.__cause__ is failure
    assert str(index) in str(caught.value)
    assert ("parent directory creation" if operation == "mkdir" else "connection") in str(caught.value)
    assert not index.exists()


def test_existing_index_schema_is_rejected_and_connection_closed(tmp_path, monkeypatch, oasst_source):
    index = tmp_path / "prior.sqlite"
    assert list(raw_examples(oasst_source["settings"], index)) == oasst_source["expected"]
    before = digest(index)
    connections, _ = tracked_connections(monkeypatch)
    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        list(raw_examples(oasst_source["settings"], index))
    assert connections[0].close_calls == 1
    assert digest(index) == before


def test_missing_index_rejected_before_connection(monkeypatch, oasst_source):
    def forbidden(*args, **kwargs):
        raise AssertionError("missing index must not connect")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with pytest.raises(ValueError, match="on-disk OASST parent index required"):
        list(raw_examples(oasst_source["settings"]))


def test_jsonl_remains_independent_of_sqlite_index(tmp_path, monkeypatch):
    row = {"original_id": "jsonl", "group_id": "group", "prompt": "question", "answer": "answer"}
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    parent = tmp_path / "not a directory"
    parent.write_text("keep", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("JSONL must not open SQLite")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    settings = {"source_path": str(source), "max_source_rows": 1}
    assert list(raw_examples(settings, parent / "index.sqlite")) == [row]
    assert list(raw_examples(settings)) == [row]
    assert parent.read_text(encoding="utf-8") == "keep"


def test_legacy_preparation_closes_owned_reader_on_example_limit(
    tmp_path, stage1_oasst_fixture, monkeypatch
):
    from dataclasses import asdict
    import uuid
    import ncm_lab.data.prepare as preparation
    from ncm_lab.data.artifacts import Dataset
    from ncm_lab.runtime.budget import Budget
    from ncm_lab.runtime.io import read
    from ncm_lab.runtime.runner import execute

    fixture = stage1_oasst_fixture
    value = read(fixture["experiment"]["stages"]["prepare"])
    value.pop("stage1", None)
    value.update(
        stage="data", run_root=str(tmp_path / "legacy runs"),
        output_ref=str(tmp_path / "legacy handoff.json"),
        budget=asdict(Budget(
            "legacy-index-" + uuid.uuid4().hex, max_wall_seconds=60, max_phase_wall_seconds=120,
            max_steps=2000, max_storage_bytes=20000000, max_phase_storage_bytes=40000000,
        )),
    )
    value["data"]["max_examples"] = 30
    readers = []

    def retain(*args, **kwargs):
        reader = raw_examples(*args, **kwargs)
        readers.append(reader)  # A reference prevents GC from hiding a missing close().
        return reader

    monkeypatch.setattr(preparation, "raw_examples", retain)
    connections, _ = tracked_connections(monkeypatch)
    result = execute(value)
    assert result["status"] == "completed", result
    assert len(Dataset(result["artifact"])) == 30
    assert len(readers) == 1 and readers[0].gi_frame is None
    assert len(connections) == 1 and connections[0].close_calls == 1
    assert digest(fixture["oasst"]["path"]) == fixture["oasst"]["settings"]["source_sha256"]


def test_stage1_consumer_error_closes_owned_reader(tmp_path, stage1_oasst_fixture, monkeypatch):
    import ncm_lab.stage1.data as preparation
    from ncm_lab.stage1.config import resolve
    from ncm_lab.stage1.workflow import plan, load, configuration
    from ncm_lab.runtime.runner import execute

    fixture = stage1_oasst_fixture
    experiment = plan(resolve(fixture["config"]), tmp_path / "candidate storage failure")["experiment"]
    path, state, settings = load(experiment)
    value = configuration("prepare", path, state, settings)
    # Explicit test-only cap forces an error in the consumer's loop body, after
    # the actual SQLite generator has yielded and remains suspended with a DB.
    value["stage1_data"]["max_candidate_bytes"] = 1
    readers = []

    def retain(*args, **kwargs):
        reader = raw_examples(*args, **kwargs)
        readers.append(reader)
        return reader

    monkeypatch.setattr(preparation, "raw_examples", retain)
    connections, _ = tracked_connections(monkeypatch)
    result = execute(value)
    assert result["status"] == "blocked" and "candidate storage" in result["error"]
    assert len(readers) == 1 and readers[0].gi_frame is None
    assert len(connections) == 1 and connections[0].close_calls == 1
    assert not Path(value["output_ref"]).exists()
