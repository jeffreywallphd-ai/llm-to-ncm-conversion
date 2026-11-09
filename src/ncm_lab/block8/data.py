"""Sealed parent answers and disk-backed offset views; no duplicated prefix files."""

from collections import Counter
from contextlib import closing
from dataclasses import asdict
import json
import sqlite3
from ncm_lab.block8.config import DataPolicy
from ncm_lab.data.artifacts import Dataset, LOCKED
from ncm_lab.data.records import Example
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.io import canonical_hash, digest, read, reference, resolve_reference, write_json
from ncm_lab.stage3.source_index import SourceIndex
from ncm_lab.teacher.model import model_identity, load_tokenizer

SCHEMA = "hierarchical_block8_parent_offset_data_v1"


def prepare(run, settings, inputs):
    try:
        return _prepare(run, settings, inputs)
    except sqlite3.Error as exc:
        if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_FULL:
            raise
        path = run.path / "dataset/parents.sqlite"
        write_json(
            run.path / "dataset/storage_failure.json",
            dict(
                schema="block8_data_storage_failure_v1",
                index_bytes=path.stat().st_size,
                max_index_bytes=settings["data"]["max_index_bytes"],
                error=str(exc),
                failed_index_preserved=True,
            ),
        )
        raise BudgetStop("block8 parent/offset SQLite allowance exhausted; failed index preserved") from exc


def _prepare(run, settings, inputs):
    policy = DataPolicy(**settings["data"])
    if not 1 <= len(inputs) <= 2:
        raise ValueError("supply one train/dev artifact and at most one locked evaluation artifact")
    from ncm_lab.block8.runtime import checked
    from ncm_lab.data.sources.tulu import tokenizer_binding

    checked(settings["model_asset_policy"])
    for ref in settings["source"]["policy_receipts"].values():
        checked(ref)
    tokenizer_files = tokenizer_binding(settings["source"]["resolved_data"], settings["model"], run.meter)
    tokenizer = load_tokenizer(settings["model"])
    template = canonical_hash(tokenizer.chat_template)
    folder = run.path / "dataset"
    folder.mkdir()
    path = folder / "parents.sqlite"
    codec = SourceIndex(max_bytes=policy.max_index_bytes, max_payload_bytes=policy.max_row_bytes)
    counts, views, excluded = Counter(), Counter(), Counter()
    origins, answer_positions, prompt_positions = Counter(), Counter(), Counter()
    receipts = []
    with closing(sqlite3.connect(path)) as db:
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={policy.max_index_bytes // page_size}")
        db.executescript("""
            CREATE TABLE parents(id INTEGER PRIMARY KEY, original_id TEXT UNIQUE, group_id TEXT, split TEXT, payload BLOB, sha TEXT);
            CREATE TABLE aliases(alias TEXT PRIMARY KEY, split TEXT);
            CREATE TABLE views(parent INTEGER, start INTEGER, split TEXT, PRIMARY KEY(parent,start));
            CREATE INDEX view_split ON views(split,parent,start);
        """)
        seen = 0
        for source in inputs:
            source = resolve_reference(source)
            # Hash with the run's clock checks before the shared reader verifies it again.
            digest(source, run.meter.check)
            artifact = Dataset(source)
            if artifact.manifest["schema"] not in {
                "native_data_v1",
                "stage3_candidates_v1",
                "stage3_distilled_v1",
                LOCKED,
            }:
                raise ValueError("block8 accepts tokenized reference artifacts, not feature/code caches")
            identity = artifact.manifest["identity"]
            if (
                identity["model"] != model_identity(settings["model"])
                or identity["template_sha256"] != template
            ):
                raise ValueError("source model/tokenizer/template differs from block8 configuration")
            receipts.append(reference(source))
            for index in range(len(artifact)):
                seen += 1
                if seen > policy.max_parents:
                    raise BudgetStop("block8 source-parent allowance exhausted; no partial dataset published")
                example, _ = artifact[index]
                if (artifact.manifest["schema"] == LOCKED) != (example.split == "test"):
                    raise ValueError("locked block8 evaluation lineage cannot enter a training artifact")
                if (example.source_type == "software_fixture") != bool(
                    settings["model"].get("fixture")
                ) or example.provenance.get("template_sha256") != template:
                    raise ValueError("block8 record lineage/template differs from its source artifact")
                if not settings["model"].get("fixture") and any(
                    example.provenance.get(k) != settings["model"].get(k)
                    for k in ("model_id", "revision", "tokenizer_revision", "thinking")
                ):
                    raise ValueError("block8 record model/tokenizer pin differs")
                run.meter.charge(
                    source=len(example.prompt_ids) + len(example.answer_ids),
                    answer=len(example.answer_ids),
                    steps=1,
                )
                row_bytes = len(json.dumps(asdict(example)).encode())
                if row_bytes > policy.max_row_bytes:
                    raise BudgetStop("block8 parent exceeds declared row-byte allowance")
                from ncm_lab.block8.runtime import charge_write

                charge_write(run, 2 * row_bytes + 16384)
                # Check lineage/overlap BEFORE length filtering; no split movement is allowed.
                for alias in ("group:" + example.group_id, "prompt:" + canonical_hash(example.prompt_ids)):
                    previous = db.execute("SELECT split FROM aliases WHERE alias=?", (alias,)).fetchone()
                    if previous and previous[0] != example.split:
                        raise ValueError("conversation/duplicate prompt crosses block8 splits")
                    db.execute("INSERT OR IGNORE INTO aliases VALUES(?,?)", (alias, example.split))
                eos = settings["model"]["eos_ids"]
                if (
                    not example.eos_genuine
                    or example.censored
                    or example.answer_ids[-1] not in eos
                    or any(x in eos for x in example.answer_ids[:-1])
                ):
                    excluded["incomplete_or_invalid_eos"] += 1
                    continue
                if len(example.prompt_ids) > policy.max_prompt_tokens:
                    excluded["prompt_length"] += 1
                    continue
                if len(example.answer_ids) > policy.max_answer_tokens:
                    excluded["answer_length"] += 1
                    continue
                if any(i >= len(tokenizer) or i < 0 for i in example.prompt_ids + example.answer_ids):
                    raise ValueError("source token outside pinned tokenizer vocabulary")
                payload = asdict(example)
                encoded, logical = codec.encode(payload)
                starts = list(range(0, len(example.answer_ids), 8))
                if sum(views.values()) + len(starts) > policy.max_views:
                    raise BudgetStop("block8 offset-view allowance exhausted")
                from ncm_lab.block8.runtime import charge_write

                charge_write(run, logical + 16384 + len(starts) * 256)
                cursor = db.execute(
                    "INSERT INTO parents(original_id,group_id,split,payload,sha) VALUES(?,?,?,?,?)",
                    (example.original_id, example.group_id, example.split, encoded, example.identity),
                )
                db.executemany(
                    "INSERT INTO views VALUES(?,?,?)",
                    [(cursor.lastrowid, start, example.split) for start in starts],
                )
                counts[example.split] += 1
                views[example.split] += len(starts)
                answer_positions[example.split] += len(example.answer_ids)
                prompt_positions[example.split] += len(example.prompt_ids)
                origins[example.source_type] += 1
                db.commit()
                run.meter.check_disk()
        for split in ("train", "dev", "test"):
            if counts[split] < getattr(policy, "minimum_" + split):
                write_json(
                    folder / "capacity_failure.json", dict(parents=dict(counts), excluded=dict(excluded))
                )
                raise ValueError(
                    f"block8 {split} parent capacity: need {getattr(policy, 'minimum_' + split)}, found {counts[split]}"
                )
        db.execute("PRAGMA optimize")
    manifest = dict(
        schema=SCHEMA,
        policy=asdict(policy),
        architecture=settings["architecture"],
        model=model_identity(settings["model"]),
        template_sha256=template,
        tokenizer_files=tokenizer_files,
        parents_file=path.name,
        parents_sha256=digest(path, run.meter.check),
        parents=dict(counts),
        views=dict(views),
        unique_answer_positions=dict(answer_positions),
        unique_prompt_positions=dict(prompt_positions),
        excluded=dict(excluded),
        origins=dict(origins),
        sources=receipts,
        parent_rows_inspected=seen,
        scope="unique parent records/token positions; offset views are derived exposures, not new answers",
        target_policy=policy.target_policy,
        broader_overlap_scope="supplied artifacts and their sealed source policies only",
    )
    write_json(folder / "manifest.json", manifest, exclusive=True)
    return reference(folder / "manifest.json")


class BlockDataset:
    def __init__(self, path, settings, check=None):
        path = resolve_reference(path)
        self.path, self.manifest = path, read(path)
        value = self.manifest
        if (
            value.get("schema") != SCHEMA
            or value["policy"] != settings["data"]
            or value["architecture"] != settings["architecture"]
            or value["model"] != model_identity(settings["model"])
        ):
            raise ValueError("incompatible block8 dataset contract")
        self.file = (path.parent / value["parents_file"]).resolve()
        if self.file.parent != path.parent.resolve() or digest(self.file, check) != value["parents_sha256"]:
            raise ValueError("changed block8 parent/offset index")
        self.codec = SourceIndex(max_payload_bytes=value["policy"]["max_row_bytes"])
        self.sha256 = digest(path, check)
        self.db = sqlite3.connect(self.file.as_uri() + "?mode=ro&immutable=1", uri=True)
        self.ids = {
            split: [
                r[0] for r in self.db.execute("SELECT id FROM parents WHERE split=? ORDER BY id", (split,))
            ]
            for split in ("train", "dev", "test")
        }
        if {s: len(ids) for s, ids in self.ids.items()} != value["parents"]:
            self.close()
            raise ValueError("block8 parent counts changed")

    def close(self):
        self.db.close()

    def parent(self, key):
        row = self.db.execute("SELECT payload,sha FROM parents WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("missing block8 parent")
        example = Example(**self.codec.decode(row[0]))
        if example.identity != row[1]:
            raise ValueError("changed block8 parent record")
        return example

    def view(self, key, start, size=8):
        example = self.parent(key)
        if size != 8 or start % size or not 0 <= start < len(example.answer_ids):
            raise ValueError("invalid hierarchical block offset")
        # No re-rendering, no new assistant turn, no decode/retokenize boundary changes.
        return example.prompt_ids + example.answer_ids[:start], example.answer_ids[start : start + size]

    def views(self, split, limit):
        return self.db.execute(
            "SELECT parent,start FROM views WHERE split=? ORDER BY parent,start LIMIT ?", (split, limit)
        ).fetchall()


def import_tulu(run, settings):
    """Compose the approved complete-scan adapter without distillation or native training."""
    from ncm_lab.stage3.data import source
    from ncm_lab.stage3.config import Qualification
    from ncm_lab.stage3.source_index import SourceIndex
    from ncm_lab.block8.runtime import checked

    for ref in settings["source"]["policy_receipts"].values():
        checked(ref)

    cfg = dict(run.value)
    source_settings = dict(settings["source"]["resolved_data"])
    source_settings["max_prompt_tokens"] = settings["data"]["max_prompt_tokens"]
    source_settings["max_answer_tokens"] = settings["data"]["max_answer_tokens"]
    q = Qualification(
        train_examples=settings["data"]["minimum_train"],
        minimum_train_examples=settings["data"]["minimum_train"],
        dev_examples=settings["data"]["minimum_dev"],
        test_examples=settings["data"]["minimum_test"],
        max_source_index_bytes=settings["source"]["max_source_index_bytes"],
        max_candidate_bytes=settings["source"]["max_candidate_bytes"],
    )
    cfg.update(
        data=source_settings,
        model=settings["model"],
        architecture={"canvas": settings["data"]["max_answer_tokens"], "latent_width": 384},
        stage1_data=asdict(q),
        stage1_context={"experiment": str(run.experiment / "manifest.json"), "stage": "block8_source"},
        stage3=dict(
            source_index=asdict(
                SourceIndex(
                    max_bytes=q.max_source_index_bytes,
                    candidate_bytes=q.max_candidate_bytes,
                    reserve_free_bytes=settings["resources"]["reserve_free_bytes"],
                )
            ),
            distillation={"max_candidates": settings["source"]["max_candidates"]},
            previous_locked_manifests=[],
            payload_write_caps={
                "per_run": settings["resources"]["max_write_bytes"],
                "aggregate": settings["resources"]["max_phase_write_bytes"],
            },
        ),
    )
    run.value = cfg
    write_json(run.path / "source_adapter_config.json", cfg, exclusive=True)
    run.meter.stage3_run = run
    result = source(run, cfg)
    return dict(
        data=reference(result["artifact"]),
        evaluation=result["evaluation_data"],
        qualification=result["qualification"],
        source_storage=result["source_storage"],
    )
