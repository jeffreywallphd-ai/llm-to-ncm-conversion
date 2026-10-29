"""Bounded native reference preparation from pinned source training data."""

import hashlib
import json
from ncm_lab.data.records import Example
from ncm_lab.data.artifacts import ArtifactWriter
from ncm_lab.teacher.model import model_identity, load_tokenizer
from ncm_lab.runtime.io import root, read, digest, canonical_hash
from ncm_lab.data.tokenization import prompt_tokens


def raw_examples(settings, index_path=None, check=None):
    path = root() / settings["source_path"]
    if settings.get("format", "jsonl") == "jsonl":
        with path.open(encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                if i >= settings["max_source_rows"]:
                    break
                row = json.loads(line)
                if row.get("split", "train") != "train":
                    raise ValueError("only source training releases are admitted")
                if row.get("source_type", "reference") != "reference" or row.get("censored", False):
                    raise ValueError(
                        "raw preparation admits complete references only; do not relabel generated/censored answers"
                    )
                yield row
    else:
        import pyarrow.parquet as pq

        columns = [
            "message_id",
            "parent_id",
            "message_tree_id",
            "text",
            "role",
            "lang",
            "deleted",
            "review_result",
        ]
        import sqlite3

        if index_path is None:
            raise ValueError("bounded on-disk OASST parent index required")
        database = sqlite3.connect(index_path)
        database.execute("PRAGMA cache_size=-4096")
        database.execute("PRAGMA temp_store=FILE")
        database.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY,parent TEXT,tree TEXT,text TEXT,role TEXT,lang TEXT,deleted INTEGER,approved INTEGER)"
        )
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
            if check:
                check()
            records = []
            for row in batch.to_pylist():
                count += 1
                if count > settings["max_source_rows"]:
                    break
                records.append(
                    (
                        row["message_id"],
                        row["parent_id"],
                        row["message_tree_id"],
                        row["text"],
                        row["role"],
                        row["lang"],
                        bool(row.get("deleted")),
                        row.get("review_result") is True,
                    )
                )
            database.executemany("INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?,?,?)", records)
            database.commit()
            if count >= settings["max_source_rows"]:
                break
        try:
            query = "SELECT a.id,a.tree,p.text,a.text FROM messages a JOIN messages p ON a.parent=p.id AND a.tree=p.tree WHERE a.role='assistant' AND p.role='prompter' AND a.lang='en' AND p.lang='en' AND a.deleted=0 AND p.deleted=0 AND a.approved=1 AND p.approved=1 ORDER BY a.id"
            for identity, tree, prompt, answer in database.execute(query):
                if check:
                    check()
                yield {
                    "original_id": identity,
                    "group_id": tree,
                    "prompt": prompt,
                    "answer": answer,
                    "split": "train",
                }
        finally:
            database.close()


def prepare_data(run, value):
    settings = value["data"]
    path = root() / settings["source_path"]
    source_sha = digest(path, run.meter.check)
    if settings.get("source_sha256"):
        if source_sha != settings["source_sha256"]:
            raise ValueError("source content changed; pin actual source SHA256")
    else:
        from ncm_lab.data.download import verify_asset

        asset = next(
            (
                a
                for a in read(root() / "configs/data/assets.lock.json")["assets"]
                if a["id"] == settings.get("source_asset_id")
            ),
            None,
        )
        if asset is None or verify_asset(path, asset, run.meter) is None:
            raise ValueError("verified pinned source asset required")
    tokenizer = load_tokenizer(value["model"])
    template = canonical_hash(tokenizer.chat_template)
    identity = {
        "model": model_identity(value["model"]),
        "template_sha256": template,
        "thinking": False,
        "source_sha256": source_sha,
        "canvas": value["architecture"]["canvas"],
        "source_revision": settings["revision"],
        "license": settings["license"],
    }
    writer = ArtifactWriter(run.path / "artifacts/data", "native_data_v1", identity, run.meter)
    seen = set()
    excluded = {"overlength": 0, "prompt_length": 0, "duplicate": 0}
    total = 0
    eos = tokenizer.eos_token_id
    for row in raw_examples(settings, run.path / "artifacts/source-index.sqlite", run.meter.check_disk):
        run.meter.check()
        key = canonical_hash([row["prompt"], row["answer"]])
        if key in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(key)
        prompt = prompt_tokens(tokenizer, row["prompt"])
        answer = tokenizer.encode(row["answer"], add_special_tokens=False)
        if eos in answer:
            answer = answer[: answer.index(eos)]
        answer = answer + [eos]
        canvas = value["architecture"]["canvas"]
        censored = False
        if len(answer) > canvas:
            if settings.get("overlength", "exclude") == "exclude":
                excluded["overlength"] += 1
                continue
            answer = answer[:canvas]
            censored = True
        if len(prompt) > settings["max_prompt_tokens"]:
            excluded["prompt_length"] += 1
            continue
        fraction = (
            int(
                hashlib.sha256((str(settings.get("seed", 17)) + ":" + row["group_id"]).encode()).hexdigest()[
                    :16
                ],
                16,
            )
            / 2**64
        )
        split = "dev" if fraction < settings.get("dev_fraction", 0.15) else "train"
        provenance = {
            "model_id": value["model"]["model_id"],
            "revision": value["model"]["revision"],
            "tokenizer_revision": value["model"]["tokenizer_revision"],
            "template_sha256": template,
            "thinking": False,
            "source_sha256": source_sha,
            "dataset_revision": settings["revision"],
            "license": settings["license"],
            "eos_ids": value["model"].get("eos_ids", [eos]),
        }
        example = Example(
            row["original_id"],
            row["group_id"],
            split,
            "software_fixture" if value["model"].get("fixture") else "reference",
            prompt,
            answer,
            not censored,
            censored,
            provenance,
        )
        amount = len(prompt) + len(answer)
        if total + amount > settings["max_unique_tokens"]:
            break
        run.meter.charge(source=amount, answer=len(answer), steps=1)
        total += amount
        writer.append(example)
        if writer.count >= settings["max_examples"]:
            break
    manifest = writer.finish(
        {
            "excluded": excluded,
            "requested_unique_tokens": settings["max_unique_tokens"],
            "unique_token_shortfall": max(0, settings["max_unique_tokens"] - total),
            "source_pool_target_is_not_evidence_of_availability": True,
        }
    )
    if not all(writer.groups.values()):
        raise ValueError("bounded source slice did not yield both whole-group train/dev partitions")
    run.publish(manifest)
    return {
        "artifact": str(manifest),
        "stats": writer.stats,
        "excluded": excluded,
        "unique_token_shortfall": max(0, settings["max_unique_tokens"] - total),
    }
