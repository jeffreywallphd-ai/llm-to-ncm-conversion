"""Bounded multi-shard complete-context Tülu preparation with a disk-backed index."""

from collections import Counter
import json
import math
import sqlite3
import unicodedata
from ncm_lab.data.artifacts import ArtifactWriter, Dataset
from ncm_lab.data.records import Example
from ncm_lab.data.download import asset_path, verify_asset
from ncm_lab.data.tokenization import validated_messages, prompt_tokens
from ncm_lab.data.selection import band, fair_rows, rank
from ncm_lab.data.language import EnglishFilter
from ncm_lab.data.capacity import create_capacity
from ncm_lab.runtime.io import root, read, canonical_hash, atomic_json, reference
from ncm_lab.teacher.model import load_tokenizer, model_identity

DATASET = "allenai/tulu-3-sft-mixture"
REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
ADMISSION_ORDER = [
    "row_identity",
    "source_inventory",
    "source_policy",
    "message_bounds_and_roles",
    "special_delimiters",
    "language",
    "render",
    "interior_eos",
    "lengths",
    "oasst_overlap",
    "canonical_dedup_and_group_union",
    "group_split",
    "pool_selection",
]


def source_assets(settings, lock=None):
    lock = lock or read(root() / settings["asset_lock"])
    ids = settings.get("source_asset_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("explicit unique source asset list required")
    lookup = {a["id"]: a for a in lock["assets"]}
    result = []
    for key in sorted(ids):
        asset = lookup.get(key)
        if not asset or (
            asset.get("repo_id"),
            asset.get("revision"),
            asset.get("split"),
            asset.get("role"),
        ) != (DATASET, settings["revision"], "train", "tulu3"):
            raise ValueError("Tulu source must bind pinned dataset/revision/train/role, never notices")
        if (
            not asset["filename"].endswith(".parquet")
            or not asset.get("sha256")
            or len(asset["sha256"]) != 64
        ):
            raise ValueError("training Parquet content SHA256 required")
        asset_path(root() / settings["assets_root"], asset)
        result.append(asset)
    if settings["revision"] != REVISION:
        raise ValueError("unsupported Tulu release; explicitly version the adapter/policy")
    return result


def canonical(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def validate_limits(settings):
    caps = {
        "max_source_rows": 1200000,
        "parquet_batch_rows": 128,
        "max_messages": 300,
        "max_row_chars": 131072,
        "max_prompt_tokens": 512,
        "max_answer_tokens": 1024,
        "max_examples": 1000000,
        "max_unique_tokens": 100000000,
        "codec_minimum_answer_positions": 100000000,
        "codec_dev_examples": 256,
        "codec_validation_examples": 256,
        "generator_processed_tokens": 100000000,
    }
    if any(
        type(settings.get(key)) is not int or not 0 < settings[key] <= maximum
        for key, maximum in caps.items()
    ):
        raise ValueError("positive bounded integer preparation limits required")
    if type(settings["seed"]) is not int or settings["seed"] < 0 or not 0 < settings["dev_fraction"] < 1:
        raise ValueError("invalid deterministic split policy")
    limits = settings["split_token_caps"]
    if set(limits) != {"train", "dev"} or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("finite positive train/dev token caps required")
    language = settings["language"]
    if (
        not math.isfinite(language["minimum_probability"])
        or not 0 < language["minimum_probability"] <= 1
        or not 1 <= language["min_evidence_chars"] <= language["max_evidence_chars"] <= 4096
    ):
        raise ValueError("invalid bounded language-evidence policy")


def tokenizer_binding(settings, spec, meter):
    if spec.get("fixture"):
        return {"scope": "software fixture tokenizer"}
    from ncm_lab.config.architecture import PIN

    if (spec["model_id"], spec["revision"], spec["tokenizer_revision"]) != ("Qwen/Qwen3-8B", PIN, PIN):
        raise ValueError("pinned Qwen model/tokenizer identity required")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}
    assets = [
        a
        for a in read(root() / settings["asset_lock"])["assets"]
        if a["role"] == "target" and a["filename"] in required
    ]
    if {a["filename"] for a in assets} != required:
        raise ValueError("complete tokenizer-only lock selection required")
    verified = {}
    for asset in assets:
        path = asset_path(root() / settings["assets_root"], asset)
        if asset["revision"] != PIN or path != (root() / spec["path"] / asset["filename"]).resolve():
            raise ValueError("tokenizer path/revision differs from canonical selected assets")
        sha = verify_asset(path, asset, meter)
        if not sha:
            raise ValueError("missing/corrupt pinned tokenizer asset: " + asset["id"])
        verified[asset["filename"]] = sha
    return verified


class Groups:
    def __init__(self, database):
        self.database = database
        database.execute("CREATE TABLE groups(node TEXT PRIMARY KEY,parent TEXT)")

    def find(self, node):
        self.database.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (node, node))
        trail = []
        while True:
            parent = self.database.execute("SELECT parent FROM groups WHERE node=?", (node,)).fetchone()[0]
            if parent == node:
                break
            trail.append(node)
            node = parent
        self.database.executemany("UPDATE groups SET parent=? WHERE node=?", [(node, item) for item in trail])
        return node

    def union(self, left, right):
        left, right = sorted((self.find(left), self.find(right)))
        self.database.execute("UPDATE groups SET parent=? WHERE node=?", (left, right))
        return left


def overlap_index(database, settings, tokenizer, run):
    """All explicit pilot partitions and/or the pinned English approved OASST release."""
    database.execute("CREATE TABLE overlap(prompt TEXT PRIMARY KEY)")
    policy = settings["overlap"]
    identities = {}
    count = 0
    if policy.get("artifact"):
        data = Dataset(policy["artifact"], "native_data_v1")
        if data.manifest["identity"]["template_sha256"] != canonical_hash(tokenizer.chat_template):
            raise ValueError("overlap artifact uses a different chat renderer")
        identities["artifact_sha256"] = data.sha256
        for _, row in data.iter_metadata():
            run.meter.check()
            database.execute(
                "INSERT OR IGNORE INTO overlap VALUES (?)", (canonical_hash(row["example"]["prompt_ids"]),)
            )
            count += 1
    if policy.get("source_asset_id"):
        from ncm_lab.data.prepare import raw_examples
        import pyarrow.parquet as pq

        lock = read(root() / settings["asset_lock"])
        asset = next(a for a in lock["assets"] if a["id"] == policy["source_asset_id"])
        if (
            asset["role"] != "oasst1"
            or asset["split"] != "train"
            or not asset["filename"].endswith(".parquet")
        ):
            raise ValueError("overlap requires pinned OASST training data")
        path = asset_path(root() / settings["assets_root"], asset)
        sha = verify_asset(path, asset, run.meter)
        if not sha:
            raise ValueError("OASST overlap source missing or corrupt")
        if pq.ParquetFile(path).metadata.num_rows > policy["max_source_rows"]:
            raise ValueError("overlap row bound cannot inspect the pinned source completely")
        identities["source_sha256"] = sha
        options = {
            "format": "oasst_parquet",
            "source_path": str(path),
            "max_source_rows": policy["max_source_rows"],
        }
        for row in raw_examples(
            options, run.path / "artifacts/oasst-overlap-source.sqlite", run.meter.check_disk
        ):
            ids = prompt_tokens(tokenizer, row["prompt"])
            answer = tokenizer.encode(row["answer"], add_special_tokens=False)
            run.meter.charge(steps=1, source=len(ids) + len(answer), answer=len(answer))
            if len(ids) <= policy["max_prompt_tokens"]:
                database.execute("INSERT OR IGNORE INTO overlap VALUES (?)", (canonical_hash(ids),))
                count += 1
    if not identities:
        raise ValueError("explicit hash-bound OASST overlap input required")
    database.commit()
    return {
        "inputs": identities,
        "policy": policy,
        "references_considered": count,
        "distinct_prompts": database.execute("SELECT count(*) FROM overlap").fetchone()[0],
        "coverage": "Exact rendered prompt matches across supplied train/dev; not paraphrase or benchmark decontamination.",
    }


def prepare_tulu(run, value):
    import pyarrow.parquet as pq

    settings = value["data"]
    validate_limits(settings)
    if not value["model"].get("fixture") and settings["codec_minimum_answer_positions"] < 10000000:
        raise ValueError("real extension requires at least 10M distinct codec training positions")
    if (
        settings["max_examples"] > 1000000
        or settings["max_unique_tokens"] > 100000000
        or sum(settings["split_token_caps"].values()) > settings["max_unique_tokens"]
        or settings["max_prompt_tokens"] > 512
        or settings["max_answer_tokens"] > value["architecture"]["canvas"]
        or len(settings["training_band_shares"]) != 5
        or any(w <= 0 for w in settings["training_band_shares"])
        or abs(sum(settings["training_band_shares"]) - 1) > 1e-9
    ):
        raise ValueError("invalid bounded extension admission/selection limits")
    assets = source_assets(settings)
    verified = []
    for asset in assets:
        path = asset_path(root() / settings["assets_root"], asset)
        sha = verify_asset(path, asset, run.meter)
        if not sha:
            raise ValueError("missing/corrupt pinned Tulu shard: " + asset["id"])
        verified.append(
            {
                "id": asset["id"],
                "sha256": sha,
                "revision": asset["revision"],
                "filename": asset["filename"],
                "bytes": asset["size_bytes"],
            }
        )
    tokenizer_files = tokenizer_binding(settings, value["model"], run.meter)
    tokenizer = load_tokenizer(value["model"])
    language = EnglishFilter(settings["language"])
    policy = read(root() / settings["source_policy"])
    if (
        policy.get("schema") != "tulu_component_policy_v1"
        or policy.get("dataset_revision") != settings["revision"]
    ):
        raise ValueError("versioned pinned component policy required")
    (run.path / "artifacts").mkdir(exist_ok=True)
    database = sqlite3.connect(run.path / "artifacts/preparation.sqlite")
    database.execute("PRAGMA cache_size=-4096")
    database.execute("PRAGMA temp_store=FILE")
    database.execute(
        "CREATE TABLE rows(key TEXT PRIMARY KEY,id TEXT,node TEXT,split TEXT,band INTEGER,source TEXT,rank TEXT,payload TEXT,tokens INTEGER,selected INTEGER)"
    )
    groups = Groups(database)
    excluded, inventory = Counter(), Counter()
    scan = {
        "complete": False,
        "rows_read": 0,
        "excluded": excluded,
        "source_inventory": inventory,
        "admission_order": ADMISSION_ORDER,
    }
    overlap = overlap_index(database, settings, tokenizer, run)
    normalized_settings = {
        k: v
        for k, v in settings.items()
        if k not in {"asset_lock", "assets_root", "source_asset_ids", "source_policy"}
    }
    preprocessing = canonical_hash(
        [
            normalized_settings,
            policy,
            language.identity,
            overlap,
            ADMISSION_ORDER,
            tokenizer_files,
            canonical_hash(tokenizer.chat_template),
            model_identity(value["model"]),
        ]
    )
    identity = {
        "model": model_identity(value["model"]),
        "dataset": DATASET,
        "source_revision": settings["revision"],
        "source_sha256": canonical_hash([verified, preprocessing]),
        "shards": verified,
        "template_sha256": canonical_hash(tokenizer.chat_template),
        "thinking": False,
        "canvas": value["architecture"]["canvas"],
        "latent_width": value["architecture"]["latent_width"],
        "license": "ODC-BY-1.0 collection; component terms apply",
        "preprocessing_sha256": preprocessing,
        "tokenizer_file_sha256": tokenizer_files,
        "language": language.identity,
        "source_policy_sha256": canonical_hash(policy),
        "overlap": overlap,
    }
    eos = tokenizer.eos_token_id
    special_ids = set(getattr(tokenizer, "all_special_ids", [eos])) - {
        getattr(tokenizer, "unk_token_id", None)
    }
    special_text = set(getattr(tokenizer, "all_special_tokens", [])) | {
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
    }
    total_rows = sum(
        pq.ParquetFile(asset_path(root() / settings["assets_root"], a)).metadata.num_rows for a in assets
    )
    try:
        for asset in assets:
            parquet = pq.ParquetFile(asset_path(root() / settings["assets_root"], asset))
            columns = [
                name
                for name in ("id", "messages", "source", "group_id", "conversation_id")
                if name in parquet.schema_arrow.names
            ]
            for batch in parquet.iter_batches(batch_size=settings["parquet_batch_rows"], columns=columns):
                run.meter.check_disk()
                for row in batch.to_pylist():
                    if scan["rows_read"] >= settings["max_source_rows"]:
                        break
                    scan["rows_read"] += 1
                    run.meter.charge(steps=1)
                    original, source = row.get("id"), row.get("source")
                    if not isinstance(original, str) or not original.strip() or len(original) > 512:
                        excluded["row_identity"] += 1
                        continue
                    if not isinstance(source, str) or source not in policy["components"]:
                        raise ValueError("unmapped Tulu component label: " + str(source))
                    inventory[source] += 1
                    component = policy["components"][source]
                    if not component["include"]:
                        excluded["source_policy:" + component["exclusion_reason"]] += 1
                        continue
                    try:
                        messages = validated_messages(
                            row.get("messages"),
                            final_assistant=True,
                            max_messages=settings["max_messages"],
                            max_chars=settings["max_row_chars"],
                        )
                    except ValueError as exc:
                        excluded[str(exc)] += 1
                        continue
                    messages = [{"role": m["role"], "content": canonical(m["content"])} for m in messages]
                    context, answer = messages[:-1], messages[-1]["content"]
                    if any(token in m["content"] for m in messages for token in special_text):
                        excluded["special_delimiter"] += 1
                        continue
                    reason = language.admit(context, answer)
                    if reason:
                        excluded[reason] += 1
                        continue
                    prompt = prompt_tokens(tokenizer, context)
                    target = tokenizer.encode(answer, add_special_tokens=False)
                    run.meter.charge(source=len(prompt) + len(target) + 1, answer=len(target) + 1)
                    if any(token in special_ids for token in target):
                        excluded["interior_eos_or_special"] += 1
                        continue
                    target.append(eos)  # Exactly one configured terminator; raw delimiters were rejected.
                    if len(prompt) > settings["max_prompt_tokens"]:
                        excluded["prompt_length"] += 1
                        continue
                    if len(target) > settings["max_answer_tokens"]:
                        excluded["answer_length"] += 1
                        continue
                    prompt_key = canonical_hash(prompt)
                    if database.execute("SELECT 1 FROM overlap WHERE prompt=?", (prompt_key,)).fetchone():
                        excluded["oasst_exact_prompt_or_reference"] += 1
                        continue
                    original_id = DATASET + ":" + source + ":" + original
                    node = "conversation:" + canonical_hash(
                        [DATASET, source, row.get("group_id") or row.get("conversation_id") or original]
                    )
                    groups.union(node, "prompt:" + prompt_key)
                    key = canonical_hash([context, answer])
                    existing = database.execute("SELECT id,node FROM rows WHERE key=?", (key,)).fetchone()
                    if existing:
                        excluded["canonical_duplicate"] += 1
                        groups.union(node, existing[1])
                        if original_id >= existing[0]:
                            continue
                    provenance = {
                        "model_id": value["model"]["model_id"],
                        "revision": value["model"]["revision"],
                        "tokenizer_revision": value["model"]["tokenizer_revision"],
                        "template_sha256": identity["template_sha256"],
                        "thinking": False,
                        "source_sha256": identity["source_sha256"],
                        "dataset_revision": settings["revision"],
                        "license": component["license"],
                        "component_source": source,
                        "answer_origin": component["answer_origin"],
                        "attribution": component["attribution"],
                        "component_terms": component["terms"],
                        "shard_sha256": asset["sha256"],
                        "original_row_id": original,
                        "target_selection": "final_assistant_complete_context_v1",
                        "preprocessing_sha256": identity["preprocessing_sha256"],
                        "eos_ids": [eos],
                    }
                    payload = {
                        "original_id": original_id,
                        "group_id": node,
                        "split": "train",
                        "source_type": "software_fixture" if value["model"].get("fixture") else "reference",
                        "prompt_ids": prompt,
                        "answer_ids": target,
                        "eos_genuine": True,
                        "censored": False,
                        "provenance": provenance,
                    }
                    database.execute(
                        "INSERT OR REPLACE INTO rows VALUES (?,?,?,?,?,?,?,?,?,NULL)",
                        (
                            key,
                            original_id,
                            node,
                            "",
                            band(len(target)),
                            source,
                            rank(original_id, settings["seed"]),
                            json.dumps(payload),
                            len(prompt) + len(target),
                        ),
                    )
                database.commit()
                if scan["rows_read"] >= settings["max_source_rows"]:
                    break
            if scan["rows_read"] >= settings["max_source_rows"]:
                break
        scan["complete"] = scan["rows_read"] == total_rows
        scan["release_rows"] = total_rows
        for key, node in database.execute("SELECT key,node FROM rows ORDER BY key"):
            run.meter.check()
            group = groups.find(node)
            split = (
                "dev"
                if int(rank(group, settings["seed"])[:16], 16) / 2**64 < settings["dev_fraction"]
                else "train"
            )
            database.execute("UPDATE rows SET node=?,split=? WHERE key=?", (group, split, key))
        database.execute("CREATE INDEX ranked ON rows(split,band,source,rank,key)")
        database.commit()
        scan["eligible"] = {
            s: {
                "examples": database.execute("SELECT count(*) FROM rows WHERE split=?", (s,)).fetchone()[0],
                "bands": [
                    database.execute("SELECT count(*) FROM rows WHERE split=? AND band=?", (s, b)).fetchone()[
                        0
                    ]
                    for b in range(5)
                ],
                "sources": dict(
                    database.execute("SELECT source,count(*) FROM rows WHERE split=? GROUP BY source", (s,))
                ),
            }
            for s in ("train", "dev")
        }
        scan["connected_groups"] = database.execute("SELECT count(DISTINCT node) FROM rows").fetchone()[0]
        for split, summary in scan["eligible"].items():
            query = "SELECT count(DISTINCT node),coalesce(sum(json_array_length(json_extract(payload,'$.prompt_ids'))),0),coalesce(sum(json_array_length(json_extract(payload,'$.answer_ids'))),0),coalesce(sum(tokens),0) FROM rows WHERE split=?"
            summary.update(
                zip(
                    ("groups", "prompt_tokens", "answer_positions", "combined_tokens"),
                    database.execute(query, (split,)).fetchone(),
                )
            )
            summary["origins"] = dict(
                database.execute(
                    "SELECT json_extract(payload,'$.provenance.answer_origin'),count(*) FROM rows WHERE split=? GROUP BY 1",
                    (split,),
                )
            )
        selected = 0
        for split in ("train", "dev"):
            used = 0
            for key in fair_rows(
                database, split, settings["training_band_shares"] if split == "train" else None
            ):
                run.meter.check()
                tokens = database.execute("SELECT tokens FROM rows WHERE key=?", (key,)).fetchone()[0]
                if used + tokens > settings["split_token_caps"][split]:
                    excluded["pool_token_remainder:" + split] += 1
                    continue
                if selected >= settings["max_examples"]:
                    break
                database.execute("UPDATE rows SET selected=? WHERE key=?", (selected, key))
                selected += 1
                used += tokens
        database.commit()
        if not selected:
            capacity_path = run.path / "artifacts/capacity.json"
            atomic_json(
                capacity_path,
                {
                    "schema": "native_capacity_v1",
                    "status": "blocked",
                    "data_manifest_sha256": None,
                    "scan": scan,
                    "blockers": ["zero admitted examples; capacity cannot qualify"],
                },
            )
            return {
                "status": "blocked",
                "capacity": reference(capacity_path),
                "qualification": "blocked",
                "blockers": ["zero admitted examples; no training artifact published"],
            }
        writer = ArtifactWriter(run.path / "artifacts/data", "native_data_v1", identity, run.meter)
        for payload, group, split in database.execute(
            "SELECT payload,node,split FROM rows WHERE selected IS NOT NULL ORDER BY selected"
        ):
            run.meter.check_disk()
            value_row = json.loads(payload)
            value_row.update(group_id=group, split=split)
            writer.append(Example(**value_row))
        manifest = writer.finish(
            {
                "admission_order": ADMISSION_ORDER,
                "capacity_file": "capacity.json",
                "settings": normalized_settings,
                "eos_policy": "reject_raw_specials_append_one_tokenizer_eos_v1",
            }
        )
        report, capacity_path = create_capacity(manifest, settings, scan, run.meter)
        if report["status"] == "qualified":
            run.publish(manifest)
        return {
            "status": "completed" if report["status"] == "qualified" else "blocked",
            "artifact": str(manifest),
            "capacity": reference(capacity_path),
            "stats": writer.stats,
            "qualification": report["status"],
            "blockers": report["blockers"],
        }
    finally:
        atomic_json(run.path / "artifacts/scan.json", scan)
        database.close()
