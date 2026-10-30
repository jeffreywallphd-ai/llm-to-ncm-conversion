"""Shared pre-teacher qualification and deterministic whole-example codec subset."""

from collections import Counter
from pathlib import Path
import sqlite3
import tempfile
from ncm_lab.data.artifacts import Dataset
from ncm_lab.data.selection import band, fair_rows, rank, SHARES
from ncm_lab.runtime.io import canonical_hash, read, digest, write_json, reference


def codec_selection(data, minimum, max_dev=256, check=lambda: None):
    with tempfile.TemporaryDirectory(prefix="ncm-qualification-") as directory:
        database = sqlite3.connect(Path(directory) / "selection.sqlite")
        try:
            database.execute("PRAGMA cache_size=-4096")
            database.execute(
                "CREATE TABLE rows(key INTEGER PRIMARY KEY,id TEXT,split TEXT,band INTEGER,source TEXT,rank TEXT,length INTEGER)"
            )
            for index, row in data.iter_metadata():
                check()
                example = row["example"]
                database.execute(
                    "INSERT INTO rows VALUES (?,?,?,?,?,?,?)",
                    (
                        index,
                        example["original_id"],
                        example["split"],
                        band(len(example["answer_ids"])),
                        example["provenance"].get("component_source", "legacy"),
                        rank(example["original_id"]),
                        len(example["answer_ids"]),
                    ),
                )
            database.execute("CREATE INDEX ranked ON rows(split,band,source,rank,key)")
            database.commit()
            available = database.execute(
                "SELECT coalesce(sum(length),0) FROM rows WHERE split='train'"
            ).fetchone()[0]
            train, dev, positions = [], [], 0
            for index in fair_rows(database, "train", SHARES):
                check()
                if minimum and positions >= minimum:
                    break
                train.append(index)
                positions += database.execute("SELECT length FROM rows WHERE key=?", (index,)).fetchone()[0]
            for index in fair_rows(database, "dev", coverage=True):
                if len(dev) >= max_dev:
                    break
                dev.append(index)

            def ids(indices):
                return [
                    database.execute("SELECT id FROM rows WHERE key=?", (i,)).fetchone()[0] for i in indices
                ]

            result = {
                "policy": "source_length_whole_example_codec_v1",
                "minimum_training_answer_positions": minimum,
                "available_training_answer_positions": available,
                "selected_training_answer_positions": positions,
                "train_ids": ids(train),
                "dev_ids": ids(dev),
                "train_indices": train,
                "dev_indices": dev,
            }
            result["order_sha256"] = canonical_hash([result["train_ids"], result["dev_ids"]])
            result["whole_example_overshoot"] = max(0, positions - minimum) if minimum else 0
            result["maximum_whole_example_overshoot"] = data.manifest["identity"]["canvas"] - 1
            return result
        finally:
            database.close()


def distribution(data):
    result = {}
    for split in ("train", "dev"):
        result[split] = dict(
            examples=0,
            groups=set(),
            prompt_tokens=0,
            answer_positions=0,
            combined_tokens=0,
            sources=Counter(),
            origins=Counter(),
            bands=[0] * 5,
        )
    for _, row in data.iter_metadata():
        example = row["example"]
        value = result[example["split"]]
        value["examples"] += 1
        value["groups"].add(example["group_id"])
        value["prompt_tokens"] += len(example["prompt_ids"])
        value["answer_positions"] += len(example["answer_ids"])
        value["combined_tokens"] += len(example["prompt_ids"]) + len(example["answer_ids"])
        value["sources"][example["provenance"].get("component_source", "legacy")] += 1
        value["origins"][example["provenance"].get("answer_origin", "unknown")] += 1
        value["bands"][band(len(example["answer_ids"]))] += 1
    for value in result.values():
        value["groups"] = len(value["groups"])
        value["achieved_band_shares"] = [n / max(1, value["examples"]) for n in value["bands"]]
    return result


def create_capacity(manifest, settings, scan, meter):
    data = Dataset(manifest, "native_data_v1")
    selection = codec_selection(
        data, settings["codec_minimum_answer_positions"], settings["codec_dev_examples"], meter.check
    )
    selected = distribution(data)
    blockers = []
    if not scan["complete"]:
        blockers.append("input scan is incomplete; qualification requires the declared complete scan")
    if selection["available_training_answer_positions"] < settings["codec_minimum_answer_positions"]:
        blockers.append(
            "insufficient distinct training answer positions; dev and repeated epochs cannot fill the quota"
        )
    if not selection["dev_ids"]:
        blockers.append("no held-out codec development examples")
    hidden = 4096 if not data.manifest["identity"]["model"].get("fixture") else 16
    latent = data.manifest["identity"].get("latent_width", 384)
    codec_dev_positions = sum(
        len(data.metadata(i)["example"]["answer_ids"]) for i in selection["dev_indices"]
    )
    report = {
        "schema": "native_capacity_v1",
        "data_manifest_sha256": data.sha256,
        "preprocessing_sha256": data.manifest["identity"]["preprocessing_sha256"],
        "identity": data.manifest["identity"],
        "scan": scan,
        "selected": selected,
        "codec_selection": selection,
        "codec_validation_examples_evaluated": min(
            len(selection["dev_ids"]), settings["codec_validation_examples"]
        ),
        "training_band_target_shares": settings["training_band_shares"],
        "training_band_example_shortfalls": [
            max(0, round(selected["train"]["examples"] * share) - actual)
            for share, actual in zip(settings["training_band_shares"], selected["train"]["bands"])
        ],
        "generator_processed_token_objective": settings["generator_processed_tokens"],
        "generator_complete_pass_equivalent": settings["generator_processed_tokens"]
        / max(1, selected["train"]["combined_tokens"]),
        "pool_token_caps": settings["split_token_caps"],
        "unfilled_tokens": {
            s: settings["split_token_caps"][s] - selected[s]["combined_tokens"] for s in selected
        },
        "storage_estimates_bytes": {
            "teacher_bf16": (selection["selected_training_answer_positions"] + codec_dev_positions)
            * hidden
            * 2,
            "fine_codes_bf16": sum(v["answer_positions"] for v in selected.values()) * latent * 2,
            "metadata_index_allowance": settings["max_examples"] * 16384,
            "generator_four_recovery_versions": 4 * 8250600480 * 10,
            "generator_export": 8250600480 * 2,
            "note": "Estimates only; include retained/pending artifacts plus separately configured headroom.",
        },
        "status": "blocked" if blockers else "qualified",
        "blockers": blockers,
        "warnings": [
            "English classification and exact overlap are heuristics; paraphrases/benchmark decontamination are not established."
        ],
        "unique_definition": "Distinct admitted prepared-example positions; repeated epochs/dev are excluded from training quota.",
    }
    if not selected["train"]["bands"][-1]:
        report["warnings"].append("ZERO TRAINING COVERAGE in the 769-1024 answer-position band")
    path = Path(manifest).parent / "capacity.json"
    write_json(path, report, exclusive=True)
    write_json(path.with_suffix(".ref.json"), reference(path), exclusive=True)
    summary = (
        f"Capacity {report['status']}: {selected['train']['examples']} train examples, "
        f"{selection['available_training_answer_positions']} distinct train answer positions; "
        f"minimum {settings['codec_minimum_answer_positions']}; dev {selected['dev']['examples']}.\n"
    )
    path.with_suffix(".txt").write_text(summary, encoding="utf-8")
    return report, path


def qualified_selection(data, minimum, max_dev=256, check=lambda: None):
    """Recompute the committed subset before any teacher allocation, including standalone use."""
    selected = codec_selection(data, minimum, max_dev, check)
    if selected["available_training_answer_positions"] < minimum:
        raise ValueError(
            f"capacity blocked before teacher allocation: {selected['available_training_answer_positions']} < {minimum} training answer positions"
        )
    if data.manifest["identity"].get("dataset") == "allenai/tulu-3-sft-mixture":
        path = data.path / "capacity.json"
        ref = read(data.path / "capacity.ref.json")
        if digest(path) != ref["sha256"]:
            raise ValueError("capacity report changed")
        report = read(path)
        if (
            report["data_manifest_sha256"] != data.sha256
            or report["preprocessing_sha256"] != data.manifest["identity"]["preprocessing_sha256"]
            or report["codec_selection"] != selected
            or report["status"] != "qualified"
        ):
            raise ValueError("capacity/data/selection mismatch or unqualified source")
    return selected
