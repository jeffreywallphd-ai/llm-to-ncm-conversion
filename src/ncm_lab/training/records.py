import json
from pathlib import Path
import torch


def normalize_record(row, split="train", origin="target_distillation"):
    row = dict(row)
    if origin not in {"target_distillation", "software_fixture"}:
        raise ValueError("only target continuations or explicit software fixtures are training records")
    if row.get("split") != split or row.get("origin") != origin:
        raise ValueError(f"expected {split}-only {origin} records")
    ids = row.get("input_ids")
    if row.get("policy_hash"):
        from ncm_lab.training.source_contracts import policy_record

        if policy_record(row["generation_policy"])[1] != row["policy_hash"]:
            raise ValueError("invalid recorded canonical teacher policy hash")
    if not row.get("group_id") or not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise ValueError("missing source/group identity or token IDs")
    if "prompt_length" not in row:
        if "generated_tokens" not in row:
            if origin != "software_fixture":
                raise ValueError("ambiguous prompt/response boundary")
            row["prompt_length"] = 1
        else:
            row["prompt_length"] = len(ids) - row["generated_tokens"]
            row["boundary_migration"] = "derived_from_trusted_generated_tokens"
    p = row["prompt_length"]
    if not isinstance(p, int) or not 0 < p <= len(ids):
        raise ValueError("invalid prompt/response boundary")
    if row.get("generated_tokens", len(ids) - p) != len(ids) - p:
        raise ValueError("inconsistent completion length")
    mask = row.get("response_mask", [i >= p for i in range(len(ids))])
    if len(mask) != len(ids) or any(mask[:p]):
        raise ValueError("response mask exposes prompt labels")
    eos = set(row.get("generation_policy", {}).get("eos_ids", []))
    terminal = next((i for i in range(p, len(ids)) if ids[i] in eos), len(ids))
    if any(mask[terminal + 1 :]):
        raise ValueError("response mask includes post-terminal tokens")
    row["response_mask"] = mask
    return row


def load_records(path, split="train", origin="target_distillation", allow_fixture=False):
    with Path(path).open(encoding="utf-8") as stream:
        raw = [json.loads(line) for line in stream if line.strip()]
        rows = [
            normalize_record(row, split, row.get("origin", origin) if allow_fixture else origin)
            for row in raw
        ]
    if not rows:
        raise ValueError("empty corpus")
    return rows


def anchors(row):
    return [
        a
        for a in range(row["prompt_length"], len(row["input_ids"]) - 1)
        if row["response_mask"][a] and row["response_mask"][a + 1]
    ]


def block_labels(row, anchor, horizon=15, device="cpu"):
    labels = torch.full((1, horizon), -100, dtype=torch.long, device=device)
    for j in range(horizon):
        position = anchor + j + 1
        if position < len(row["input_ids"]) and row["response_mask"][position]:
            labels[0, j] = row["input_ids"][position]
    return labels


def disjoint(train, dev):
    if not dev or any(r["split"] != "dev" for r in dev):
        raise ValueError("selection requires held-out development records")
    if {r["group_id"] for r in train} & {r["group_id"] for r in dev}:
        raise ValueError("train/development group overlap")


def admitted_hash(rows):
    from ncm_lab.runtime.io import canonical_hash

    return canonical_hash([{k: v for k, v in row.items() if k != "legacy_source_migrated"} for row in rows])


def subset_for_tier(rows, config):
    if config.get("overlap_audit"):
        from ncm_lab.data.overlap import validate_audit

        audit = validate_audit(config["overlap_audit"])
        if any(r.get("original_id") not in audit["training_original_ids"] for r in rows):
            raise ValueError("teacher data includes unaudited source prompts")
    elif config.get("research_scope") == "locked_study":
        raise ValueError("locked adaptation requires the declared benchmark overlap audit")
    if not config.get("data_tier"):
        return rows
    from ncm_lab.data.ladder import resolve_ladder

    selected = resolve_ladder(rows, [config["data_tier"]], config.get("data_seed", 0))[0]
    if selected["status"] != "available":
        raise ValueError("requested distinct-response tier unavailable; repeated tokens cannot fill it")
    admitted = set(selected["original_ids"])
    return [r for r in rows if r["original_id"] in admitted]
