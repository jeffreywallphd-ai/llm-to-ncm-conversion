"""Complete references, dedup-connected group splits and sealed locked evaluation."""

from collections import Counter, defaultdict, deque
from contextlib import closing
from dataclasses import asdict
import hashlib
import unicodedata
from ncm_lab.data.prepare import raw_examples, validate_source_binding, DataCapacityError
from ncm_lab.data.records import Example, LOCKED_SCHEMA
from ncm_lab.data.artifacts import ArtifactWriter, LOCKED
from ncm_lab.data.tokenization import prompt_tokens
from ncm_lab.runtime.io import root, digest, canonical_hash, write_json, reference
from ncm_lab.teacher.model import load_tokenizer, model_identity
from ncm_lab.stage1.config import DataSettings, Stage2DataSettings, ACQUIRE


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def stratified(rows, count, seed):
    bins = defaultdict(list)
    for row in rows:
        bins[(len(row.answer_ids) - 1).bit_length()].append(row)
    queues = [
        deque(sorted(items, key=lambda e: canonical_hash([seed, e.original_id])))
        for _, items in sorted(bins.items())
    ]
    selected = []
    while queues and len(selected) < count:
        for queue in queues[:]:
            if queue:
                selected.append(queue.popleft())
            if not queue:
                queues.remove(queue)
            if len(selected) == count:
                break
    return selected


def partition(group, policy):
    fraction = int(hashlib.sha256(f"{policy.split_seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    if fraction < policy.test_fraction:
        return "test"
    if fraction < policy.test_fraction + policy.dev_fraction:
        return "dev"
    return "train"


def example_stats(items):
    return {
        "examples": len(items),
        "unique_answer_positions": sum(len(e.answer_ids) for e in items),
        "prompt_positions": sum(len(e.prompt_ids) for e in items),
        "length_histogram": dict(Counter(str(len(e.answer_ids)) for e in items)),
        "ids": [e.original_id for e in items],
        "groups": sorted({e.group_id for e in items}),
    }


def prepare(run, value):
    settings = value["data"]
    stage2 = value.get("stage1_context", {}).get("experiment_kind") == "stage2"
    policy = (Stage2DataSettings if stage2 else DataSettings)(**value["stage1_data"])
    if stage2 and settings.get("format") == "tulu_parquet":
        from ncm_lab.data.sources.tulu import prepare_tulu

        return prepare_tulu(run, value, stage2_policy=policy)
    if settings.get("format", "jsonl") not in {"jsonl", "oasst_parquet"}:
        raise ValueError(
            "Stage 1 uses the approved OASST or explicit approved JSONL source; no automatic Tulu substitution"
        )
    validate_source_binding(settings)
    source_path = root() / settings["source_path"]
    if not source_path.is_file():
        raise DataCapacityError(
            f"missing approved local source {source_path}; explicit sanctioned acquisition: {ACQUIRE}"
        )
    source_sha = digest(source_path, run.meter.check)
    if not settings.get("source_sha256") or source_sha != settings["source_sha256"]:
        raise ValueError("Stage 1 source SHA256 must match the explicitly approved source")
    rows, parents, duplicate_owner, ids = [], {}, {}, set()
    observations = {"source_rows_inspected": 0}
    excluded = Counter()
    size = 0

    def find(group):
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]

    def union(first, second):
        a, b = find(first), find(second)
        parents[max(a, b)] = min(a, b)

    with closing(
        raw_examples(settings, run.path / "artifacts/source-index.sqlite", run.meter.check_disk, observations)
    ) as source:
        for row in source:
            run.meter.check_disk()
            if row["original_id"] in ids:
                raise ValueError("ambiguous duplicated source example ID")
            ids.add(row["original_id"])
            if not isinstance(row["group_id"], str) or not row["group_id"]:
                raise ValueError("source parent group ID required")
            group = row["group_id"]
            find(group)
            for other in row.get("overlap_group_ids", []):
                union(group, other)
            key = canonical_hash([normalized(row["prompt"]), normalized(row["answer"])])
            if key in duplicate_owner:
                union(group, duplicate_owner[key])
                excluded["duplicate"] += 1
            else:
                duplicate_owner[key] = group
                rows.append((key, row))
            size += len(str(row).encode("utf-8"))
            if size > policy.max_candidate_bytes:
                raise DataCapacityError(
                    "bounded source candidate storage exhausted; explicitly revise approved scan settings"
                )
    try:
        tokenizer = load_tokenizer(value["model"])
    except (OSError, ValueError) as exc:
        raise DataCapacityError(
            f"missing/invalid pinned local tokenizer; explicit sanctioned acquisition: {ACQUIRE}; {exc}"
        ) from exc
    template = canonical_hash(tokenizer.chat_template)
    pools = {s: [] for s in ("train", "dev", "test")}
    assignments = {}
    for group in sorted(parents):
        canonical = find(group)
        assignments[group] = {"parent_group": canonical, "split": partition(canonical, policy)}
    for _, row in sorted(rows, key=lambda pair: pair[1]["original_id"]):
        prompt = prompt_tokens(tokenizer, row["prompt"])
        content = tokenizer.encode(row["answer"], add_special_tokens=False)
        run.meter.charge(source=len(prompt) + len(content) + 1, steps=1)
        eos = tokenizer.eos_token_id
        if eos in content:
            if content.index(eos) != len(content) - 1:
                excluded["embedded_eos"] += 1
                continue
            content = content[:-1]
        answer = content + [eos]
        if len(answer) > value["architecture"]["canvas"]:
            excluded["overlength"] += 1
            continue
        if len(prompt) > settings["max_prompt_tokens"]:
            excluded["prompt_length"] += 1
            continue
        membership = assignments[row["group_id"]]
        split = membership["split"]
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
            "original_parent_group": row["group_id"],
            "prompt_text": row["prompt"],
            "reference_text": row["answer"],
            "objective_rule": row.get("objective_rule"),
        }
        pools[split].append(
            Example(
                row["original_id"],
                membership["parent_group"],
                split,
                "software_fixture" if value["model"].get("fixture") else "reference",
                prompt,
                answer,
                True,
                False,
                provenance,
                schema=LOCKED_SCHEMA if split == "test" else "native_answer_example_v1",
            )
        )
    reserved_groups = set()
    if stage2 and settings.get("format") == "oasst_parquet":
        # Keep the group partition stable and reserve the standard Stage 1 test groups.
        reserved_groups = {e.group_id for e in stratified(pools["test"], 200, 20261029)}
        pools["test"] = [e for e in pools["test"] if e.group_id not in reserved_groups]
        excluded["stage1_test_groups"] = len(reserved_groups)
    selected = {
        s: stratified(items, getattr(policy, s + "_examples"), policy.validation_seed)
        for s, items in pools.items()
    }
    report = {
        **observations,
        "source_size_bytes": source_path.stat().st_size,
        "scan_cap_rows": settings["max_source_rows"],
        "source_sha256": source_sha,
        "template_sha256": template,
        "canvas": value["architecture"]["canvas"],
        "excluded": dict(excluded),
        "group_assignments": assignments,
        "natural_eligible_pool": {s: example_stats(items) for s, items in pools.items()},
        "common_8_position_eligible_subset": {
            s: example_stats([e for e in items if len(e.answer_ids) <= 8]) for s, items in pools.items()
        },
        "common_subset_note": "Report matched IDs when comparing sizes; differing natural eligibility pools do not isolate architectural effects.",
    }
    if stage2:
        report["reserved_stage1_test_groups"] = sorted(reserved_groups)
    identity = {
        "model": model_identity(value["model"]),
        "template_sha256": template,
        "source_sha256": source_sha,
        "canvas": value["architecture"]["canvas"],
        "thinking": False,
    }
    return publish_qualified(run, policy, selected, report, identity)


def publish_qualified(run, policy, selected, report, identity):
    stage2 = isinstance(policy, Stage2DataSettings)
    blockers = [
        f"{split}: need {getattr(policy, split + '_examples')} distinct complete pairs, available "
        f"{report['natural_eligible_pool'][split]['examples']}"
        for split, examples in selected.items()
        if len(examples) < getattr(policy, split + "_examples")
    ]
    report = {
        "schema": "stage2_data_qualification_v1" if stage2 else "stage1_data_qualification_v1",
        "policy": asdict(policy),
        **report,
        "selected": {split: example_stats(items) for split, items in selected.items()},
        "status": "blocked" if blockers else "qualified",
        "blockers": blockers,
        "diagnostic": policy.diagnostic,
        "distinct_counts_exclude_repeated_presentations": True,
    }
    qualification = run.path / "qualification.json"
    write_json(qualification, report, exclusive=True)
    if blockers:
        action = (
            "Use a new experiment with the explicit alternate --source, after its assets are available."
            if stage2
            else "Supply an explicit approved --data-config during new planning; no quota/canvas changes are automatic."
        )
        raise DataCapacityError(
            f"Stage {'2' if stage2 else '1'} data capacity blocked before teacher allocation: "
            + "; ".join(blockers)
            + f". Report: {qualification}. {action}"
        )
    identity = {
        **identity,
        "stage1_split_policy": asdict(policy),
        "qualification_sha256": digest(qualification),
    }
    if stage2:
        identity["qualification_mode"] = "stage2_locked_short_answers_v1"
    identity["source_contract_sha256"] = canonical_hash(
        {**identity, "selected_ids": {s: [e.original_id for e in items] for s, items in selected.items()}}
    )
    locked = ArtifactWriter(
        run.path / "artifacts/evaluation", LOCKED, {**identity, "partition": "locked_test"}, run.meter
    )
    for example in selected["test"]:
        locked.append(example)
    evaluation = locked.finish({"never_used_for_training_normalization_or_selection": True})
    writer = ArtifactWriter(run.path / "artifacts/data", "native_data_v1", identity, run.meter)
    for example in selected["train"] + selected["dev"]:
        writer.append(example)
    artifact = writer.finish(
        {"qualification": reference(qualification), "locked_evaluation": reference(evaluation)}
    )
    run.publish(artifact)
    return {
        "artifact": str(artifact),
        "qualification": reference(qualification),
        "evaluation_data": reference(evaluation),
        "stats": writer.stats,
        "diagnostic": policy.diagnostic,
        "model_loaded": False,
    }
