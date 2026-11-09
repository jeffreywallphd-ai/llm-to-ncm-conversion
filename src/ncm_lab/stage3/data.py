"""Disk-backed candidates, durable one-candidate distillation, and final qualification."""

from collections import Counter
from contextlib import closing
from dataclasses import asdict, replace
import json
import sqlite3
import sys
import torch
from ncm_lab.data.artifacts import ArtifactWriter, Dataset, LOCKED
from ncm_lab.data.records import Example, LOCKED_SCHEMA
from ncm_lab.data.prepare import raw_examples, validate_source_binding, DataCapacityError
from ncm_lab.data.tokenization import prompt_tokens
from ncm_lab.runtime.io import root, read, digest, canonical_hash, reference, write_json
from ncm_lab.stage1.data import partition, normalized
from ncm_lab.stage3.config import Qualification
from ncm_lab.stage3.storage import charge, check_cache
from ncm_lab.teacher.model import load_tokenizer, load_original, model_identity

CANDIDATES = "stage3_candidates_v1"
DISTILLED = "stage3_distilled_v1"


def identity(value, tokenizer, source_sha):
    return dict(
        model=model_identity(value["model"]),
        template_sha256=canonical_hash(tokenizer.chat_template),
        source_sha256=source_sha,
        canvas=value["architecture"]["canvas"],
        thinking=False,
        qualification_mode="stage3_qualified_v1",
        source_index_policy=value["stage3"]["source_index"],
    )


def source(run, value):
    from ncm_lab.stage3.source_index import SourceStorage

    storage = SourceStorage(run)
    storage.preflight()
    run.source_storage = storage
    try:
        result = _source(run, value)
    except sqlite3.Error as exc:
        storage.sqlite_failure(exc)
    result["source_storage"] = storage.finish()
    return result


def _source(run, value):
    policy = Qualification(**value["stage1_data"])
    if value["data"].get("format") == "tulu_parquet":
        from ncm_lab.data.sources.tulu import prepare_tulu

        return prepare_tulu(run, value, stage2_policy=policy)
    if not value["model"].get("fixture"):
        raise ValueError("JSONL source is supported only for explicit software fixtures")
    from ncm_lab.data.sources.tulu import Groups

    settings = value["data"]
    validate_source_binding(settings)
    sha = digest(root() / settings["source_path"], run.meter.check)
    if sha != settings["source_sha256"]:
        raise ValueError("source content changed")
    tokenizer = load_tokenizer(value["model"])
    folder = run.path / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(folder / "candidates.sqlite")) as db:
        run.source_storage.configure(db)
        db.execute(
            "CREATE TABLE rows(key TEXT PRIMARY KEY,id TEXT,node TEXT,split TEXT,rank TEXT,payload TEXT)"
        )
        groups = Groups(db)
        with closing(raw_examples(settings, folder / "source.sqlite", run.meter.check_disk)) as rows:
            for row in rows:
                prompt = prompt_tokens(tokenizer, row["prompt"])
                answer = tokenizer.encode(row["answer"], add_special_tokens=False)
                run.meter.charge(steps=1, source=len(prompt) + len(answer) + 1)
                if len(prompt) > settings["max_prompt_tokens"] or tokenizer.eos_token_id in answer:
                    continue
                key = canonical_hash(normalized(row["prompt"]))
                node = row["group_id"]
                groups.union(node, "prompt:" + key)
                groups.union(node, "rendered_prompt:" + canonical_hash(prompt))
                provenance = dict(
                    model_id=value["model"]["model_id"],
                    revision=value["model"]["revision"],
                    tokenizer_revision=value["model"]["tokenizer_revision"],
                    template_sha256=canonical_hash(tokenizer.chat_template),
                    thinking=False,
                    source_sha256=sha,
                    dataset_revision=settings["revision"],
                    license=settings["license"],
                    eos_ids=[tokenizer.eos_token_id],
                    reference_text=row["answer"],
                    prompt_text=row["prompt"],
                    target_origin="software_fixture",
                    objective_rule=row.get("objective_rule"),
                )
                example = Example(
                    row["original_id"],
                    node,
                    "train",
                    "software_fixture",
                    prompt,
                    answer + [tokenizer.eos_token_id],
                    True,
                    False,
                    provenance,
                )
                encoded, raw_bytes = run.source_storage.policy.encode(asdict(example))
                charge(run, raw_bytes + 16384)
                db.execute(
                    "INSERT OR IGNORE INTO rows VALUES(?,?,?,?,?,?)",
                    (
                        key,
                        example.original_id,
                        node,
                        "",
                        canonical_hash([policy.validation_seed, example.original_id]),
                        encoded,
                    ),
                )
                run.source_storage.check(phase="scan")
        for key, node in db.execute("SELECT key,node FROM rows"):
            group = groups.find(node)
            db.execute("UPDATE rows SET node=?,split=? WHERE key=?", (group, partition(group, policy), key))
        db.commit()
        return publish_candidates(run, db, policy, {}, identity(value, tokenizer, sha))


def publish_candidates(run, db, policy, scan, source_identity):
    """Source qualification never claims final teacher-target capacity."""
    value = run.value
    cap = value["stage3"]["distillation"]["max_candidates"]
    prior_groups, prior_prompts = set(), set()
    previous = []
    for ref in value["stage3"]["previous_locked_manifests"]:
        from ncm_lab.stage1.workflow import checked

        data = Dataset(checked(ref), LOCKED)
        previous.append(ref)
        for _, row in data.iter_metadata():
            prior_groups.add(row["example"]["group_id"])
            prior_prompts.add(canonical_hash(row["example"]["prompt_ids"]))
    writer = ArtifactWriter(run.path / "artifacts/candidates", CANDIDATES, source_identity, run.meter)
    locked = ArtifactWriter(run.path / "artifacts/locked", LOCKED, source_identity, run.meter)
    # Unique prompt selection is deterministic and independent of original answer length.
    db.execute("CREATE TABLE IF NOT EXISTS selected_prompts(key TEXT PRIMARY KEY)")
    counts = Counter()
    for payload, group, split in db.execute("SELECT payload,node,split FROM rows ORDER BY rank,key"):
        example = Example(**run.source_storage.policy.decode(payload))
        prompt_hash = canonical_hash(example.prompt_ids)
        if group in prior_groups or prompt_hash in prior_prompts:
            continue
        if db.execute("SELECT 1 FROM selected_prompts WHERE key=?", (prompt_hash,)).fetchone():
            continue
        example = replace(
            example,
            group_id=group,
            split=split,
            schema=LOCKED_SCHEMA if split == "test" else "native_answer_example_v1",
        )
        if split == "test":
            if counts["test"] >= policy.test_examples or len(example.answer_ids) > source_identity["canvas"]:
                continue
            locked.append(example)
        else:
            if counts["train"] + counts["dev"] >= cap:
                continue
            writer.append(example)
        counts[split] += 1
        db.execute("INSERT INTO selected_prompts VALUES(?)", (prompt_hash,))
        run.meter.check_disk()
        run.source_storage.check(phase="publish_candidates", publish=True)
    if counts["test"] < policy.test_examples:
        raise DataCapacityError(
            f"locked original-reference capacity: need {policy.test_examples}, found {counts['test']}"
        )
    evaluation = locked.finish()
    artifact = writer.finish({"locked_evaluation": reference(evaluation)})
    report = dict(
        schema="stage3_source_capacity_v1",
        candidates=dict(counts),
        scan=scan,
        final_target_capacity="pending_distillation",
        model_loaded=False,
        previous_locked=previous,
        cross_experiment_scope="only explicitly supplied verified manifests; broader independence unverified",
    )
    report_path = run.path / "qualification.json"
    write_json(report_path, report, exclusive=True)
    run.publish(artifact)
    return dict(
        artifact=str(artifact),
        evaluation_data=reference(evaluation),
        qualification=reference(report_path),
        stats=writer.stats,
        model_loaded=False,
        final_target_capacity="pending_distillation",
    )


def generation_policy(model, tokenizer, canvas, eos_ids):
    """Neutralize inherited length/forced termination processors explicitly."""
    from transformers import GenerationConfig

    cfg = GenerationConfig()
    overrides = dict(
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=canvas,
        min_length=0,
        min_new_tokens=0,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        max_time=None,
        use_cache=True,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        bad_words_ids=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        forced_decoder_ids=None,
        stop_strings=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def acceptance(tokens, tokenizer, eos_ids, canvas, threshold):
    if not tokens or len(tokens) > canvas or tokens[-1] not in eos_ids:
        return "no_natural_eos"
    if any(t in eos_ids for t in tokens[:-1]):
        return "interior_eos"
    if any(type(t) is not int or t < 0 or t >= len(tokenizer) for t in tokens):
        return "invalid_token"
    content = tokenizer.decode(tokens, skip_special_tokens=True)
    if not content.strip():
        return "empty_content"
    special = set(tokenizer.all_special_ids) - set(eos_ids)
    if any(t in special for t in tokens[:-1]):
        return "internal_special_token"
    body = tokens[:-1]
    if len(body) >= 16 and 1 - len(set(body)) / len(body) >= threshold:
        return "degenerate_repetition"
    return None


def distill(run, value, bound):
    data = Dataset(bound["source"]["manifest"], CANDIDATES)
    tokenizer = load_tokenizer(value["model"])
    target_policy = value["stage3"]["target_policy"]
    teacher = (
        load_original(value["model"], value["device"], getattr(torch, value["dtype"]), frozen=True)
        if target_policy == "teacher-consistent"
        else None
    )
    eos = model_identity(value["model"])["eos_ids"]
    canvas = value["architecture"]["canvas"]
    cfg = generation_policy(teacher, tokenizer, canvas, eos) if teacher is not None else None
    contract = dict(
        source=bound["source"],
        teacher=model_identity(value["model"]),
        policy=value["stage3"],
        generation=cfg.to_dict() if cfg else None,
        canvas=canvas,
    )
    cache = __import__("pathlib").Path(value["stage1_context"]["experiment"]).parent / "distillation"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / "committed.sqlite"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS contract(hash TEXT PRIMARY KEY)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,input_hash TEXT,attempts INTEGER,payload TEXT,hash TEXT)"
        )
        key = canonical_hash(contract)
        existing = db.execute("SELECT hash FROM contract").fetchone()
        if existing and existing[0] != key:
            raise ValueError("distillation cache policy/source changed")
        db.execute("INSERT OR IGNORE INTO contract VALUES(?)", (key,))
        db.commit()
        writer = ArtifactWriter(
            run.path / "artifacts/distilled",
            DISTILLED,
            {**data.manifest["identity"], "distillation_sha256": key},
            run.meter,
        )
        reasons, origins = Counter(), Counter()
        completed = 0
        admitted = Counter()
        quotas = value["stage1_data"]
        for i in range(len(data)):
            example, _ = data[i]
            if example.split == "test":
                raise ValueError("locked test cannot enter distillation")
            prior = db.execute(
                "SELECT input_hash,attempts,payload,hash FROM records WHERE id=?", (example.original_id,)
            ).fetchone()
            if prior and prior[2] is not None:
                record = json.loads(prior[2])
                if prior[0] != example.identity or canonical_hash(record) != prior[3]:
                    raise ValueError("committed distillation record changed")
            else:
                attempts = prior[1] if prior else 0
                if attempts >= value["stage3"]["distillation"]["max_record_attempts"]:
                    raise ValueError("distillation per-record attempt cap exhausted")
                charge(run, 16384)
                check_cache(run, cache)
                db.execute(
                    "INSERT OR REPLACE INTO records VALUES(?,?,?,NULL,NULL)",
                    (example.original_id, example.identity, attempts + 1),
                )
                db.commit()  # Durable attempted work precedes generation.
                run.meter.charge(
                    steps=1, source=canvas * (len(example.prompt_ids) + canvas), answer=canvas, canvas=canvas
                )
                trusted = (
                    example.provenance.get("authoritative_reference") is True
                    and example.provenance.get("objective_rule") == "normalized_exact"
                    and example.provenance.get("component_source")
                    in value["stage3"]["distillation"]["authoritative_components"]
                )
                if teacher is not None and not trusted:
                    with torch.inference_mode():
                        prompt = torch.tensor([example.prompt_ids], device=value["device"])
                        output = teacher.generate(
                            input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=cfg
                        )
                        tokens = output[0, len(example.prompt_ids) :].tolist()
                    origin = "teacher_generated"
                else:
                    tokens = example.answer_ids
                    origin = "reference"
                reason = acceptance(
                    tokens, tokenizer, eos, canvas, value["stage3"]["distillation"]["repetition_threshold"]
                )
                record = dict(
                    source_id=example.original_id,
                    input_hash=example.identity,
                    prompt_sha256=canonical_hash(example.prompt_ids),
                    policy_sha256=key,
                    raw_generated_ids=tokens,
                    target_origin=origin,
                    rejection=reason,
                    termination="natural_eos" if tokens and tokens[-1] in eos else "length_cap",
                    correctness="reference_rule_supported" if trusted else "unknown",
                    teacher=contract["teacher"],
                )
                encoded = json.dumps(record, allow_nan=False)
                run.meter.check_disk(len(encoded) + 65536)
                charge(run, len(encoded.encode()) + 16384)
                db.execute(
                    "UPDATE records SET payload=?,hash=? WHERE id=?",
                    (encoded, canonical_hash(record), example.original_id),
                )
                db.commit()
                check_cache(run, cache)
            reasons[record["rejection"] or "accepted"] += 1
            if not record["rejection"]:
                provenance = {
                    **example.provenance,
                    "target_origin": record["target_origin"],
                    "target_policy_sha256": key,
                    "distillation_record_sha256": canonical_hash(record),
                    "correctness": record["correctness"],
                }
                target = replace(
                    example,
                    answer_ids=record["raw_generated_ids"],
                    provenance=provenance,
                    source_type="software_fixture"
                    if value["model"].get("fixture")
                    else record["target_origin"],
                )
                writer.append(target)
                origins[record["target_origin"]] += 1
                b = 0 if len(target.answer_ids) <= 64 else 1 if len(target.answer_ids) <= 128 else 2
                admitted[(target.split, b)] += 1
                admitted[target.split] += 1
            completed += 1
            shares = [int(quotas["train_examples"] * 0.25), int(quotas["train_examples"] * 0.35)]
            shares.append(quotas["train_examples"] - sum(shares))
            if admitted["dev"] >= quotas["dev_examples"] and all(
                admitted[("train", b)] >= n for b, n in enumerate(shares)
            ):
                break
            if completed % 1000 == 0:
                print(
                    f"distill committed={completed}/{len(data)} admission={dict(reasons)}",
                    file=sys.stderr,
                    flush=True,
                )
        artifact = writer.finish(
            {
                "policy": contract,
                "decisions": dict(reasons),
                "target_origins": dict(origins),
                "durable_records": str(db_path),
                "durable_records_sha256": digest(db_path),
            }
        )
    run.publish(artifact)
    return dict(
        artifact=str(artifact),
        stats=writer.stats,
        decisions=dict(reasons),
        target_origins=dict(origins),
        generation_policy=contract["generation"],
        cache=str(db_path),
    )


def qualify(run, value, bound):
    data = Dataset(bound["distill"]["manifest"], DISTILLED)
    source = Dataset(bound["source"]["manifest"], CANDIDATES)
    locked_source = Dataset(source.manifest["extra"]["locked_evaluation"]["manifest"], LOCKED)
    policy = Qualification(**value["stage1_data"])
    pools = {s: [] for s in ("train", "dev")}
    for i, row in data.iter_metadata():
        run.meter.charge(
            steps=1, source=len(row["example"]["prompt_ids"]) + len(row["example"]["answer_ids"])
        )
        pools[row["example"]["split"]].append(i)
    selected = {}
    for split in pools:
        indices = sorted(
            pools[split],
            key=lambda i: canonical_hash(
                [policy.validation_seed, data.metadata(i)["example"]["original_id"]]
            ),
        )
        count = getattr(policy, split + "_examples")
        if split == "train":
            bins = [
                [
                    i
                    for i in indices
                    if (
                        0
                        if len(data.metadata(i)["example"]["answer_ids"]) <= 64
                        else 1
                        if len(data.metadata(i)["example"]["answer_ids"]) <= 128
                        else 2
                    )
                    == b
                ]
                for b in range(3)
            ]
            desired = [int(count * 0.25), int(count * 0.35)]
            desired.append(count - sum(desired))
            chosen = [i for b, n in zip(bins, desired, strict=True) for i in b[:n]]
            used = set(chosen)
            chosen += [i for i in indices if i not in used][: count - len(chosen)]
            selected[split] = chosen
        else:
            selected[split] = indices[:count]
    actual = {s: len(v) for s, v in selected.items()}
    blockers = []
    for split, minimum in (("train", policy.minimum_train_examples), ("dev", policy.dev_examples)):
        if actual[split] < minimum:
            blockers.append(f"{split}: need at least {minimum}, accepted {actual[split]}")
    report = dict(
        schema="stage3_data_qualification_v1",
        status="blocked" if blockers else "qualified",
        blockers=blockers,
        selected={},
        actual_counts={**actual, "test": len(locked_source)},
        target_policy=value["stage3"]["target_policy"],
        training_underfill_warning=actual["train"] < policy.train_examples,
        redistribution="unfilled length-bin quotas redistributed in fixed rank order without duplication",
        cross_experiment_scope=read(bound["source"]["manifest"])["identity"].get(
            "cross_experiment_scope", "explicit supplied manifests only"
        ),
        distribution_warning="distilled training/dev styles can differ from locked original-reference test",
    )
    for split, indices in selected.items():
        lengths = Counter(len(data.metadata(i)["example"]["answer_ids"]) for i in indices)
        examples = (data.metadata(i)["example"] for i in indices)
        origins, components, groups = Counter(), Counter(), set()
        for example in examples:
            origins[example["provenance"].get("target_origin", example["source_type"])] += 1
            components[example["provenance"].get("component_source", "software_fixture")] += 1
            groups.add(example["group_id"])
        report["selected"][split] = dict(
            examples=len(indices),
            length_histogram=dict(lengths),
            target_origins=dict(origins),
            components=dict(components),
            groups=sorted(groups),
            correctness_coverage="unknown for free-form targets; no external judge",
            unique_answer_positions=sum(k * v for k, v in lengths.items()),
            actual_length_bins=[
                sum(v for k, v in lengths.items() if low <= k <= high)
                for low, high in ((1, 64), (65, 128), (129, max(256, value["architecture"]["canvas"])))
            ],
        )
    report["selected"]["test"] = {"examples": len(locked_source), **locked_source.manifest["stats"]}
    qualification = run.path / "qualification.json"
    write_json(qualification, report, exclusive=True)
    if report["training_underfill_warning"] and not blockers:
        print(
            f"WARNING: Stage 3 admitted {actual['train']} distinct training pairs; target {policy.train_examples}, minimum {policy.minimum_train_examples}. Actual counts are sealed; no examples duplicated.",
            file=sys.stderr,
            flush=True,
        )
    if blockers:
        raise DataCapacityError(
            "; ".join(blockers) + f"; inspect {qualification}; new recipe required to change quotas"
        )
    final_identity = {
        **data.manifest["identity"],
        "qualification_sha256": digest(qualification),
        "source_contract_sha256": canonical_hash([data.sha256, selected]),
        "qualification_mode": "stage3_qualified_v1",
    }
    locked = ArtifactWriter(run.path / "artifacts/evaluation", LOCKED, final_identity, run.meter)
    for i in range(len(locked_source)):
        locked.append(locked_source[i][0])
    evaluation = locked.finish()
    writer = ArtifactWriter(run.path / "artifacts/data", "native_data_v1", final_identity, run.meter)
    for indices in selected.values():
        for i in indices:
            writer.append(data[i][0])
    artifact = writer.finish(
        {"qualification": reference(qualification), "locked_evaluation": reference(evaluation)}
    )
    run.publish(artifact)
    return dict(
        artifact=str(artifact),
        qualification=reference(qualification),
        evaluation_data=reference(evaluation),
        stats=writer.stats,
        model_loaded=False,
    )
