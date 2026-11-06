"""Frozen student and original AR reference, loaded serially, with durable outputs."""

import gc
import json
from pathlib import Path
import time
import torch
from ncm_lab.runtime.io import read, write_json, reference, canonical_hash
from ncm_lab.stage1.workflow import checked
from ncm_lab.data.artifacts import Dataset, LOCKED
from ncm_lab.stage3.data import generation_policy
from ncm_lab.stage3.storage import charge, check_cache
from ncm_lab.teacher.model import load_original, load_tokenizer, model_identity


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def evaluate(run, value, bound):
    from ncm_lab.stage1.evaluation import evaluate as student_evaluate

    (run.path / "artifacts").mkdir(parents=True, exist_ok=True)
    cache = Path(value["stage1_context"]["experiment"]).parent / "final_evaluation"
    cache.mkdir(parents=True, exist_ok=True)
    contract = canonical_hash([bound, value["stage3"], value["inference"]])
    receipt = cache / "student.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["contract"] != contract:
            raise ValueError("frozen evaluation contract changed")
        student = read(checked(saved["artifact"]))
        checked(student["raw_outputs"])
    else:
        student_result = student_evaluate(run, value, bound)
        student = student_result["evaluation"]
        write_json(
            receipt, {"contract": contract, "artifact": reference(student_result["artifact"])}, exclusive=True
        )
    # student_evaluate's model is out of scope before original Qwen allocation.
    gc.collect()
    if value["device"] == "cuda":
        torch.cuda.empty_cache()
    started = time.perf_counter()
    model = load_original(value["model"], value["device"], getattr(torch, value["dtype"]), frozen=True)
    tokenizer = load_tokenizer(value["model"])
    sync(value["device"])
    load_seconds = time.perf_counter() - started
    data = Dataset(bound["evaluation"]["manifest"], LOCKED)
    eos = model_identity(value["model"])["eos_ids"]
    cap = value["architecture"]["canvas"]
    cfg = generation_policy(model, tokenizer, cap, eos)
    options = value["stage3"]["evaluation"]
    record_dir = cache / "original_qwen"
    record_dir.mkdir(exist_ok=True)
    rows = []
    warmed = False
    if value["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index in data.indices("test"):
            example, _ = data[index]
            record_path = record_dir / (canonical_hash(example.original_id) + ".json")
            record_contract = canonical_hash([contract, example.identity, cfg.to_dict()])
            if record_path.exists():
                saved = read(record_path)
                if saved["contract"] != record_contract or canonical_hash(saved["row"]) != saved["sha256"]:
                    raise ValueError("cached final original-Qwen output changed")
                rows.append(saved["row"])
                continue
            ids = torch.tensor([example.prompt_ids], device=value["device"])
            times = []
            warmup = options["warmup"] if not warmed else 0
            for repetition in range(options["repeats"] + warmup):
                run.meter.charge(
                    steps=1, source=cap * (len(example.prompt_ids) + cap), answer=cap, canvas=cap
                )
                sync(value["device"])
                started = time.perf_counter()
                output = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg
                )
                sync(value["device"])
                elapsed = time.perf_counter() - started
                if repetition >= warmup:
                    times.append(elapsed)
                tokens = output[0, len(example.prompt_ids) :].tolist()
            warmed = True
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            rule = example.provenance.get("objective_rule")
            row = dict(
                id=example.original_id,
                group_id=example.group_id,
                example_sha256=example.identity,
                prompt=example.provenance["prompt_text"],
                reference=example.provenance["reference_text"],
                prediction=text,
                token_ids=tokens,
                returned_positions=len(tokens),
                cap=cap,
                eos_found=bool(tokens and tokens[-1] in eos),
                empty_answer=not text.strip(),
                repeated_token_fraction=1 - len(set(tokens)) / max(1, len(tokens)),
                generation_seconds=times,
                warmup_generations=warmup,
                measurement_attempt=run.id,
                target_origin="original_reference",
                objective_rule=rule,
                task_correct=" ".join(text.split()) == " ".join(example.provenance["reference_text"].split())
                if rule == "normalized_exact"
                else None,
                reference_positions=len(example.answer_ids),
                correctness_scope="unknown without supported validator",
            )
            charge(run, len(json.dumps(row).encode()) + 16384)
            check_cache(run, cache)
            write_json(
                record_path,
                {"contract": record_contract, "row": row, "sha256": canonical_hash(row)},
                exclusive=True,
            )
            check_cache(run, cache)
            rows.append(row)
    peak = torch.cuda.max_memory_allocated() if value["device"] == "cuda" else None
    del model
    gc.collect()
    if value["device"] == "cuda":
        torch.cuda.empty_cache()
    raw = run.path / "artifacts/original_qwen.jsonl"
    with raw.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    ar = run.path / "artifacts/original_qwen.json"
    write_json(
        ar,
        dict(
            schema="stage3_original_qwen_reference_v1",
            used_to_repair_ncm=False,
            frozen_selection=bound["generator"],
            locked_data=bound["evaluation"],
            generation=cfg.to_dict(),
            raw_outputs=reference(raw),
            model_loading_seconds=load_seconds,
            peak_allocated_bytes=peak,
            examples=len(rows),
            environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
            caching_scope="exact frozen evaluation contract only",
        ),
        exclusive=True,
    )
    student_rows = [json.loads(line) for line in checked(student["raw_outputs"]).read_text().splitlines()]
    bins = {}
    for low, high in ((1, 64), (65, 128), (129, max(256, cap))):
        matched = [r for r in rows if low <= r["reference_positions"] <= high]
        ids = {r["id"] for r in matched}
        predictions = [r for r in student_rows if r["split"] == "test" and r["id"] in ids]
        bins[f"{low}-{high}"] = dict(
            examples=len(matched),
            ids=sorted(ids),
            student_empty=sum(r["empty_answer"] for r in predictions),
            reference_empty=sum(r["empty_answer"] for r in matched),
            student_absent_eos=sum(r["absent_eos"] for r in predictions),
            reference_absent_eos=sum(not r["eos_found"] for r in matched),
        )
    report = {
        **student,
        "schema": "stage3_selected_evaluation_v1",
        "ar_reference": reference(ar),
        "ar_raw_outputs": reference(raw),
        "length_bins": bins,
        "serial_model_loading": True,
        "matched_prompt_ids": [r["id"] for r in rows],
        "human_review": "required for open-ended relevance/coherence",
        "distillation_distribution_difference": "training/dev may be teacher-generated; test uses original references",
    }
    artifact = run.path / "artifacts/stage3_evaluation.json"
    write_json(artifact, report, exclusive=True)
    run.publish(artifact)
    return dict(artifact=str(artifact), evaluation=report)
