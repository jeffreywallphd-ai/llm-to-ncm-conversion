"""Selected-export text CE and raw prompt-only generations; locked data never selects."""

import json
import time
import torch
from ncm_lab.data.artifacts import Dataset, LOCKED
from ncm_lab.data.records import collate
from ncm_lab.inference.export import load_export, generate_tokens
from ncm_lab.training.losses import chunked_ce
from ncm_lab.runtime.io import write_json, reference, disk_path


def evaluate(run, value, bound):
    model, tokenizer, manifest = load_export(bound["generator"]["manifest"], value["device"], run.meter.check)
    if manifest["architecture"] != value["architecture"]:
        raise ValueError("evaluation export geometry differs from the stored experiment")
    development = Dataset(bound["data"]["manifest"], "native_data_v1")
    locked = Dataset(bound["evaluation"]["manifest"], LOCKED)
    if (
        development.manifest["identity"]["qualification_sha256"]
        != locked.manifest["identity"]["qualification_sha256"]
    ):
        raise ValueError("locked evaluation/source qualification differs")
    groups = set(development.manifest["groups"]["train"]) | set(development.manifest["groups"]["dev"])
    if groups & set(locked.manifest["groups"]["test"]):
        raise ValueError("locked evaluation group leaked into fitting or selection")
    folder = run.path / "artifacts/evaluation"
    disk_path(folder).mkdir(parents=True)
    raw = folder / "outputs.jsonl"
    summaries = {}
    with disk_path(raw).open("x", encoding="utf-8", newline="\n") as stream, torch.inference_mode():
        for split, dataset, indices in (
            ("dev", development, development.indices("dev")),
            ("test", locked, locked.indices("test")),
        ):
            results = []
            for index in indices:
                example, _ = dataset[index]
                ids, mask, labels, valid = collate([example], model.architecture.canvas, value["device"])
                run.meter.charge(
                    source=2 * len(example.prompt_ids) + len(example.answer_ids),
                    answer=len(example.answer_ids),
                    canvas=2 * model.architecture.canvas,
                    steps=2,
                )
                started = time.perf_counter()
                output = model(ids, mask)
                ce = float(
                    chunked_ce(
                        model.backbone.lm_head,
                        output.answer_states,
                        labels,
                        valid,
                        value["training"]["head_chunk"],
                    )
                )
                del output
                generation_started = time.perf_counter()
                prediction = generate_tokens(
                    model, ids, mask, manifest["eos_ids"], check=run.meter.check, inference=value["inference"]
                )
                content = [t for t in prediction["tokens"] if t not in manifest["eos_ids"]]
                text = tokenizer.decode(prediction["tokens"], skip_special_tokens=True)
                elapsed = time.perf_counter() - generation_started
                reference_text = example.provenance.get(
                    "reference_text", tokenizer.decode(example.answer_ids, skip_special_tokens=True)
                )
                rule = example.provenance.get("objective_rule")
                if rule not in {None, "normalized_exact"}:
                    raise ValueError("unsupported objective-answer rule; generated code is never executed")
                row = {
                    "split": split,
                    "id": example.original_id,
                    "group_id": example.group_id,
                    "source_sha256": example.provenance["source_sha256"],
                    "example_sha256": example.identity,
                    "prompt": example.provenance.get(
                        "prompt_text", tokenizer.decode(example.prompt_ids, skip_special_tokens=True)
                    ),
                    "reference": reference_text,
                    "prediction": text,
                    "text_ce": ce,
                    "reference_positions_including_eos": len(example.answer_ids),
                    "prompt_positions": len(example.prompt_ids),
                    "output_content_tokens": len(content),
                    "returned_positions_including_eos": len(prediction["tokens"]),
                    "eos_found": prediction["eos_found"],
                    "absent_eos": not prediction["eos_found"],
                    "empty_answer": not text.strip(),
                    "premature_eos_length_diagnostic": prediction["eos_found"]
                    and len(content) < len(example.answer_ids) - 1,
                    "repeated_token_fraction": 1 - len(set(content)) / max(1, len(content)),
                    "exact_match_diagnostic": text.strip() == reference_text.strip(),
                    "objective_rule": rule,
                    "task_correct": " ".join(text.split()) == " ".join(reference_text.split())
                    if rule
                    else None,
                    "generation_seconds": elapsed,
                    "ce_and_generation_seconds": time.perf_counter() - started,
                    "allocated_answer_positions": prediction["allocated_answer_positions"],
                    "computed_answer_positions": prediction["computed_answer_positions"],
                    "generation_backbone_traversals": prediction["backbone_traversals"],
                    "export_identity": manifest["identity"],
                    "selected_optimizer_step": manifest.get("selected_optimizer_step"),
                    "relevance_and_coherence": "requires human assessment; raw triplet retained",
                }
                if not __import__("math").isfinite(ce):
                    raise ValueError("nonfinite final text CE")
                run.meter.check_disk(len(json.dumps(row).encode()) + 4096)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                results.append(row)
            count = len(results)
            metrics = (
                "text_ce",
                "output_content_tokens",
                "empty_answer",
                "absent_eos",
                "premature_eos_length_diagnostic",
                "repeated_token_fraction",
                "exact_match_diagnostic",
                "generation_seconds",
            )
            summaries[split] = {
                "examples": count,
                **{k: sum(r[k] for r in results) / count for k in metrics},
                "invalid_output_rate_definition": "empty decoded answer",
                "invalid_output_rate": sum(r["empty_answer"] for r in results) / count,
                "task_correctness": None
                if not any(r["task_correct"] is not None for r in results)
                else sum(r["task_correct"] for r in results if r["task_correct"] is not None)
                / sum(r["task_correct"] is not None for r in results),
                "common_8_position_subset": {
                    "ids": [r["id"] for r in results if r["reference_positions_including_eos"] <= 8],
                    "text_ce": sum(
                        r["text_ce"] for r in results if r["reference_positions_including_eos"] <= 8
                    )
                    / max(1, sum(r["reference_positions_including_eos"] <= 8 for r in results))
                    if any(r["reference_positions_including_eos"] <= 8 for r in results)
                    else None,
                },
            }
    report = {
        "schema": "stage1_selected_evaluation_v1",
        "export": bound["generator"],
        "qualification_sha256": locked.manifest["identity"]["qualification_sha256"],
        "selection_frozen_before_test": True,
        "selected_optimizer_step": manifest.get("selected_optimizer_step"),
        "splits": summaries,
        "raw_outputs": reference(raw),
        "teacher_or_codec_loaded": False,
        "limitations": [
            "Exact match is diagnostic for open-ended language.",
            "Premature stopping is a length comparison, not a semantic judgment.",
            "One logical generation traversal is not measured speedup.",
            "Different natural size-specific eligible datasets do not isolate architecture.",
        ],
    }
    artifact = folder / "manifest.json"
    write_json(artifact, report, exclusive=True)
    run.publish(artifact)
    return {"artifact": str(artifact), "evaluation": report}
