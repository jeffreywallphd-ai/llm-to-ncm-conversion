"""Explicit optional original-Qwen AR reference. Never repairs NCM outputs."""

import time
import torch
from ncm_lab.teacher.model import load_original, load_tokenizer
from ncm_lab.data.artifacts import Dataset, LOCKED
from ncm_lab.runtime.io import write_json


def evaluate(run, value, bound):
    cap = value["stage1_context"]["reference_max_new_tokens"]
    data = Dataset(bound["evaluation"]["manifest"], LOCKED)
    model = load_original(value["model"], value["device"], getattr(torch, value["dtype"]), frozen=True)
    tokenizer = load_tokenizer(value["model"])
    rows = []
    with torch.inference_mode():
        for i in data.indices("test"):
            example, _ = data[i]
            ids = torch.tensor([example.prompt_ids], device=value["device"])
            # Declare the conservative no-cache AR maximum before work. Early EOS
            # does not fabricate uncharged capacity or a speedup claim.
            run.meter.charge(source=cap * (len(example.prompt_ids) + cap), answer=cap, canvas=cap, steps=1)
            if value["device"] == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(
                ids,
                max_new_tokens=cap,
                do_sample=False,
                use_cache=False,
                eos_token_id=value["model"].get("eos_ids", [tokenizer.eos_token_id]),
                pad_token_id=tokenizer.pad_token_id,
            )
            if value["device"] == "cuda":
                torch.cuda.synchronize()
            tokens = output[0, ids.shape[1] :].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            rows.append(
                {
                    "id": example.original_id,
                    "group_id": example.group_id,
                    "prompt": example.provenance["prompt_text"],
                    "reference": example.provenance["reference_text"],
                    "prediction": text,
                    "returned_positions": len(tokens),
                    "cap": cap,
                    "generation_seconds": time.perf_counter() - started,
                    "eos_found": any(
                        t in value["model"].get("eos_ids", [tokenizer.eos_token_id]) for t in tokens
                    ),
                }
            )
    artifact = run.path / "artifacts/ar_reference.json"
    write_json(
        artifact,
        {
            "schema": "stage1_optional_ar_reference_v1",
            "rows": rows,
            "source": bound["evaluation"],
            "max_new_tokens": cap,
            "frozen_selection": bound["generator"],
            "likelihood_comparable_to_parallel_ce": False,
            "used_to_repair_ncm": False,
            "work_charges": "conservative declared no-cache AR upper bound; actual returned tokens separate",
            "limitations": "AR quality/latency reference only; not an automatic speedup or correctness claim",
        },
        exclusive=True,
    )
    run.publish(artifact)
    return {"artifact": str(artifact), "reference_max_new_tokens": cap, "evaluated_examples": len(rows)}
