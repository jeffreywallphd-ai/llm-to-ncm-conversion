"""Prompt-only, optimizer-boundary observation that preserves modes and RNG."""

from contextlib import contextmanager
import random
import sys
import time
import torch
from ncm_lab.inference.export import generate_tokens


@contextmanager
def preserve_training_state(model):
    modes = [(m, m.training) for m in model.modules()]
    python = random.getstate()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    numpy = sys.modules.get("numpy")
    np_state = numpy.random.get_state() if numpy is not None else None
    observer = getattr(model, "diagnostic_observer", None)
    try:
        model.eval()
        model.diagnostic_observer = None
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        model.diagnostic_observer = observer
        random.setstate(python)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if np_state is not None:
            numpy.random.set_state(np_state)


def fixed_prompts(dataset, indices, count):
    # Metadata only: no teacher/latent tensors, references or training iterator.
    rows = []
    for index in indices[:count]:
        example = dataset.metadata(index)["example"]
        rows.append({"id": example["original_id"], "prompt_ids": example["prompt_ids"]})
    return rows


def observe_samples(model, tokenizer, prompts, recorder, meter, eos_ids):
    options = recorder.options
    started = time.perf_counter()
    rows = []
    with preserve_training_state(model), torch.no_grad():
        for prompt in prompts[: options.sample_prompts]:
            if time.perf_counter() - started >= options.sample_seconds:
                break
            meter.check()
            ids = torch.tensor([prompt["prompt_ids"]], device=next(model.parameters()).device)
            meter.charge(source=ids.numel(), canvas=model.architecture.canvas)
            sample_start = time.perf_counter()
            result = generate_tokens(
                model, ids, torch.ones_like(ids, dtype=torch.bool), eos_ids, check=meter.check
            )
            elapsed = time.perf_counter() - sample_start
            text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
            prompt_text = tokenizer.decode(prompt["prompt_ids"], skip_special_tokens=True)
            row = {
                "prompt_id": prompt["id"],
                "prompt": prompt_text[: options.text_chars],
                "text": text[: options.text_chars],
                "display_text_shortened": len(text) > options.text_chars,
                "prompt_display_shortened": len(prompt_text) > options.text_chars,
                "returned_tokens": len(result["tokens"]),
                "eos_position": len(result["tokens"]) - 1 if result["eos_found"] else None,
                "truncated": result["truncated"],
                "computed_canvas_positions": result["computed_answer_positions"],
                "generation_seconds": elapsed,
                "settings": {"decoding": "parallel argmax", "eos_ids": eos_ids},
                "model_identity": recorder.config_sha,
                "checkpoint": "current in-memory optimizer boundary",
                "cooperative_wall_limit_overrun": time.perf_counter() - started > options.sample_seconds,
            }
            recorder.append("samples", split="dev", **row)
            rows.append(row)
    return rows
