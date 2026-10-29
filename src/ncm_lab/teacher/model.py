"""Separate cache-only teacher and trainable original-Qwen loading."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from ncm_lab.config.architecture import PIN, Architecture
from ncm_lab.embedded.model import EmbeddedGenerator


def validate_model_spec(spec, config):
    if spec.get("fixture"):
        if spec.get("model_id") != "software-fixture/qwen3":
            raise ValueError("fixture source must remain explicitly labeled")
        return
    if (
        spec.get("model_id") != "Qwen/Qwen3-8B"
        or spec.get("revision") != PIN
        or spec.get("tokenizer_revision") != PIN
        or spec.get("thinking", False)
    ):
        raise ValueError("pinned Qwen3-8B/tokenizer/non-thinking identity required")
    required = {
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    if any(getattr(config, key, None) != value for key, value in required.items()):
        raise ValueError("loaded Qwen configuration differs from the pinned 8B backbone")


def model_identity(spec):
    from ncm_lab.runtime.io import canonical_hash

    identity = {k: spec.get(k) for k in ("model_id", "revision", "tokenizer_revision", "thinking", "fixture")}
    identity["eos_ids"] = spec.get("eos_ids", [2] if spec.get("fixture") else [151645, 151643])
    if spec.get("fixture"):
        identity.update(config_sha256=canonical_hash(spec["config"]), seed=spec.get("seed", 7))
    return identity


def load_original(spec, device="cpu", dtype=torch.float32, *, frozen):
    if spec.get("fixture"):
        with torch.random.fork_rng():
            torch.manual_seed(spec.get("seed", 7))
            model = Qwen3ForCausalLM(Qwen3Config(**spec["config"]))
    else:
        verify_cached_assets(spec)
        model = AutoModelForCausalLM.from_pretrained(
            spec.get("path", spec["model_id"]),
            revision=spec["revision"],
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            dtype=dtype,
            use_safetensors=True,
        )
    validate_model_spec(spec, model.config)
    model.to(device=device, dtype=dtype)
    model.config.use_cache = False
    if frozen:
        model.requires_grad_(False).eval()
    return model


def load_generator(spec, architecture, device="cpu", dtype=torch.float32):
    backbone = load_original(spec, device, dtype, frozen=False)
    return EmbeddedGenerator(backbone, Architecture(**architecture)).to(device=device, dtype=dtype)


def load_tokenizer(spec):
    if spec.get("fixture"):
        from ncm_lab.inference.export import fixture_tokenizer

        return fixture_tokenizer()
    return AutoTokenizer.from_pretrained(
        spec.get("path", spec["model_id"]),
        revision=spec["tokenizer_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )


def cached_path(spec):
    from pathlib import Path
    from ncm_lab.runtime.io import root

    if spec.get("path"):
        return (root() / spec["path"]).resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec["model_id"], revision=spec["revision"], local_files_only=True))


def verify_cached_assets(spec):
    from ncm_lab.runtime.io import root, read, digest

    path = cached_path(spec)
    assets = [a for a in read(root() / "configs/data/assets.lock.json")["assets"] if a["role"] == "target"]
    from ncm_lab.data.download import verify_asset

    for asset in assets:
        if verify_asset(path / asset["filename"], asset) is None:
            raise ValueError("missing/corrupt pinned Qwen asset: " + asset["filename"])
    return {a["filename"]: digest(path / a["filename"]) for a in assets}


def load_frozen_head(spec, device, dtype):
    if spec.get("fixture"):
        model = load_original(spec, device, dtype, frozen=True)
        return model.lm_head
    from safetensors import safe_open
    from ncm_lab.runtime.io import read
    from torch import nn

    path = cached_path(spec)
    verify_cached_assets(spec)
    index = read(path / "model.safetensors.index.json")
    with safe_open(path / index["weight_map"]["lm_head.weight"], framework="pt", device="cpu") as source:
        tensor = source.get_tensor("lm_head.weight")
    if tensor.shape != (151936, 4096):
        raise ValueError("pinned vocabulary head shape mismatch")
    with torch.device("meta"):
        head = nn.Linear(4096, 151936, bias=False)
    head.weight = nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False)
    return head.eval()


@torch.no_grad()
def prediction_states(teacher, prompt_ids, answer_ids):
    if not prompt_ids or not answer_ids:
        raise ValueError("nonempty prompt and reference answer required")
    device = teacher.get_input_embeddings().weight.device
    # Last answer input is unnecessary: state P-1+t predicts token y[t].
    tokens = torch.tensor([prompt_ids + answer_ids[:-1]], device=device)
    states = teacher.model(input_ids=tokens, use_cache=False).last_hidden_state
    return states[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(answer_ids)].detach()
