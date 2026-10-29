"""Independent inference artifact: deployed generator and tokenizer, no teacher/codec."""

from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from ncm_lab.runtime.io import read, write_json, digest, canonical_hash, resolve_reference
from ncm_lab.embedded.model import from_metadata
from ncm_lab.training.checkpoints import save_checkpoint, CheckpointReader, INFERENCE


def fixture_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["<pad>", "<bos>", "<eos>", "<unk>"] + [f"token{i}" for i in range(4, 32)]
    tokenizer = Tokenizer(WordLevel({s: i for i, s in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"
    )
    result.chat_template = "{{ bos_token }} {{ messages[0]['content'] }}"
    return result


def export_generator(model, metadata, path, meter, shard_bytes, spec):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    tokenizer = (
        fixture_tokenizer()
        if spec.get("fixture")
        else AutoTokenizer.from_pretrained(
            spec.get("path", spec["model_id"]),
            revision=spec["tokenizer_revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    if canonical_hash(tokenizer.chat_template) != metadata["source"]["template_sha256"]:
        raise ValueError("training/deployment chat template differs")
    tokenizer.save_pretrained(path / "tokenizer")
    identity = canonical_hash(metadata)
    saved = save_checkpoint(
        path / "weights",
        {"identity": identity, "metadata": metadata, "model": model.state_dict(keep_vars=True)},
        kind=INFERENCE,
        meter=meter,
        shard_bytes=shard_bytes,
    )
    manifest = {
        "schema": "native_generator_export_v1",
        "identity": identity,
        "weights_manifest": str(Path(saved["manifest"]).relative_to(path)),
        "weights_sha256": saved["sha256"],
        "tokenizer_files": {p.name: digest(p) for p in (path / "tokenizer").iterdir() if p.is_file()},
        "template_sha256": canonical_hash(tokenizer.chat_template),
        "thinking": False,
        "eos_ids": spec.get("eos_ids", [tokenizer.eos_token_id]),
        "architecture": metadata["architecture"],
        "model": metadata["model"],
        "dtype": metadata["dtype"],
        "backbone_config": metadata["backbone_config"],
        "teacher_or_codec_required": False,
    }
    write_json(path / "manifest.json", manifest, exclusive=True)
    return path / "manifest.json"


def load_export(path, device="cpu", check=None):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    path = resolve_reference(path)
    manifest = read(path)
    if manifest.get("schema") != "native_generator_export_v1":
        raise ValueError("independent native inference export required, not a recovery/legacy checkpoint")
    weights = (path.parent / manifest["weights_manifest"]).resolve()
    if not weights.is_relative_to(path.parent.resolve()) or digest(weights) != manifest["weights_sha256"]:
        raise ValueError("inference manifest/weights changed")
    for name, sha in manifest["tokenizer_files"].items():
        target = (path.parent / "tokenizer" / name).resolve()
        if target.parent != (path.parent / "tokenizer").resolve() or digest(target) != sha:
            raise ValueError("exported tokenizer changed")
    reader = CheckpointReader(weights, expected_kind=INFERENCE, identity=manifest["identity"], check=check)
    metadata = reader.materialize(reader.field("metadata"))
    model = from_metadata(metadata, device)
    reader.load_model(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path.parent / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    if canonical_hash(tokenizer.chat_template) != manifest["template_sha256"]:
        raise ValueError("exported prompt policy changed")
    return model, tokenizer, manifest


@torch.no_grad()
def generate_tokens(
    model, prompt_ids, prompt_mask, eos_ids, chunk_positions=None, check=None, inference=None
):
    from ncm_lab.inference.policy import Inference

    settings = dict(inference or {})
    if chunk_positions is not None:
        settings["head_chunk_positions"] = chunk_positions
    policy = Inference(**settings).resolve(
        model.architecture.canvas,
        model.backbone.config.vocab_size,
        model.backbone.lm_head.weight.element_size(),
        prompt_ids.shape[0],
    )
    chunk_positions = policy["resolved_head_chunk_positions"]
    if prompt_ids.shape[0] != 1:
        raise ValueError("initial text-generation API supports one request at a time")
    if int(prompt_mask.sum()) + model.architecture.canvas > model.backbone.config.max_position_embeddings:
        raise ValueError("prompt and fixed canvas exceed positional capacity")
    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(prompt_ids, prompt_mask)
    device_ids = []
    for start in range(0, model.architecture.canvas, chunk_positions):
        if check:
            check()
        device_ids.append(
            model.backbone.lm_head(output.answer_states[:, start : start + chunk_positions]).argmax(-1)[0]
        )
    tokens = transfer_token_ids(torch.cat(device_ids))
    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop = next((i + 1 for i, t in enumerate(tokens) if t in eos_ids), len(tokens))
    present = any(t in eos_ids for t in tokens)
    return {
        "tokens": tokens[:stop],
        "eos_found": present,
        "truncated": not present,
        "allocated_answer_positions": model.architecture.canvas,
        "computed_answer_positions": len(tokens),
        "backbone_traversals": 1,
        "logical_block_trace": list(output.trace),
        "forward_and_head_seconds": elapsed,
        "decoding": "parallel argmax; no AR repair or verification",
        "inference_projection": policy,
    }


def transfer_token_ids(ids):
    """One final small ID transfer, with no vocabulary-logit concatenation."""
    return ids.to(device="cpu").tolist()


def generate(path, prompt_file, device="cpu", meter=None, inference=None):
    model, tokenizer, manifest = load_export(path, device, meter.check if meter else None)
    text = Path(prompt_file).read_text(encoding="utf-8")
    if len(text) > 32768:
        raise ValueError("bounded initial prompt character limit exceeded")
    from ncm_lab.data.tokenization import prompt_tokens

    ids = prompt_tokens(tokenizer, text)
    if len(ids) > 512:
        raise ValueError("initial generation prompt token limit is 512")
    tensor = torch.tensor([ids], device=device)
    if meter:
        meter.charge(source=len(ids), canvas=model.architecture.canvas, steps=1)
        for layer in model.backbone.model.layers:
            layer.register_forward_pre_hook(lambda *args: meter.check())
    result = generate_tokens(
        model,
        tensor,
        torch.ones_like(tensor, dtype=torch.bool),
        manifest["eos_ids"],
        check=meter.check if meter else None,
        inference=inference,
    )
    return {
        **result,
        "text": tokenizer.decode(result["tokens"], skip_special_tokens=True),
        "export_identity": manifest["identity"],
    }
