"""Full-backbone denoising plus shifted AR supervision; independent sharded recovery."""

from contextlib import closing
import json
import math
from pathlib import Path
import random
import shutil
import torch
from ncm_lab.block8.config import Training
from ncm_lab.block8.data import BlockDataset
from ncm_lab.block8.model import BlockDecoder, load_model
from ncm_lab.block8.runtime import admission, checked, charge_write
from ncm_lab.runtime.io import canonical_hash, digest, reference, read, write_json
from ncm_lab.teacher.model import load_tokenizer
from ncm_lab.training.checkpoints import (
    CheckpointReader,
    INFERENCE,
    recovery_payload,
    restore_rng,
    save_checkpoint,
)
from ncm_lab.training.losses import chunked_ce
from ncm_lab.training.optimizer import TiledAdamW


def example_loss(model, context, answer, size, sampler, policy, *, fully_masked=False):
    """Gold labels/masks stay here, outside both model forward interfaces."""
    device = model.mask_embedding.device
    context_ids = torch.tensor([context], dtype=torch.long, device=device)
    labels = torch.full((1, size), -100, dtype=torch.long, device=device)
    labels[0, : len(answer)] = torch.tensor(answer, device=device)
    valid = labels.ne(-100)
    active = torch.full_like(labels, -1)
    masked = list(range(len(answer)))
    if not fully_masked and sampler.random() >= policy.fully_masked_fraction:
        count = sampler.randint(1, len(answer))
        masked = sampler.sample(masked, count)
        for i, token in enumerate(answer):
            if i not in masked:
                active[0, i] = token
    predict = torch.zeros_like(valid)
    predict[0, masked] = True
    states = model.denoise(context_ids, active)
    denoising = chunked_ce(model.backbone.lm_head, states, labels, predict, policy.head_chunk)
    ar = denoising.new_zeros(())
    if not fully_masked and policy.ar_weight:
        # State P-1+t predicts answer[t]; active denoising has NO such shift.
        clean = torch.tensor([context + answer[:-1]], device=device, dtype=torch.long)
        ar_states = model.causal_states(clean)[:, len(context) - 1 :]
        ar_labels = torch.tensor([answer], device=device)
        ar = chunked_ce(
            model.backbone.lm_head,
            ar_states,
            ar_labels,
            torch.ones_like(ar_labels, dtype=torch.bool),
            policy.head_chunk,
        )
    return denoising + policy.ar_weight * ar, denoising, ar


def charge_example(run, context, answer, size, *, auxiliary):
    run.meter.charge(source=len(context) + size, answer=len(answer), canvas=size, steps=1)
    if auxiliary:
        run.meter.charge(source=len(context) + len(answer) - 1, answer=len(answer), steps=1)


@torch.no_grad()
def validate(model, data, run, policy):
    model.eval()
    rows = data.views("dev", policy.max_validation_views)
    if not rows:
        raise ValueError("block8 validation has no held-out views")
    total = 0.0
    for key, start in rows:
        context, answer = data.view(key, start)
        charge_example(run, context, answer, 8, auxiliary=False)
        _, loss, _ = example_loss(model, context, answer, 8, random.Random(0), policy, fully_masked=True)
        value = float(loss)
        if not math.isfinite(value):
            raise ValueError("nonfinite held-out block8 loss")
        total += value
    model.train()
    return total / len(rows)


def export(model, settings, data, run, step, score):
    folder = run.path / "exports" / f"step-{step:09d}"
    folder.mkdir(parents=True)
    tokenizer = load_tokenizer(settings["model"])
    if canonical_hash(tokenizer.chat_template) != data.manifest["template_sha256"]:
        raise ValueError("block8 training/deployment template differs")
    run.meter.check_disk(64000000)
    charge_write(run, 64000000)
    tokenizer.save_pretrained(folder / "tokenizer")
    if sum(p.stat().st_size for p in (folder / "tokenizer").iterdir()) > 64000000:
        raise ValueError("block8 tokenizer exceeds its declared 64 MB export allowance")
    metadata = dict(
        model_family="block8_v1",
        architecture=settings["architecture"],
        backbone_config=model.backbone.config.to_dict(),
        model_spec=settings["model"],
        dtype=settings["dtype"],
        implementation_sha256=settings["implementation_sha256"],
        selected_step=step,
        selection_loss=score,
        data_sha256=data.sha256,
        template_sha256=data.manifest["template_sha256"],
    )
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    identity = canonical_hash(metadata)
    saved = save_checkpoint(
        folder / "weights",
        dict(identity=identity, metadata=metadata, model=model.state_dict(keep_vars=True)),
        kind=INFERENCE,
        meter=run.meter,
        shard_bytes=settings["training"]["shard_bytes"],
    )
    manifest = dict(
        schema="block8_export_v1",
        identity=identity,
        metadata=metadata,
        weights=reference(saved["manifest"]),
        tokenizer_files={p.name: digest(p) for p in (folder / "tokenizer").iterdir() if p.is_file()},
        selected_step=step,
        selection="mean_fixed_dev_view_fully_masked_token_ce_v1",
        teacher_codec_or_verifier_required=False,
    )
    write_json(folder / "manifest.json", manifest, exclusive=True)
    return reference(folder / "manifest.json")


def load_export(path, device, *, check=None):
    """Only this separate export family is accepted; verify every shard before allocation."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from ncm_lab.block8.config import Architecture
    from ncm_lab.teacher.model import validate_model_spec

    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    manifest = read(path)
    if manifest.get("schema") != "block8_export_v1":
        raise ValueError("block8 inference export required; native/recovery artifacts are incompatible")
    metadata = manifest["metadata"]
    if metadata.get("model_family") != "block8_v1" or canonical_hash(metadata) != manifest["identity"]:
        raise ValueError("block8 export metadata identity changed")
    Architecture(**metadata["architecture"])
    weights = checked(manifest["weights"])
    if not weights.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError("block8 weights escape export directory")
    reader = CheckpointReader(weights, expected_kind=INFERENCE, identity=manifest["identity"], check=check)
    if reader.materialize(reader.field("metadata")) != metadata:
        raise ValueError("block8 sharded metadata differs")
    tokenizer_dir = path.parent / "tokenizer"
    for name, sha in manifest["tokenizer_files"].items():
        p = (tokenizer_dir / name).resolve()
        if p.parent != tokenizer_dir.resolve() or digest(p, check) != sha:
            raise ValueError("block8 exported tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    if canonical_hash(tokenizer.chat_template) != metadata["template_sha256"]:
        raise ValueError("block8 exported template changed")
    config = Qwen3Config(**metadata["backbone_config"])
    validate_model_spec(metadata["model_spec"], config)
    # Construct on meta, including the single learned mask vector; initialize RoPE explicitly.
    with torch.device("meta"):
        model = BlockDecoder(Qwen3ForCausalLM(config), Architecture(**metadata["architecture"]))
    model.to(dtype=getattr(torch, metadata["dtype"]))
    model.to_empty(device=device)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    model.backbone.model.rotary_emb = Qwen3RotaryEmbedding(config).to(
        device=device, dtype=model.mask_embedding.dtype
    )
    reader.load_model(model)
    return model.eval(), tokenizer, manifest


def train(run, settings, data_ref, recovery_ref, progress, *, stop_after=None):
    policy = Training(**settings["training"])
    with closing(BlockDataset(checked(data_ref), settings, run.meter.check)) as data:
        identity = canonical_hash(
            dict(model_family="block8_v1", configuration=settings["identity"], data=data.sha256)
        )
        reader = (
            CheckpointReader(checked(recovery_ref), identity=identity, check=run.meter.check)
            if recovery_ref
            else None
        )
        admission(run, settings, training=True)
        torch.manual_seed(settings["seed"])
        model = load_model(settings)
        model.train()
        optimizer = TiledAdamW(
            model.named_parameters(),
            lr=policy.learning_rate,
            new_lr=policy.mask_learning_rate,
            weight_decay=policy.weight_decay,
            tile_elements=policy.optimizer_tile_elements,
            seed=policy.optimizer_seed,
        )
        sampler = random.Random(settings["seed"])
        metadata = dict(
            model_family="block8_v1",
            settings_identity=settings["identity"],
            architecture=settings["architecture"],
            data_sha256=data.sha256,
            numerical_policy=optimizer.precision_contract(),
        )
        state = dict(
            step=0, best_loss=None, selected=None, scheduler={"schema": "constant_separate_rates_v1"}
        )
        if reader:
            if reader.materialize(reader.field("metadata")) != metadata:
                raise ValueError("block8 recovery metadata differs")
            reader.load_model(model)
            reader.load_optimizer(optimizer)
            state = reader.materialize(reader.field("training"))
            restore_rng(reader.materialize(reader.field("rng")), sampler)
            checked(state["selected"])
        recoveries, recovery = [], recovery_ref

        def checkpoint():
            nonlocal recovery
            pointer = save_checkpoint(
                run.path / "checkpoints",
                recovery_payload(model, optimizer, identity, metadata, state, sampler),
                meter=run.meter,
                shard_bytes=policy.shard_bytes,
            )
            recovery = reference(pointer["manifest"])
            progress(recovery, state["step"])
            recoveries.append((Path(pointer["manifest"]).parent, state["selected"]))
            if len(recoveries) > 2:
                shutil.rmtree(recoveries.pop(0)[0])
            prune_exports()

        def prune_exports():
            retained = {Path(r["manifest"]).parent for _, r in recoveries}
            retained.add(Path(state["selected"]["manifest"]).parent)
            for folder in (run.path / "exports").glob("step-*"):
                if folder not in retained:
                    shutil.rmtree(folder)

        def selection():
            score = validate(model, data, run, policy)
            if state["best_loss"] is None or score < state["best_loss"]:
                selected = export(model, settings, data, run, state["step"], score)
                state.update(best_loss=score, selected=selected)
                # Keep every selected export referenced by either retained recovery.
                # A resumed failed attempt and all its exports are preserved.
                prune_exports()
            return score

        if reader is None:
            selection()
            checkpoint()
        while state["step"] < policy.steps:
            size = policy.block_at(state["step"])
            optimizer.zero_grad()
            totals = [0.0, 0.0, 0.0]
            for _ in range(policy.accumulation):
                key = sampler.choice(data.ids["train"])
                parent = data.parent(key)
                start = sampler.randrange((len(parent.answer_ids) + size - 1) // size) * size
                context, answer = data.view(key, start, size)
                charge_example(run, context, answer, size, auxiliary=bool(policy.ar_weight))
                loss, denoising, ar = example_loss(model, context, answer, size, sampler, policy)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite block8 training loss")
                (loss / policy.accumulation).backward()
                totals = [
                    x + float(y.detach()) / policy.accumulation for x, y in zip(totals, (loss, denoising, ar))
                ]
            report = optimizer.check_gradients()
            report.clip_(policy.gradient_clip)
            optimizer.step(report)
            optimizer.zero_grad()
            state["step"] += 1
            metric = dict(
                step=state["step"],
                block_size=size,
                loss=totals[0],
                masked_ce=totals[1],
                ar_ce=totals[2],
                gradient_norm=report.norm,
            )
            if state["step"] % policy.validation_interval == 0 or state["step"] == policy.steps:
                metric["dev_fully_masked_ce"] = selection()
            charge_write(run, 4096)
            with (run.path / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
            if (
                state["step"] % policy.checkpoint_interval == 0
                or state["step"] == policy.steps
                or stop_after == state["step"]
            ):
                checkpoint()
            run.meter.check_disk()
            if stop_after == state["step"] and state["step"] < policy.steps:
                return dict(
                    status="incomplete",
                    completed_step=state["step"],
                    recovery=recovery,
                    selected=state["selected"],
                )
        # At-target recovery performs no extra optimizer update.
        return dict(
            status="completed",
            completed_step=state["step"],
            recovery=recovery,
            selected=state["selected"],
            best_dev_fully_masked_ce=state["best_loss"],
            selected_step=read(checked(state["selected"]))["selected_step"],
            full_backbone_trainable=all(p.requires_grad for p in model.backbone.parameters()),
            loss_scope="equal microbatch objectives; masked-token mean plus weighted shifted AR mean; dev is mean fixed-view CE, not AR perplexity",
        )
