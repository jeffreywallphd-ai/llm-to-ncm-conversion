"""Native codec fitting and hierarchy training with sealed offline supervision."""

from contextlib import closing
import json
import math
from pathlib import Path
import random
import shutil
import torch
from ncm_lab.block8.config import Architecture, Training, CodecTraining
from ncm_lab.block8.data import BlockDataset
from ncm_lab.block8.model import load_model
from ncm_lab.block8.runtime import admission, checked, charge_write
from ncm_lab.block8.targets import TensorStore, FEATURES, TARGETS, fixed_views
from ncm_lab.codec.model import ContinuousCodec
from ncm_lab.embedded.geometry import masked_pool
from ncm_lab.runtime.io import canonical_hash, digest, reference, read, write_json
from ncm_lab.teacher.model import load_tokenizer, load_frozen_head
from ncm_lab.training.checkpoints import (
    CheckpointReader,
    INFERENCE,
    recovery_payload,
    restore_rng,
    save_checkpoint,
)
from ncm_lab.training.losses import chunked_ce, auxiliary_loss, codec_loss
from ncm_lab.training.optimizer import TiledAdamW
from ncm_lab.stage3.schedule import Schedule
from ncm_lab.monitoring.step_progress import StepEstimate, format_step

CODEC = "hierarchical_block8_selected_codec_v1"
EXPORT = "hierarchical_block8_generator_export_v1"


def padded_view(data, store, key, start, device, dtype):
    parent = data.parent(key)
    context, answer = data.view(key, start)
    raw = store.get(key, parent)[start : start + 8]
    values = torch.zeros(1, 8, raw.shape[-1], device=device, dtype=dtype)
    values[0, : len(answer)] = raw.to(device=device, dtype=dtype)
    labels = torch.full((1, 8), -100, device=device, dtype=torch.long)
    labels[0, : len(answer)] = torch.tensor(answer, device=device)
    return context, answer, values, labels, labels.ne(-100)


def example_loss(model, context, answer, fine_targets, scales, policy):
    """Reference block, masks and frozen codes never enter the native prompt-only API."""
    device = next(model.parameters()).device
    ids = torch.tensor([context], device=device, dtype=torch.long)
    labels = torch.full((1, 8), -100, device=device, dtype=torch.long)
    labels[0, : len(answer)] = torch.tensor(answer, device=device)
    valid = labels.ne(-100)
    output = model(ids, torch.ones_like(ids, dtype=torch.bool), return_aux=True)
    text = chunked_ce(model.backbone.lm_head, output.answer_states, labels, valid, policy.head_chunk)
    fine_targets = fine_targets.detach().to(dtype=next(model.parameters()).dtype)
    latent, levels, counts = auxiliary_loss(
        output.cumulative_codes, fine_targets, valid, model.architecture.resolutions, scales
    )
    ar = text.new_zeros(())
    if policy.ar_weight:
        from ncm_lab.embedded.cache import causal_text_states

        clean = torch.tensor([context + answer[:-1]], device=device)
        states = causal_text_states(
            model.backbone,
            clean,
            checkpoint_layers=model.training and model.architecture.activation_checkpointing,
        )[:, len(context) - 1 :]
        truth = torch.tensor([answer], device=device)
        ar = chunked_ce(
            model.backbone.lm_head, states, truth, torch.ones_like(truth, dtype=torch.bool), policy.head_chunk
        )
    return (
        text + policy.auxiliary_weight * latent + policy.ar_weight * ar,
        dict(text_ce=text, latent_loss=latent, ar_ce=ar),
        counts,
    )


def _save_candidate(model, settings, data, run, family, step, score, extra):
    folder = run.path / "exports" / f"step-{step:09d}"
    folder.mkdir(parents=True)
    metadata = dict(
        model_family=family,
        architecture=settings["architecture"],
        block_contract=settings["block_contract"],
        model_spec=settings["model"],
        dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
        implementation_sha256=settings["implementation_sha256"],
        selected_step=step,
        selection_loss=score,
        data_sha256=data.sha256,
        template_sha256=data.manifest["template_sha256"],
        **extra,
    )
    if family == EXPORT:
        tokenizer = load_tokenizer(settings["model"])
        if canonical_hash(tokenizer.chat_template) != data.manifest["template_sha256"]:
            raise ValueError("hierarchical training/deployment template differs")
        run.meter.check_disk(64000000)
        charge_write(run, 64000000)
        tokenizer.save_pretrained(folder / "tokenizer")
        files = {p.name: digest(p) for p in (folder / "tokenizer").iterdir() if p.is_file()}
        if sum(p.stat().st_size for p in (folder / "tokenizer").iterdir()) > 64000000:
            raise ValueError("tokenizer exceeds declared export allowance")
        metadata["backbone_config"] = model.backbone.config.to_dict()
    else:
        files = {}
        metadata["hidden_size"] = model.spec["hidden_size"]
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    identity = canonical_hash(metadata)
    saved = save_checkpoint(
        folder / "weights",
        dict(identity=identity, metadata=metadata, model=model.state_dict(keep_vars=True)),
        kind=INFERENCE,
        meter=run.meter,
        shard_bytes=settings["training" if family == EXPORT else "codec"]["shard_bytes"],
    )
    manifest = dict(
        schema=family,
        identity=identity,
        metadata=metadata,
        weights=reference(saved["manifest"]),
        tokenizer_files=files,
        selected_step=step,
        teacher_codec_or_verifier_required=False,
    )
    write_json(folder / "manifest.json", manifest, exclusive=True)
    return reference(folder / "manifest.json")


def _candidate_reader(path, family, check=None):
    path = Path(path)
    manifest = read(path)
    if manifest.get("schema") != family or manifest.get("metadata", {}).get("model_family") != family:
        raise ValueError(
            "hierarchical inference family required; custom baseline/native recovery artifacts are incompatible"
        )
    if canonical_hash(manifest["metadata"]) != manifest["identity"]:
        raise ValueError("hierarchical export metadata identity changed")
    weights = checked(manifest["weights"])
    if not weights.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError("hierarchical weights escape export directory")
    reader = CheckpointReader(weights, expected_kind=INFERENCE, identity=manifest["identity"], check=check)
    if reader.materialize(reader.field("metadata")) != manifest["metadata"]:
        raise ValueError("hierarchical shard metadata differs")
    return manifest, reader


def load_export(path, device, *, check=None):
    from transformers import AutoTokenizer
    from ncm_lab.embedded.model import from_metadata
    from ncm_lab.teacher.model import validate_model_spec
    from transformers import Qwen3Config
    from ncm_lab.block8.config import BLOCK_CONTRACT

    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    manifest, reader = _candidate_reader(path, EXPORT, check)
    metadata = manifest["metadata"]
    arch = Architecture(**metadata["architecture"])
    if arch.canvas != 8 or metadata["block_contract"] != BLOCK_CONTRACT:
        raise ValueError("hierarchical eight-token export contract required")
    validate_model_spec(metadata["model_spec"], Qwen3Config(**metadata["backbone_config"]))
    folder = path.parent / "tokenizer"
    for name, sha in manifest["tokenizer_files"].items():
        p = (folder / name).resolve()
        if p.parent != folder.resolve() or digest(p, check) != sha:
            raise ValueError("hierarchical tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if canonical_hash(tokenizer.chat_template) != metadata["template_sha256"]:
        raise ValueError("hierarchical prompt template changed")
    model = from_metadata(metadata, device)
    reader.load_model(model)
    model.set_phase("full")
    return model.eval(), tokenizer, manifest


def load_codec(path, settings, data, device, check=None):
    path = Path(path)
    receipt = read(path)
    if (
        receipt.get("schema") != CODEC
        or receipt["architecture"] != settings["architecture"]
        or receipt["data_sha256"] != data.sha256
        or receipt["settings_identity"] != settings["identity"]
    ):
        raise ValueError("incompatible hierarchical selected codec")
    candidate, reader = _candidate_reader(
        checked(receipt["selected_weights"]), "hierarchical_block8_codec_weights_v1", check
    )
    if (
        candidate["metadata"]["normalization"] != receipt["normalization"]
        or candidate["metadata"]["features_sha256"] != receipt["features_sha256"]
    ):
        raise ValueError("hierarchical codec statistics/source differ")
    model = ContinuousCodec(
        candidate["metadata"]["hidden_size"], Architecture(**settings["architecture"])
    ).to(device=device, dtype=torch.float32)
    reader.load_model(model)
    model.requires_grad_(False).eval()
    for values in receipt["scales"].values():
        if len(values) != model.architecture.latent_width or any(
            not math.isfinite(v) or v < 1e-3 for v in values
        ):
            raise ValueError("invalid frozen codec channel scales")
    if set(receipt["scales"]) != {str(i) for i in range(1, 32, 2)}:
        raise ValueError("all sixteen frozen-code scales required")
    return model, receipt


def _fit(
    run,
    settings,
    data,
    model,
    policy,
    identity,
    metadata,
    sampler,
    objective,
    validation,
    save_selected,
    recovery_ref,
    progress,
    *,
    stop_after=None,
    reader=None,
):
    optimizer = TiledAdamW(
        model.named_parameters(),
        lr=policy.learning_rate,
        new_lr=policy.new_learning_rate,
        weight_decay=policy.weight_decay,
        tile_elements=policy.optimizer_tile_elements,
        seed=policy.optimizer_seed,
    )
    schedule = Schedule(policy.steps, policy.warmup_steps, policy.minimum_lr_ratio)
    peaks = [group["lr"] for group in optimizer.groups]
    state = dict(step=0, best_loss=None, selected=None, scheduler=schedule.state(peaks, 0))
    if reader:
        if reader.materialize(reader.field("metadata")) != metadata:
            raise ValueError("hierarchical recovery metadata differs")
        reader.load_model(model)
        state = reader.materialize(reader.field("training"))
        if state["scheduler"] != schedule.state(peaks, state["step"]):
            raise ValueError("hierarchical recovery scheduler differs")
        if state["step"]:
            schedule.apply(optimizer, peaks, state["step"])
        reader.load_optimizer(optimizer)
        restore_rng(reader.materialize(reader.field("rng")), sampler)
        checked(state["selected"])
    recoveries, recovery = [], recovery_ref

    def prune():
        keep = {Path(ref["manifest"]).parent for _, ref in recoveries}
        keep.add(Path(state["selected"]["manifest"]).parent)
        for folder in (run.path / "exports").glob("step-*"):
            if folder not in keep:
                shutil.rmtree(folder)

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
        prune()

    def select():
        model.eval()
        with torch.no_grad():
            score = validation()
        model.train()
        if not math.isfinite(score):
            raise ValueError("nonfinite hierarchical dev selection objective")
        if state["best_loss"] is None or score < state["best_loss"]:
            state.update(best_loss=score, selected=save_selected(state["step"], score))
            prune()
        return score

    if reader is None:
        select()
        checkpoint()
    model.train()
    estimator = StepEstimate(policy.steps, state["step"])
    while state["step"] < policy.steps:
        optimizer.zero_grad()
        totals = {}
        for _ in range(policy.accumulation):
            loss, parts = objective(sampler)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite hierarchical training objective")
            (loss / policy.accumulation).backward()
            for name, value in parts.items():
                totals[name] = totals.get(name, 0) + float(value.detach()) / policy.accumulation
        report = optimizer.check_gradients()
        report.clip_(policy.gradient_clip)
        schedule.apply(optimizer, peaks, state["step"] + 1)
        optimizer.step(report)
        optimizer.zero_grad()
        state["step"] += 1
        state["scheduler"] = schedule.state(peaks, state["step"])
        metric = dict(
            step=state["step"], losses=totals, gradient_norm=report.norm, scheduler=state["scheduler"]
        )
        if state["step"] % policy.validation_interval == 0 or state["step"] == policy.steps:
            metric["dev_selection_loss"] = select()
        metric["progress"] = estimator.advance(state["step"])
        charge_write(run, 4096)
        with (run.path / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metric, allow_nan=False) + "\n")
        print(format_step(run.value["stage"], metric, policy.steps), flush=True)
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
    return dict(
        status="completed",
        completed_step=state["step"],
        recovery=recovery,
        selected=state["selected"],
        best_dev_loss=state["best_loss"],
        selected_step=read(checked(state["selected"]))["selected_step"],
    )


def _normalization(data, features, rows, run):
    count, total, squares = 0, 0.0, 0.0
    for key, start in rows:
        parent = data.parent(key)
        values = features.get(key, parent)[start : start + 8].double()
        run.meter.charge(source=len(parent.answer_ids), answer=values.shape[0], steps=1)
        count += values.numel()
        total += float(values.sum())
        squares += float(values.square().sum())
    if not count:
        raise ValueError("empty training-only codec statistics")
    return dict(
        variance=max(1e-6, squares / count - (total / count) ** 2),
        mean=total / count,
        elements=count,
        source="training only",
        views_sha256=canonical_hash(rows),
        views=len(rows),
    )


def fit_codec(run, settings, data_ref, features_ref, recovery_ref, progress, *, stop_after=None):
    policy = CodecTraining(**settings["codec"])
    with (
        closing(BlockDataset(checked(data_ref), settings, run.meter.check)) as data,
        closing(TensorStore(checked(features_ref), settings, data, FEATURES, run.meter.check)) as features,
    ):
        stats = fixed_views(data, features.ids["train"], policy.max_statistics_views, settings["seed"])
        dev = fixed_views(data, features.ids["dev"], policy.max_validation_views, 20261029)
        normalization = _normalization(data, features, stats, run)
        identity = canonical_hash(
            dict(family=CODEC, configuration=settings["identity"], data=data.sha256, features=features.sha256)
        )
        reader = (
            CheckpointReader(checked(recovery_ref), identity=identity, check=run.meter.check)
            if recovery_ref
            else None
        )
        admission(run, settings, training=True, kind="codec", data=data)
        torch.manual_seed(settings["seed"])
        head = load_frozen_head(settings["model"], settings["device"], torch.float32)
        hidden = head.weight.shape[1]
        model = ContinuousCodec(hidden, Architecture(**settings["architecture"])).to(
            settings["device"], dtype=torch.float32
        )
        sampler = random.Random(settings["seed"])
        extra = dict(normalization=normalization, features_sha256=features.sha256)
        metadata = dict(
            family=CODEC, configuration=settings["identity"], architecture=settings["architecture"], **extra
        )

        def loss_rows(rows, sampling, cutoff=None):
            states, valid = [], []
            for key, start in rows:
                parent = data.parent(key)
                _, answer, value, _, mask = padded_view(
                    data, features, key, start, settings["device"], torch.float32
                )
                run.meter.charge(source=len(parent.answer_ids), answer=len(answer), canvas=8, steps=1)
                states.append(value)
                valid.append(mask)
            loss, parts = codec_loss(
                model,
                head,
                torch.cat(states),
                torch.cat(valid),
                normalization["variance"],
                sampling,
                beta=policy.prediction_weight,
                max_head_positions=32,
                chunk_positions=policy.head_chunk,
                cutoff_index=cutoff,
            )
            return loss, dict(codec_loss=loss)

        def objective(sampling):
            rows = []
            for _ in range(policy.batch_size):
                key = sampling.choice(features.ids["train"])
                start = sampling.randrange((len(data.parent(key).answer_ids) + 7) // 8) * 8
                rows.append((key, start))
            return loss_rows(rows, sampling)

        def validation():
            return sum(
                float(loss_rows([row], random.Random(0), i % 31)[0]) for i, row in enumerate(dev)
            ) / len(dev)

        result = _fit(
            run,
            settings,
            data,
            model,
            policy,
            identity,
            metadata,
            sampler,
            objective,
            validation,
            lambda step, score: _save_candidate(
                model, settings, data, run, "hierarchical_block8_codec_weights_v1", step, score, extra
            ),
            recovery_ref,
            progress,
            stop_after=stop_after,
            reader=reader,
        )
        if result["status"] != "completed":
            return result
        candidate, chosen = _candidate_reader(
            checked(result["selected"]), "hierarchical_block8_codec_weights_v1", run.meter.check
        )
        chosen.load_model(model)
        model.requires_grad_(False).eval()
        totals, squares, counts = {}, {}, {}
        with torch.no_grad():
            for key, start in stats:
                parent = data.parent(key)
                _, answer, states, _, valid = padded_view(
                    data, features, key, start, settings["device"], torch.float32
                )
                run.meter.charge(source=len(parent.answer_ids), answer=len(answer), canvas=8, steps=1)
                fine = model.fine_codes(states)
                for level in range(1, 32, 2):
                    pooled, mask = masked_pool(fine, valid, model.architecture.resolutions[level])
                    values = pooled[mask].double()
                    totals[level] = totals.get(level, 0) + values.sum(0)
                    squares[level] = squares.get(level, 0) + values.square().sum(0)
                    counts[level] = counts.get(level, 0) + values.shape[0]
        scales = {
            str(k): (squares[k] / counts[k] - (totals[k] / counts[k]).square())
            .clamp_min(0)
            .sqrt()
            .clamp_min(1e-3)
            .float()
            .tolist()
            for k in totals
        }
        receipt = dict(
            schema=CODEC,
            architecture=settings["architecture"],
            settings_identity=settings["identity"],
            data_sha256=data.sha256,
            features_sha256=features.sha256,
            selected_weights=result["selected"],
            normalization=normalization,
            scales=scales,
            statistics_views_sha256=canonical_hash(stats),
            selected_step=result["selected_step"],
            teacher_head_frozen=not any(p.requires_grad for p in head.parameters()),
            train_only_statistics=True,
            coordinate_system="raw encoder codes; scales affect losses only",
        )
        write_json(run.path / "selected_codec.json", receipt, exclusive=True)
        return dict(**result, codec=reference(run.path / "selected_codec.json"), teacher_head_frozen=True)


def train(run, settings, data_ref, targets_ref, codec_ref, recovery_ref, progress, *, stop_after=None):
    policy = Training(**settings["training"])
    with (
        closing(BlockDataset(checked(data_ref), settings, run.meter.check)) as data,
        closing(TensorStore(checked(targets_ref), settings, data, TARGETS, run.meter.check)) as targets,
    ):
        if (
            targets.manifest["codec"] != codec_ref
            or targets.manifest["encoder_dtype"] != "float32"
            or targets.manifest["storage_dtype"] != "float32"
        ):
            raise ValueError("hierarchical targets/selected codec or precision differ")
        receipt = read(checked(codec_ref))
        if receipt.get("schema") != CODEC or receipt["data_sha256"] != data.sha256:
            raise ValueError("hierarchical selected codec receipt required")
        identity = canonical_hash(
            dict(
                family=EXPORT,
                configuration=settings["identity"],
                data=data.sha256,
                targets=targets.sha256,
                codec=codec_ref,
            )
        )
        reader = (
            CheckpointReader(checked(recovery_ref), identity=identity, check=run.meter.check)
            if recovery_ref
            else None
        )
        admission(run, settings, training=True, kind="generator", data=data)
        torch.manual_seed(settings["seed"])
        model = load_model(settings)
        sampler = random.Random(settings["seed"])
        metadata = dict(
            family=EXPORT,
            configuration=settings["identity"],
            architecture=settings["architecture"],
            targets_sha256=targets.sha256,
            codec=codec_ref,
            scales_sha256=canonical_hash(receipt["scales"]),
        )
        dev = fixed_views(data, targets.ids["dev"], policy.max_validation_views, 20261029)

        def objective(sampling):
            key = sampling.choice(targets.ids["train"])
            parent = data.parent(key)
            start = sampling.randrange((len(parent.answer_ids) + 7) // 8) * 8
            context, answer, fine, _, _ = padded_view(
                data, targets, key, start, settings["device"], getattr(torch, settings["dtype"])
            )
            run.meter.charge(
                source=len(context) + 8 + len(parent.answer_ids), answer=len(answer), canvas=8, steps=1
            )
            if policy.ar_weight:
                run.meter.charge(source=len(context) + len(answer) - 1, answer=len(answer), steps=1)
            loss, parts, counts = example_loss(model, context, answer, fine, receipt["scales"], policy)
            if len(counts) != 16:
                raise ValueError("all sixteen hierarchical cumulative losses required")
            return loss, parts

        def validation():
            total = 0.0
            for key, start in dev:
                context, answer = data.view(key, start)
                run.meter.charge(source=len(context) + 8, answer=len(answer), canvas=8, steps=1)
                ids = torch.tensor([context], device=settings["device"])
                output = model(ids, torch.ones_like(ids, dtype=torch.bool))
                labels = torch.full((1, 8), -100, device=settings["device"], dtype=torch.long)
                labels[0, : len(answer)] = torch.tensor(answer, device=settings["device"])
                total += float(
                    chunked_ce(
                        model.backbone.lm_head,
                        output.answer_states,
                        labels,
                        labels.ne(-100),
                        policy.head_chunk,
                    )
                )
            return total / len(dev)

        result = _fit(
            run,
            settings,
            data,
            model,
            policy,
            identity,
            metadata,
            sampler,
            objective,
            validation,
            lambda step, score: _save_candidate(
                model,
                settings,
                data,
                run,
                EXPORT,
                step,
                score,
                dict(codec=codec_ref, targets_sha256=targets.sha256),
            ),
            recovery_ref,
            progress,
            stop_after=stop_after,
            reader=reader,
        )
        result.update(
            full_backbone_trainable=all(p.requires_grad for p in model.backbone.parameters()),
            hierarchy_updates=32,
            cumulative_supervision_levels=16,
            teacher_or_codec_loaded_during_generator=False,
            loss_scope="native token CE + weighted mean cumulative-code loss + weighted causal AR CE; equal actual microbatch objectives",
        )
        return result
