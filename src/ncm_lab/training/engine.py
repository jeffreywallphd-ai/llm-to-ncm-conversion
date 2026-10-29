"""Connected bounded native codec/generator training, held-out selection and recovery."""

from pathlib import Path
import gc
import random
import time
import torch
from ncm_lab.codec.model import ContinuousCodec
from ncm_lab.config.architecture import Architecture
from ncm_lab.data.artifacts import Dataset, Cursor
from ncm_lab.data.records import collate, ALIGNMENT
from ncm_lab.embedded.geometry import masked_pool
from ncm_lab.runtime.io import canonical_hash, write_json, digest, implementation_hash, atomic_json, read
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.teacher.model import load_generator, load_frozen_head, model_identity
from ncm_lab.training.optimizer import TiledAdamW
from ncm_lab.training.losses import chunked_ce, auxiliary_loss, codec_loss
from ncm_lab.training.validation_plan import codec_validation_plan
from ncm_lab.monitoring.records import Accumulator, Measure
from ncm_lab.monitoring.metrics import quantities
from ncm_lab.monitoring.diagnostics import Activity, gradients
from ncm_lab.training.checkpoints import (
    CheckpointReader,
    INFERENCE,
    save_checkpoint,
    recovery_payload,
    native_identity,
    restore_rng,
)


def padded_tensors(dataset, indices, canvas, device, dtype):
    rows = [dataset[i] for i in indices]
    examples = [row[0] for row in rows]
    if any(row[1] is None for row in rows):
        raise ValueError("missing precomputed targets; online teacher fallback is forbidden")
    width = rows[0][1].shape[-1]
    data = torch.zeros(len(rows), canvas, width, device=device, dtype=dtype)
    for i, (_, tensor) in enumerate(rows):
        data[i, : tensor.shape[0]] = tensor.to(device=device, dtype=dtype)
    ids, prompt_mask, labels, valid = collate(examples, canvas, device)
    return examples, ids, prompt_mask, labels, valid, data


def state_normalization(dataset, indices, meter):
    count = 0
    total = 0.0
    squares = 0.0
    for index in indices:
        meter.check()
        _, tensor = dataset[index]
        values = tensor.double()
        count += values.numel()
        total += float(values.sum())
        squares += float(values.square().sum())
    if not count:
        raise ValueError("empty training normalization source")
    return {
        "variance": max(1e-6, squares / count - (total / count) ** 2),
        "mean": total / count,
        "elements": count,
        "source": "training only",
        "training_indices_sha256": canonical_hash(indices),
    }


@torch.no_grad()
def latent_scales(codec, dataset, indices, meter):
    totals = {}
    squares = {}
    counts = {}
    device = next(codec.parameters()).device
    dtype = next(codec.parameters()).dtype
    for index in indices:
        meter.check()
        _, _, _, _, valid, states = padded_tensors(dataset, [index], codec.architecture.canvas, device, dtype)
        fine = codec.fine_codes(states)
        for level in range(1, len(codec.architecture.resolutions), 2):
            pooled, mask = masked_pool(fine, valid, codec.architecture.resolutions[level])
            data = pooled[mask].double()
            totals[level] = totals.get(level, 0) + data.sum(0)
            squares[level] = squares.get(level, 0) + data.square().sum(0)
            counts[level] = counts.get(level, 0) + data.shape[0]
    scales = {
        str(k): (squares[k] / counts[k] - (totals[k] / counts[k]).square())
        .clamp_min(0)
        .sqrt()
        .clamp_min(1e-3)
        .float()
        .tolist()
        for k in totals
    }
    return scales, {
        "valid_bins": counts,
        "positive_scale_floor": 1e-3,
        "coordinate_system": "raw encoder codes; scales used only in losses",
        "mean_channel_variance": {
            str(k): float((squares[k] / counts[k] - (totals[k] / counts[k]).square()).mean()) for k in totals
        },
    }


def architecture_from_metadata(metadata, device):
    # Allocate exactly one complete destination. No model/state-sized Python bytes copy.
    from ncm_lab.embedded.model import from_metadata

    return from_metadata(metadata, device)


def make_optimizer(model, options, generator):
    return TiledAdamW(
        model.named_parameters(),
        options["learning_rate"],
        options["new_learning_rate"] if generator else options["learning_rate"],
        options["weight_decay"],
        tile_elements=options.get("optimizer_tile_elements", 1048576),
        policy=options.get("optimizer_policy", "fp32_moments_bf16_stochastic_v1"),
        seed=options.get("optimizer_seed", 1729),
    )


def _representatives(model):
    names = [
        "backbone.model.embed_tokens.weight",
        "backbone.model.layers.0.self_attn.q_proj.weight",
        "backbone.model.layers.8.mlp.down_proj.weight",
        "backbone.model.layers.35.self_attn.o_proj.weight",
        "backbone.model.norm.weight",
        "backbone.lm_head.weight",
        "queries",
        "refinements.0.down.weight",
        "refinements.15.up.weight",
        "gates",
    ]
    parameters = dict(model.named_parameters())
    result = {}
    for name in names:
        if name not in parameters:
            continue
        p = parameters[name]
        selected = p[1] if "embed_tokens" in name else p
        result[name] = selected.detach().flatten()[:8].clone()
    return result


def prune(folder, selected):
    """Retain two newest recovery versions plus selected; interrupted writes survive."""
    import shutil

    folder = Path(folder).resolve()
    candidates = sorted(folder.glob("version-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    protected = {p.resolve() for p in candidates[:2]}
    if selected:
        protected.add(Path(selected["manifest"]).parent.resolve())
    for path in candidates[2:]:
        resolved = path.resolve()
        if resolved in protected:
            continue
        if path.is_symlink() or resolved.parent != folder:
            raise ValueError("unsafe checkpoint retention path")
        shutil.rmtree(resolved)


def fit(run, value, bound, *, resume=None, stop_after=None):
    options = value["training"]
    architecture = Architecture(**value["architecture"])
    torch.manual_seed(options["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options["seed"])
    generator = value["stage"] in {"generator", "profile"}
    key = "targets" if generator else "features"
    dataset = Dataset(bound[key]["manifest"], "native_targets_v1" if generator else "native_features_v1")
    source = dataset.manifest["identity"]
    if source.get("state_alignment") != ALIGNMENT:
        raise ValueError("native prediction-state alignment mismatch")
    if source["model"] != model_identity(value["model"]):
        raise ValueError("training teacher/model identity mismatch")
    if source.get("canvas", source.get("architecture", {}).get("canvas")) != architecture.canvas:
        raise ValueError("training canvas differs from native source cache")
    train = dataset.indices("train")
    dev = dataset.indices("dev")
    if not train or not dev:
        raise ValueError("held-out native train/dev data required")
    if generator and source["architecture"] != architecture.to_dict():
        raise ValueError("latent target architecture changed; rebuild explicitly")
    unique_answers = sum(len(dataset.metadata(i)["example"]["answer_ids"]) for i in train)
    if unique_answers < options["minimum_unique_answer_positions"]:
        raise ValueError("requested unique answer pool unavailable; repetition cannot fill it")
    normalization = (
        source.get("normalization") if generator else state_normalization(dataset, train, run.meter)
    )
    dtype = getattr(torch, value["dtype"])
    device = value["device"]
    restore_started = time.perf_counter()
    restored_checkpoint_ref = None
    reader = CheckpointReader(resume, check=run.meter.check) if resume else None
    previous_metadata = reader.materialize(reader.field("metadata")) if reader else None
    if (
        previous_metadata
        and previous_metadata.get("optimizer_precision", {}).get("policy")
        != "fp32_moments_bf16_stochastic_v1"
    ):
        raise ValueError(
            "incompatible optimizer numerical policy: legacy nearest-rounded recovery cannot resume under stochastic BF16 policy"
        )
    head = None
    if generator:
        model = (
            architecture_from_metadata(previous_metadata, device)
            if reader
            else load_generator(value["model"], value["architecture"], device, dtype)
        )
    else:
        hidden = dataset.manifest["extra"]["hidden_size"]
        model = ContinuousCodec(hidden, architecture).to(device=device, dtype=dtype)
        head = load_frozen_head(value["model"], device, dtype)
    optimizer = make_optimizer(model, options, generator)
    validation_plan = (
        None
        if generator
        else codec_validation_plan(
            dataset,
            dev,
            architecture.resolutions,
            options["max_validation_examples"],
            options.get("validation_seed", 20261029),
        )
    )
    metadata = {
        "component": "generator" if generator else "codec",
        "architecture": architecture.to_dict(),
        "model": model_identity(value["model"]),
        "backbone_config": model.backbone.config.to_dict() if generator else None,
        "dataset_sha256": dataset.sha256,
        "source": source,
        "dtype": value["dtype"],
        "normalization": normalization,
        "optimizer_precision": optimizer.precision_contract(),
        "selection_policy": validation_plan
        if not generator
        else {
            "policy": "generator_equal_example_loss_v1",
            "indices": dev[: options["max_validation_examples"]],
        },
        "training_contract": {
            k: v
            for k, v in options.items()
            if k
            not in {
                "steps",
                "epochs",
                "validation_interval",
                "checkpoint_interval",
                "recovery_probe",
                "max_validation_examples",
            }
        },
        "phase_id": value["budget"]["phase_id"],
        "implementation_sha256": implementation_hash(),
        "runtime": {
            "torch": str(torch.__version__),
            "device": device,
            "attention": "explicit-mask SDPA; no AR KV reuse",
        },
    }
    if generator:
        metadata["backbone_config"].pop("_name_or_path", None)
    identity = native_identity(metadata)
    sampler = random.Random(options["seed"])
    state = {
        "optimizer_steps": 0,
        "processed_source_tokens": 0,
        "answer_position_exposures": 0,
        "allocated_canvas_positions": 0,
        "backbone_answer_positions": 0,
        "phase": "warmup" if generator and options["warmup_processed_tokens"] else "full",
        "best_dev_loss": None,
        "selected": None,
    }
    if reader:
        if reader.manifest["identity"] != identity:
            raise ValueError("native recovery config/data/codec/optimizer identity mismatch")
        reader.load_model(model)
        reader.load_optimizer(optimizer)
        state = reader.materialize(reader.field("training"))
        restore_rng(reader.materialize(reader.field("rng")), sampler)
        if state.get("selected_this_step"):
            state["selected"] = {
                "schema": "native_checkpoint_pointer_v1",
                "manifest": str(reader.path),
                "sha256": digest(reader.path),
            }
        restored_checkpoint_ref = {"path": str(reader.path), "sha256": digest(reader.path)}
        reader = None
    cursor = Cursor(train, sampler, state.get("cursor"))
    if generator:
        model.set_phase(state["phase"])
    model.train()
    initial_representatives = _representatives(model) if generator else {}
    write_json(
        run.path / "parameter_groups.json",
        {
            "groups": optimizer.groups,
            "parameters": {
                n: {"elements": p.numel(), "trainable": p.requires_grad, "dtype": str(p.dtype)}
                for n, p in model.named_parameters()
            },
            "policy": optimizer.precision_contract(),
        },
        exclusive=True,
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    restored_peak = None
    before_restore_peak = None
    stop_reason = None
    recorder = run.recorder
    recorder.step = state["optimizer_steps"]
    recorder.phase = state["phase"]
    recorder.lineage(
        state.get("attempt_id"),
        restored_checkpoint_ref,
        state,
    )
    if resume:
        recorder.event(
            "checkpoint_restored",
            duration_seconds=time.perf_counter() - restore_started,
            phase="checkpointing",
            checkpoint=restored_checkpoint_ref,
        )
    state["attempt_id"] = recorder.attempt
    interval = Accumulator()
    interval_start = state["optimizer_steps"] + 1
    sample_prompts = []
    tokenizer = None
    if generator and recorder.options.enabled and recorder.options.sample_prompts:
        from ncm_lab.monitoring.samples import fixed_prompts
        from ncm_lab.teacher.model import load_tokenizer

        sample_prompts = fixed_prompts(dataset, dev, recorder.options.sample_prompts)
        tokenizer = load_tokenizer(value["model"])

    def objective(indices, training=True, planned=None):
        detail = {"groups": [], "statistics_enabled": recorder.detail_due(recorder.step)}
        examples, ids, mask, labels, valid, tensor = padded_tensors(
            dataset, indices, architecture.canvas, device, dtype
        )
        source_positions = sum(
            len(e.prompt_ids) + len(e.answer_ids) if generator else len(e.answer_ids) for e in examples
        )
        # Forward work is charged even when the later update fails or is interrupted.
        run.meter.charge(
            source=source_positions, answer=int(valid.sum()), canvas=len(examples) * architecture.canvas
        )
        if generator:
            output = model(ids, mask, return_aux=True)
            text = chunked_ce(
                model.backbone.lm_head, output.answer_states, labels, valid, options["head_chunk"]
            )
            latent, groups, counts = auxiliary_loss(
                output.cumulative_codes,
                tensor,
                valid,
                architecture.resolutions,
                source["scales"],
                detail["groups"] if recorder.options.enabled else None,
            )
            loss = text + options["auxiliary_weight"] * latent
            metrics = {
                "loss": float(loss.detach()),
                "text_loss": float(text.detach()),
                "latent_loss": float(latent.detach()),
                "groups": [float(x.detach()) for x in groups],
                "group_valid_bins": counts,
                "backbone_answer_positions": sum(t["answer_positions"] for t in output.trace) * len(examples),
            }
        else:
            loss, metrics = codec_loss(
                model,
                head,
                tensor,
                valid,
                normalization["variance"],
                sampler if training else random.Random(planned["head_seed"]),
                options["prediction_weight"],
                options["head_positions"],
                options["head_chunk"],
                detail if recorder.options.enabled else None,
                cutoff_index=None if training else planned["cutoff_index"],
            )
        observations = (
            quantities(
                metrics,
                detail,
                generator=generator,
                auxiliary_weight=options["auxiliary_weight"],
                answer_count=int(valid.sum()),
                stage="generator" if generator else "codec",
                split="train" if training else "dev",
            )
            if recorder.options.enabled
            else {}
        )
        return loss, metrics, source_positions, int(valid.sum()), observations

    def evaluate():
        started = time.perf_counter()
        recorder.event("validation_started")
        prior_phase = recorder.phase
        recorder.phase = "validation"
        observed = Accumulator()
        prior = model.training
        model.eval()
        rows = []
        with torch.no_grad():
            assignments = (
                [{"index": i} for i in dev[: options["max_validation_examples"]]]
                if generator
                else validation_plan["assignments"]
            )
            for planned in assignments:
                _, metrics, _, _, observations = objective([planned["index"]], False, planned)
                observed.add(observations)
                rows.append(metrics)
        model.train(prior)
        if recorder.options.enabled:
            recorder.metric(
                observed.finish(),
                split="dev",
                interval_start_step=recorder.step,
                interval_end_step=recorder.step,
                normalization_sha256=canonical_hash(source.get("scales", normalization)),
                scale_floor=1e-3,
            )
        recorder.phase = prior_phase
        recorder.event("validation_completed", duration_seconds=time.perf_counter() - started)
        score = sum(row["loss"] for row in rows) / len(rows)
        recorder.event(
            "selection_evaluated",
            score=score,
            plan_sha256=canonical_hash(validation_plan),
            policy="generator_equal_example_loss_v1" if generator else validation_plan["policy"],
            per_example_components=rows,
        )
        return score

    try:
        while (
            state["optimizer_steps"] < options["steps"]
            and cursor.epoch < options["epochs"]
            and not (cursor.position == len(cursor.order) and cursor.epoch + 1 >= options["epochs"])
        ):
            if state["processed_source_tokens"] >= options["max_processed_tokens"]:
                raise BudgetStop("declared processed-token endpoint reached")
            optimizer.zero_grad()
            step_started = time.perf_counter()
            aggregate = []
            step_source_count = step_answer_count = 0
            step_observations = Accumulator()
            recorder.step = state["optimizer_steps"] + 1
            executed_phase = state["phase"]
            recorder.phase = executed_phase
            executed_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            executed_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            activity = Activity() if generator and recorder.detail_due(recorder.step) else None
            remaining = (options["epochs"] - cursor.epoch) * len(cursor.order) - cursor.position
            microbatches = min(
                options["accumulation"], (remaining + options["batch_size"] - 1) // options["batch_size"]
            )
            run.meter.charge(steps=1)
            for _ in range(microbatches):
                indices = cursor.take(options["batch_size"], options["epochs"])
                if not indices:
                    break
                if generator:
                    model.diagnostic_observer = activity
                try:
                    loss, metrics, source_count, answer_count, observations = objective(indices)
                finally:
                    if generator:
                        model.diagnostic_observer = None
                step_observations.add(observations)
                if not bool(torch.isfinite(loss)):
                    if recorder.options.enabled:
                        recorder.metric(observations, split="diagnostic", attempted_update=True)
                    raise ValueError("nonfinite native training loss")
                (loss / microbatches).backward()
                state["processed_source_tokens"] += source_count
                state["answer_position_exposures"] += answer_count
                step_source_count += source_count
                step_answer_count += answer_count
                state["allocated_canvas_positions"] += len(indices) * architecture.canvas
                state["backbone_answer_positions"] += metrics.get("backbone_answer_positions", 0)
                aggregate.append(metrics)
            report = optimizer.check_gradients()
            norm = report.norm
            diagnostic_gradients = gradients(model) if recorder.detail_due(recorder.step) else {}
            limit = options.get("gradient_clip", 1.0)
            report.clip_(limit)
            optimizer.step(report)
            del report  # Its parameter/gradient ownership must end before recovery allocation.
            state["optimizer_steps"] += 1
            state["last_executed_phase"] = executed_phase
            phase_boundary = (
                generator
                and executed_phase == "warmup"
                and state["processed_source_tokens"] >= options["warmup_processed_tokens"]
            )
            if phase_boundary:
                state["phase"] = "full"
            metrics = {
                **aggregate[-1],
                "step": state["optimizer_steps"],
                "microbatches": len(aggregate),
                "phase": executed_phase,
                "next_phase": state["phase"],
                "trainable_parameters": executed_trainable,
                "frozen_parameters": executed_frozen,
                "gradient_norm": norm,
                "optimizer_update": optimizer.last_update_summary,
                "step_seconds": time.perf_counter() - step_started,
                "processed_source_tokens": state["processed_source_tokens"],
                "unique_available_answer_positions": unique_answers,
                "answer_position_exposures": state["answer_position_exposures"],
                "allocated_canvas_positions": state["allocated_canvas_positions"],
            }
            observed = step_observations.finish()
            # First average the trainer's equally weighted microbatch objective;
            # then weight completed optimizer steps equally across an interval.
            observed = {
                name: Measure(
                    row.value,
                    1,
                    row.unit,
                    "equal completed optimizer steps; each objective averages its actual microbatches",
                    row.status,
                )
                if "/loss/" in name or "weighted_contribution" in name
                else row
                for name, row in observed.items()
            }
            if not generator and recorder.options.enabled:
                for cutoff in range(len(architecture.resolutions) - 1):
                    observed.setdefault(
                        f"codec/train/cutoff_{cutoff + 1:02d}/reconstruction_normalized_mse", Measure(None, 0)
                    )
            interval.add(observed)
            if recorder.due(
                recorder.step,
                final=phase_boundary
                or state["optimizer_steps"] == options["steps"]
                or stop_after is not None
                and state["optimizer_steps"] >= stop_after
                or cursor.position == len(cursor.order)
                and cursor.epoch + 1 >= options["epochs"],
            ):
                from ncm_lab.runtime.memory import observe

                values = interval.finish()
                values.update(diagnostic_gradients)
                for i, row in enumerate(activity.rows if activity else []):
                    for name, number in row.items():
                        if name not in {"update", "resolution"}:
                            values[f"hierarchy/update_{row['update']:02d}/{name}"] = Measure(
                                number,
                                reduction="last sampled logical forward; all inference-visible slots; gated updates",
                            )
                seconds = metrics["step_seconds"]
                values.update(
                    {
                        "resources/step_wall_seconds": Measure(
                            seconds,
                            unit="seconds",
                            reduction="host elapsed optimizer step; not CUDA device-event timing",
                        ),
                        "resources/source_tokens_per_step_second": Measure(
                            step_source_count / seconds,
                            unit="tokens/second",
                            reduction="all step nonpadding source tokens / host step wall",
                        ),
                        "resources/answer_tokens_per_step_second": Measure(
                            step_answer_count / seconds,
                            unit="tokens/second",
                            reduction="all step supervised tokens / host step wall",
                        ),
                        "resources/global_gradient_l2": Measure(norm),
                        "resources/processed_source_tokens": Measure(
                            state["processed_source_tokens"], unit="tokens"
                        ),
                        "resources/answer_position_exposures": Measure(
                            state["answer_position_exposures"], unit="tokens"
                        ),
                        "resources/allocated_canvas_positions": Measure(
                            state["allocated_canvas_positions"], unit="positions"
                        ),
                        "resources/budget_wall_seconds_used": Measure(
                            time.perf_counter() - run.meter.started, unit="seconds"
                        ),
                        "resources/budget_wall_seconds_remaining": Measure(
                            max(
                                0,
                                run.meter.budget.max_wall_seconds - (time.perf_counter() - run.meter.started),
                            ),
                            unit="seconds",
                        ),
                        "resources/trainable_parameters": Measure(executed_trainable),
                        "resources/frozen_parameters": Measure(executed_frozen),
                    }
                )
                for name, number in observe().items():
                    if name != "scope":
                        values["resources/" + name] = Measure(number, unit="bytes")
                phase_ledger = read(run.meter.path)
                for name, number in run.meter.used.items():
                    values["budgets/run/" + name + "/charged"] = Measure(number)
                    values["budgets/run/" + name + "/remaining"] = Measure(
                        getattr(run.meter.budget, "max_" + name) - number
                    )
                    phase_number = phase_ledger["charges"][name]
                    values["budgets/phase/" + name + "/charged"] = Measure(phase_number)
                    values["budgets/phase/" + name + "/remaining"] = Measure(
                        getattr(run.meter.budget, "max_phase_" + name) - phase_number
                    )
                values["budgets/phase/wall_seconds_reserved"] = Measure(
                    phase_ledger["wall"],
                    unit="seconds",
                    reduction="conservative active wall reservations plus completed actual charges",
                )
                values["resources/run_storage_bytes"] = Measure(
                    sum(p.stat().st_size for p in run.path.rglob("*") if p.is_file()),
                    unit="bytes",
                    reduction="observed current run including retained checkpoint versions",
                )
                for group in optimizer.groups:
                    values["optimizer/learning_rate/" + group["name"]] = Measure(group["lr"])
                for name in ("intended_fp32_l2", "stored_parameter_l2", "scalar_elements"):
                    values["optimizer/update/" + name] = Measure(optimizer.last_update_summary[name])
                for name in (
                    "memory_allocated",
                    "memory_reserved",
                    "max_memory_allocated",
                    "max_memory_reserved",
                ):
                    values["resources/gpu_" + name] = Measure(
                        getattr(torch.cuda, name)() if device == "cuda" else None, unit="bytes"
                    )
                recorder.metric(
                    values,
                    interval_start_step=interval_start,
                    interval_end_step=recorder.step,
                    microbatches=len(aggregate),
                    accumulation=options["accumulation"],
                    amp_scaler="not used",
                    skipped_update=False,
                    normalization_sha256=canonical_hash(source.get("scales", normalization)),
                    scale_floor=1e-3,
                    hierarchy_logical_observations=activity.logical_count if activity else 0,
                )
                interval_start = recorder.step + 1
            selection = None
            if state["optimizer_steps"] % options["validation_interval"] == 0:
                metrics["dev_loss"] = evaluate()
                selection = metrics["dev_loss"]
                if generator and sample_prompts and recorder.step % recorder.options.sample_interval == 0:
                    from ncm_lab.monitoring.samples import observe_samples

                    sample_started = time.perf_counter()
                    recorder.event("sampling_started")
                    observe_samples(
                        model,
                        tokenizer,
                        sample_prompts,
                        recorder,
                        run.meter,
                        model_identity(value["model"])["eos_ids"],
                    )
                    recorder.event(
                        "sampling_completed", duration_seconds=time.perf_counter() - sample_started
                    )
            run.metric(metrics)
            if phase_boundary:
                run.event(
                    "phase_transition",
                    previous_phase=executed_phase,
                    next_phase=state["phase"],
                    last_completed_step=state["optimizer_steps"],
                    effective_from_step=state["optimizer_steps"] + 1,
                )
                model.set_phase(state["phase"])
            if (
                state["optimizer_steps"] % options["checkpoint_interval"] == 0
                or selection is not None
                or state["optimizer_steps"] == options["steps"]
            ):
                state["cursor"] = cursor.state()
                chosen = selection is not None and (
                    state["best_dev_loss"] is None or selection < state["best_dev_loss"]
                )
                state["selected_this_step"] = chosen
                if chosen:
                    state["best_dev_loss"] = selection
                save_started = time.perf_counter()
                recorder.event("checkpointing_started")
                saved = save_checkpoint(
                    run.path / "checkpoints",
                    recovery_payload(model, optimizer, identity, metadata, state, sampler),
                    meter=run.meter,
                    shard_bytes=options["shard_bytes"],
                )
                if chosen:
                    state["selected"] = saved
                    atomic_json(run.path / "checkpoints/selected.json", saved)
                prune(run.path / "checkpoints", state["selected"])
                recorder.event(
                    "checkpoint_completed",
                    manifest=saved["manifest"],
                    sha256=saved["sha256"],
                    duration_seconds=time.perf_counter() - save_started,
                )
                recorder.durable()
            if generator and options["recovery_probe"] and state["optimizer_steps"] == 2:
                if device == "cuda":
                    before_restore_peak = torch.cuda.max_memory_allocated()
                # The completed loss graph still owns leaf parameters after backward.
                # Release it before constructing the fresh recovery destination.
                del loss
                del model, optimizer
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                reader = CheckpointReader(saved["manifest"], identity=identity, check=run.meter.check)
                model = architecture_from_metadata(metadata, device)
                reader.load_model(model)
                optimizer = make_optimizer(model, options, True)
                reader.load_optimizer(optimizer)
                restore_rng(reader.materialize(reader.field("rng")), sampler)
                reader = None
                model.set_phase(state["phase"])
                model.train()
                restored_peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
                run.event("recovery_probe_restored", step=2, peak_allocated_bytes=restored_peak)
            if stop_after is not None and state["optimizer_steps"] >= stop_after:
                break
    except BudgetStop as exc:
        stop_reason = str(exc)
        if recorder.options.enabled and interval.values:
            recorder.step = state["optimizer_steps"]
            recorder.metric(
                interval.finish(),
                terminal=True,
                interval_start_step=interval_start,
                interval_end_step=recorder.step,
                final_available=True,
            )
    if not state["selected"]:
        raise ValueError("no completed held-out selection; recovery remains diagnostic only")
    # Export the held-out selected state without holding a second model/optimizer copy.
    selected_reader = CheckpointReader(
        state["selected"]["manifest"], identity=identity, check=run.meter.check
    )
    selected_reader.load_model(model)
    selected_reader = None
    if generator:
        from ncm_lab.inference.export import export_generator

        artifact = export_generator(
            model,
            metadata,
            run.path / "artifacts/generator",
            run.meter,
            options["shard_bytes"],
            value["model"],
        )
        changed = {
            n: bool(
                (
                    initial_representatives[n]
                    != (
                        dict(model.named_parameters())[n][1]
                        if "embed_tokens" in n
                        else dict(model.named_parameters())[n]
                    )
                    .detach()
                    .flatten()[:8]
                ).any()
            )
            for n in initial_representatives
        }
        diagnostics = {"representative_weights_changed": changed, "warmup_in_total_token_budget": True}
    else:
        model.eval().requires_grad_(False)
        scales, scale_statistics = latent_scales(model, dataset, train, run.meter)
        folder = run.path / "artifacts/codec"
        folder.mkdir(parents=True, exist_ok=True)
        weights = save_checkpoint(
            folder / "weights",
            {"identity": identity, "metadata": metadata, "model": model.state_dict(keep_vars=True)},
            kind=INFERENCE,
            meter=run.meter,
            shard_bytes=options["shard_bytes"],
        )
        artifact = folder / "manifest.json"
        write_json(
            artifact,
            {
                "schema": "native_codec_v1",
                "identity": identity,
                "spec": model.spec,
                "trained_dtype": value["dtype"],
                "weights_manifest": str(Path(weights["manifest"]).relative_to(folder)),
                "weights_manifest_sha256": weights["sha256"],
                "normalization": normalization,
                "scales": scales,
                "latent_statistics": scale_statistics,
                "source": source,
                "selection": {
                    "criterion": "held-out normalized reconstruction plus head consistency",
                    "best_dev_loss": state["best_dev_loss"],
                    "recovery": state["selected"],
                    "validation_plan": validation_plan,
                },
                "frozen": True,
            },
            exclusive=True,
        )
        diagnostics = {
            "latent_statistics": scale_statistics,
            "normalization": normalization,
            "head_frozen": not any(p.requires_grad for p in head.parameters()),
        }
    run.publish(artifact)
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    if before_restore_peak is not None:
        peak = max(peak, before_restore_peak)
    reserved = torch.cuda.max_memory_reserved() if device == "cuda" else None
    return {
        "artifact": str(artifact),
        "identity": identity,
        "optimizer_steps": state["optimizer_steps"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count_at_end": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_covered_parameter_count": sum(p.numel() for p in optimizer.parameters.values()),
        "optimizer_state_tensor_bytes": sum(
            v.numel() * v.element_size()
            for s in optimizer.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        ),
        "optimizer_precision_policy": optimizer.precision_contract(),
        "optimizer_rounding_draws": optimizer.rounding_draws,
        "last_optimizer_update": optimizer.last_update_summary,
        "last_executed_phase": state.get("last_executed_phase"),
        "selection_policy": validation_plan,
        "processed_source_tokens": state["processed_source_tokens"],
        "answer_position_exposures": state["answer_position_exposures"],
        "allocated_canvas_positions": state["allocated_canvas_positions"],
        "backbone_answer_positions": state["backbone_answer_positions"],
        "unique_available_answer_positions": unique_answers,
        "best_dev_loss": state["best_dev_loss"],
        "stop_reason": stop_reason,
        "status": "budget_stopped" if stop_reason else "completed",
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "restoration_peak_allocated_bytes": restored_peak,
        "checkpoint": str(run.path / "checkpoints/current.json"),
        "selected_checkpoint": state["selected"],
        "first_step_included_in_wall_budget": True,
        **diagnostics,
    }
