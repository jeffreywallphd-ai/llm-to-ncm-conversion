"""Connected bounded native codec/generator training, held-out selection and recovery."""

from pathlib import Path
import gc
import random
import time
import sys
import torch
from ncm_lab.codec.model import ContinuousCodec
from ncm_lab.config.architecture import Architecture
from ncm_lab.data.artifacts import Dataset, Cursor
from ncm_lab.data.records import collate, ALIGNMENT
from ncm_lab.embedded.geometry import masked_pool
from ncm_lab.runtime.io import (
    canonical_hash,
    write_json,
    digest,
    implementation_hash,
    atomic_json,
    read,
    disk_path,
    plain_path,
)
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
        meter.charge(source=tensor.shape[0], answer=tensor.shape[0])
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
        meter.charge(source=int(valid.sum()), answer=int(valid.sum()), canvas=codec.architecture.canvas)
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


def prune(folder, selected, *, separate_selection=False):
    """Retain two newest recovery versions plus selected; interrupted writes survive."""
    import shutil

    folder = Path(folder).resolve()
    candidates = sorted(
        (plain_path(p) for p in disk_path(folder).glob("version-*")),
        key=lambda p: disk_path(p).stat().st_mtime,
        reverse=True,
    )
    recent = candidates[:2]
    if separate_selection:
        from ncm_lab.training.checkpoints import RECOVERY
        recoveries = [p for p in candidates if read(p / "manifest.json")["schema"] == RECOVERY]
        recent = recoveries[:2]
    protected = {p.resolve() for p in recent}
    if separate_selection:
        for version in recent:
            reader = CheckpointReader(version / "manifest.json")
            training = reader.materialize(reader.field("training"))
            prior = training.get("selected")
            if prior and not training.get("selected_this_step"):
                if digest(prior["manifest"]) != prior["sha256"]:
                    raise ValueError("retained recovery selected weights changed")
                protected.add(Path(prior["manifest"]).parent.resolve())
    if selected:
        protected.add(Path(selected["manifest"]).parent.resolve())
    for path in candidates if separate_selection else candidates[2:]:
        resolved = path.resolve()
        if resolved in protected:
            continue
        if disk_path(path).is_symlink() or resolved.parent != folder:
            raise ValueError("unsafe checkpoint retention path")
        shutil.rmtree(disk_path(resolved))


def fit(run, value, bound, *, resume=None, stop_after=None, recovery_compatibility=None):
    from ncm_lab.training.recovery_compatibility import verify_recovery

    options = value["training"]
    stage1 = value.get("stage1")
    stage3 = value.get("stage3")
    schedule = None
    peaks = [options["learning_rate"], options["new_learning_rate"] if value["stage"] in {"generator", "profile"} else options["learning_rate"]]
    if stage3:
        from ncm_lab.stage3.schedule import Schedule
        schedule = Schedule(**stage3["schedule"])
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
    early_reader = CheckpointReader(resume, check=run.meter.check) if stage1 and resume else None
    early_metadata = early_reader.materialize(early_reader.field("metadata")) if early_reader else None
    normalization = (
        source.get("normalization")
        if generator
        else (
            early_metadata["normalization"]
            if early_metadata
            else state_normalization(dataset, train, run.meter)
        )
    )
    dtype = getattr(torch, value["dtype"])
    device = value["device"]
    restore_started = time.perf_counter()
    restored_checkpoint_ref = None
    reader = early_reader or (CheckpointReader(resume, check=run.meter.check) if resume else None)
    previous_metadata = reader.materialize(reader.field("metadata")) if reader else None
    if reader:
        # Validate implementation/scheduler/identity before allocating model or optimizer.
        verify_recovery(
            reader,
            {**previous_metadata, "implementation_sha256": implementation_hash()},
            recovery_compatibility,
        )
    if previous_metadata and (
        previous_metadata["architecture"] != architecture.to_dict()
        or previous_metadata["dataset_sha256"] != dataset.sha256
        or previous_metadata["source"] != source
        or previous_metadata["phase_id"] != value["budget"]["phase_id"]
        or previous_metadata["dtype"] != value["dtype"]
    ):
        raise ValueError(
            "resume architecture/data/codec/phase mismatch before model allocation; changing canvas requires fresh training"
        )
    if stage3 and previous_metadata and previous_metadata.get("stage3_contract") != stage3:
        raise ValueError("Stage 3 schedule/recipe changed before allocation")
    if stage1 and previous_metadata:
        from ncm_lab.training.validation_plan import stage1_codec_plan

        expected_selection = (
            {"policy": stage1["selection"], "indices": dev[: options["max_validation_examples"]]}
            if generator
            else stage1_codec_plan(
                dataset,
                dev,
                architecture.resolutions,
                options["max_validation_examples"],
                options["validation_seed"],
            )
        )
        expected_training = {
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
        }
        if (
            previous_metadata.get("stage1_contract") != stage1
            or previous_metadata["training_contract"] != expected_training
            or previous_metadata["runtime"]["device"] != device
            or previous_metadata["runtime"]["torch"] != str(torch.__version__)
            or previous_metadata["model"] != model_identity(value["model"])
        ):
            raise ValueError(
                "Stage 1 recovery reduction/selection/training/runtime contract mismatch before allocation"
            )
        if previous_metadata["selection_policy"] != expected_selection:
            raise ValueError("Stage 1 recovery fixed selection subset/plan mismatch before allocation")
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
    from ncm_lab.training.validation_plan import stage1_codec_plan

    validation_plan = (
        None
        if generator
        else (stage1_codec_plan if stage1 else codec_validation_plan)(
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
            "policy": stage1["selection"] if stage1 else "generator_equal_example_loss_v1",
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
    if stage3:
        metadata["stage3_contract"] = stage3
    if stage1:
        metadata["stage1_contract"] = stage1
        metadata["precision_boundary"] = {
            "cached_source_dtype": source["dtype"],
            "parameter_dtype": value["dtype"],
            "target_cast": stage1["generator_target_cast"]
            if generator
            else "cached_states_explicitly_cast_to_fp32_codec",
        }
    if options.get("objective") == "epochs":
        metadata["training_contract"]["objective_epochs"] = options["epochs"]
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
        **({"example_presentations": 0} if stage1 else {}),
    }
    if reader:
        transition = verify_recovery(reader, metadata, recovery_compatibility)
        if transition:
            run.event("policy_recovery_transition", **transition)
        reader.load_model(model)
        if schedule:
            saved_training = reader.materialize(reader.field("training"))
            if saved_training["optimizer_steps"]:
                schedule.apply(optimizer, peaks, saved_training["optimizer_steps"])
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
        early_reader = None
        early_metadata = None
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
    objective_kind = options.get("objective", "steps")
    max_update_tokens = (
        max(
            len(dataset.metadata(i)["example"]["prompt_ids"])
            + len(dataset.metadata(i)["example"]["answer_ids"])
            if generator
            else len(dataset.metadata(i)["example"]["answer_ids"])
            for i in train
        )
        * options["batch_size"]
        * options["accumulation"]
    )

    def objective_reached():
        if objective_kind == "tokens":
            return state["processed_source_tokens"] >= options["max_processed_tokens"] and (
                not generator or state.get("last_executed_phase") == "full"
            )
        if objective_kind == "epochs":
            return cursor.position == len(cursor.order) and cursor.epoch + 1 >= options["epochs"]
        return state["optimizer_steps"] >= options["steps"]

    def unfinished(reason, code):
        checkpoint = run.path / "checkpoints/current.json"
        return {
            "status": "budget_stopped" if code == "resource_interrupted" else "incomplete",
            "completion": {"code": code, "objective": objective_kind, "reason": reason},
            "stop_reason": reason,
            "optimizer_steps": state["optimizer_steps"],
            "processed_source_tokens": state["processed_source_tokens"],
            "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
            "artifact_published": False,
        }

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

        if stage3:
            from ncm_lab.stage3.sampling import fixed_prompts as stage3_prompts
            sample_prompts = stage3_prompts(dataset, dev, recorder.options.sample_prompts, options["validation_seed"])
        else:
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
                report_head_agreement=bool(stage1 and not training),
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

    last_validation = None

    def evaluate(full_dev=False, selection_eligible=True):
        nonlocal last_validation
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
                [{"index": i} for i in (dev if full_dev else dev[: options["max_validation_examples"]])]
                if generator
                else (
                    stage1_codec_plan(
                        dataset, dev, architecture.resolutions, len(dev), options["validation_seed"]
                    )["assignments"]
                    if stage1 and full_dev
                    else validation_plan["assignments"]
                )
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
        score_key = "text_loss" if generator and stage1 else "loss"
        score = sum(row[score_key] for row in rows) / len(rows)
        if stage1:
            from collections import defaultdict

            cutoff_rows = defaultdict(list)
            for row in rows:
                if "cutoff" in row:
                    cutoff_rows[str(row["cutoff"])].append(row)
            keys = (
                ("loss", "text_loss", "latent_loss")
                if generator
                else ("loss", "mse_full", "mse_cutoff", "head_kl", "head_agreement", "head_positions")
            )
            last_validation = {key: sum(r[key] for r in rows) / len(rows) for key in keys if key in rows[0]}
            last_validation.update(
                selection_score=score,
                plan_sha256=canonical_hash(metadata["selection_policy"]),
                assignments=len(rows),
                distinct_examples=len({a["index"] for a in assignments}),
                per_cutoff={
                    k: {
                        metric: sum(r[metric] for r in group) / len(group)
                        for metric in ("mse_full", "mse_cutoff", "head_kl", "loss")
                    }
                    for k, group in cutoff_rows.items()
                },
                selection_eligible=selection_eligible,
                full_dev=full_dev,
            )
            if not all(__import__("math").isfinite(last_validation[k]) for k in keys if k in last_validation):
                raise ValueError("nonfinite Stage 1 held-out validation")
            recorder.event("stage1_validation", components=last_validation)
        recorder.event(
            "selection_evaluated",
            score=score,
            plan_sha256=canonical_hash(validation_plan),
            policy=metadata["selection_policy"]["policy"],
            per_example_components=rows,
        )
        return score

    try:
        if stage1 and "initial_validation" not in state:
            if state["optimizer_steps"]:
                raise ValueError("Stage 1 recovery lacks its immutable initialization measurement")
            from ncm_lab.monitoring.samples import preserve_training_state, observe_samples

            sampler_state = sampler.getstate()
            with preserve_training_state(model):
                evaluate(selection_eligible=False)
                state["initial_validation"] = last_validation
                if generator and sample_prompts:
                    observe_samples(
                        model,
                        tokenizer,
                        sample_prompts,
                        recorder,
                        run.meter,
                        model_identity(value["model"])["eos_ids"],
                    )
            sampler.setstate(sampler_state)
            write_json(run.path / "initialization.json", state["initial_validation"], exclusive=True)
            recorder.event("initialization_measured", eligible_for_selection=False)
        if resume and objective_reached():
            # No update is repeated. A target-boundary checkpoint still needs a
            # fresh attempt's validation, durable recovery and export/handoff.
            selection = evaluate()
            chosen = state["best_dev_loss"] is None or selection < state["best_dev_loss"]
            state["selected_this_step"] = chosen
            if chosen:
                state["best_dev_loss"] = selection
                if stage1:
                    state["selected_validation"] = last_validation
                    state["selected_optimizer_step"] = state["optimizer_steps"]
            state["cursor"] = cursor.state()
            saved = save_checkpoint(
                run.path / "checkpoints",
                recovery_payload(model, optimizer, identity, metadata, state, sampler),
                meter=run.meter,
                shard_bytes=options["shard_bytes"],
            )
            if chosen:
                state["selected"] = saved
                atomic_json(run.path / "checkpoints/selected.json", saved)
            recorder.event(
                "checkpoint_completed",
                manifest=saved["manifest"],
                sha256=saved["sha256"],
                finalization_only=True,
            )
            recorder.metric(
                {
                    "resources/budget_wall_seconds_used": Measure(
                        run.meter.elapsed_seconds(), unit="seconds"
                    ),
                    "resources/budget_wall_seconds_remaining": Measure(
                        max(0, run.meter.budget.max_wall_seconds - run.meter.elapsed_seconds())
                        if run.meter.budget.enforce_wall_time_limits
                        else None,
                        unit="seconds",
                        status="valid" if run.meter.budget.enforce_wall_time_limits else "disabled_by_policy",
                    ),
                    "resources/enforce_wall_time_limits": Measure(
                        int(run.meter.budget.enforce_wall_time_limits)
                    ),
                },
                terminal=True,
            )
        while (
            state["optimizer_steps"] < options["steps"]
            and cursor.epoch < options["epochs"]
            and not (cursor.position == len(cursor.order) and cursor.epoch + 1 >= options["epochs"])
        ):
            if objective_reached():
                break
            if state["processed_source_tokens"] >= options["max_processed_tokens"]:
                if objective_kind != "tokens":
                    raise BudgetStop("processed-token safety limit reached before objective completion")
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
                if stage1:
                    state["example_presentations"] += len(indices)
                state["backbone_answer_positions"] += metrics.get("backbone_answer_positions", 0)
                aggregate.append(metrics)
            report = optimizer.check_gradients()
            norm = report.norm
            diagnostic_gradients = gradients(model) if recorder.detail_due(recorder.step) else {}
            limit = options.get("gradient_clip", 1.0)
            report.clip_(limit)
            if schedule:
                schedule.apply(optimizer, peaks, state["optimizer_steps"] + 1)
            optimizer.step(report)
            del report  # Its parameter/gradient ownership must end before recovery allocation.
            state["optimizer_steps"] += 1
            if schedule:
                state["scheduler"] = schedule.state(peaks, state["optimizer_steps"])
                run.event("learning_rates", **state["scheduler"])
            state["last_executed_phase"] = executed_phase
            terminal_update = (
                objective_reached()
                or state["optimizer_steps"] == options["steps"]
                or cursor.position == len(cursor.order)
                and cursor.epoch + 1 >= options["epochs"]
                or stop_after is not None
                and state["optimizer_steps"] >= stop_after
            )
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
            if stage1:
                for component in ("loss", "text_loss", "latent_loss", "mse_full", "mse_cutoff", "head_kl"):
                    if component in aggregate[0]:
                        metrics[component] = sum(row[component] for row in aggregate) / len(aggregate)
                metrics.update(
                    example_presentations=state["example_presentations"],
                    actual_accumulation=len(aggregate),
                    microbatch_head_positions=[row.get("head_positions") for row in aggregate],
                )
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
                or terminal_update
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
                            run.meter.elapsed_seconds(), unit="seconds"
                        ),
                        "resources/budget_wall_seconds_remaining": Measure(
                            max(0, run.meter.budget.max_wall_seconds - run.meter.elapsed_seconds())
                            if run.meter.budget.enforce_wall_time_limits
                            else None,
                            unit="seconds",
                            status="valid"
                            if run.meter.budget.enforce_wall_time_limits
                            else "disabled_by_policy",
                        ),
                        "resources/enforce_wall_time_limits": Measure(
                            int(run.meter.budget.enforce_wall_time_limits)
                        ),
                        "resources/trainable_parameters": Measure(executed_trainable),
                        "resources/frozen_parameters": Measure(executed_frozen),
                    }
                )
                for name, number in observe().items():
                    if name.endswith("_bytes") and (number is None or type(number) is int):
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
                from ncm_lab.runtime.budget import phase_wall_report

                wall = phase_wall_report(phase_ledger)
                values["budgets/phase/wall_seconds_charged_and_reserved"] = Measure(
                    phase_ledger["wall"],
                    unit="seconds",
                    reduction="known legacy balance plus active finite reservations and closed elapsed charges; open disabled-clock durations unknown",
                )
                values["budgets/phase/wall_seconds_reserved"] = Measure(
                    wall["new_attempts_active_reserved_seconds"]
                    if not wall["legacy_charges_and_reservations_seconds"]
                    else None,
                    unit="seconds",
                    status="valid"
                    if not wall["legacy_charges_and_reservations_seconds"]
                    else "legacy_split_unknown",
                    reduction="active finite reservations only; disabled-clock attempts reserve zero",
                )
                values["resources/run_storage_bytes"] = Measure(
                    sum(p.stat().st_size for p in disk_path(run.path).rglob("*") if p.is_file()),
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
            if terminal_update or state["optimizer_steps"] % options["validation_interval"] == 0:
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
            if stage3 and generator and state["optimizer_steps"] in {1000, 5000, 10000, 20000}:
                write_json(run.path / f"milestones/update-{state['optimizer_steps']}.json",
                           {"training": metrics, "development": last_validation,
                            "scheduler": state["scheduler"], "locked_test_seen": False}, exclusive=True)
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
            improved = selection is not None and (
                state["best_dev_loss"] is None or selection < state["best_dev_loss"]
            )
            recovery_due = (state["optimizer_steps"] % options["checkpoint_interval"] == 0
                            or terminal_update or state["optimizer_steps"] == options["steps"])
            if (
                state["optimizer_steps"] % options["checkpoint_interval"] == 0
                or (improved if stage1 else selection is not None)
                or state["optimizer_steps"] == options["steps"]
                or terminal_update
            ):
                state["cursor"] = cursor.state()
                chosen = selection is not None and (
                    state["best_dev_loss"] is None or selection < state["best_dev_loss"]
                )
                state["selected_this_step"] = chosen
                if chosen:
                    state["best_dev_loss"] = selection
                    if stage1:
                        state["selected_validation"] = last_validation
                        state["selected_optimizer_step"] = state["optimizer_steps"]
                save_started = time.perf_counter()
                recorder.event("checkpointing_started")
                if stage3 and not recovery_due:
                    saved = save_checkpoint(
                        run.path / "checkpoints",
                        {"identity": identity, "metadata": metadata, "model": model.state_dict(keep_vars=True),
                         "training": {"optimizer_steps": state["optimizer_steps"]}},
                        kind=INFERENCE, pointer="selected.json", meter=run.meter,
                        shard_bytes=options["shard_bytes"],
                    )
                else:
                    saved = save_checkpoint(
                        run.path / "checkpoints",
                        recovery_payload(model, optimizer, identity, metadata, state, sampler),
                        meter=run.meter, shard_bytes=options["shard_bytes"],
                    )
                if chosen:
                    state["selected"] = saved
                    atomic_json(run.path / "checkpoints/selected.json", saved)
                prune(run.path / "checkpoints", state["selected"], separate_selection=bool(stage3))
                recorder.event(
                    "checkpoint_completed",
                    manifest=saved["manifest"],
                    sha256=saved["sha256"],
                    duration_seconds=time.perf_counter() - save_started,
                )
                recorder.durable()
            if stage1 and (terminal_update or recorder.step % stage1["progress_interval"] == 0):
                from ncm_lab.runtime.memory import observe

                memory = observe()
                components = {
                    k: metrics[k]
                    for k in ("text_loss", "latent_loss", "mse_full", "mse_cutoff", "head_kl")
                    if k in metrics
                }
                print(
                    f"{value['stage']} step {recorder.step}/{options['steps']} phase={executed_phase} loss={metrics['loss']:.6g} components={components} dev={metrics.get('dev_loss', state['best_dev_loss'])} grad={norm:.5g} presentations={state['example_presentations']} rates={[g['lr'] for g in optimizer.groups]} selected={state.get('selected_optimizer_step')} host_available_bytes={memory['host_available_bytes']} clocks=disabled",
                    file=__import__("sys").stderr,
                    flush=True,
                )
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
        return unfinished(stop_reason, "resource_interrupted")
    diagnostic_stop = (
        not stage1
        and stop_after is not None
        and value["model"].get("fixture")
        and state["optimizer_steps"] >= stop_after
    )
    if not objective_reached() and not diagnostic_stop:
        return unfinished("step/epoch safety cap reached before declared objective", "objective_incomplete")
    if not state["selected"]:
        raise ValueError("no completed held-out selection; recovery remains diagnostic only")
    # Export the held-out selected state without holding a second model/optimizer copy.
    if stage1 and digest(state["selected"]["manifest"]) != state["selected"]["sha256"]:
        raise ValueError("Stage 1 selected checkpoint manifest changed")
    optimizer_summary = {
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
    }
    if stage3:
        optimizer.zero_grad()
        optimizer = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    selected_kind = read(state["selected"]["manifest"])["schema"]
    selected_reader = CheckpointReader(state["selected"]["manifest"], expected_kind=selected_kind, check=run.meter.check)
    transition = None
    if selected_reader.manifest["schema"] == INFERENCE and stage3:
        if selected_reader.materialize(selected_reader.field("metadata")) != metadata:
            raise ValueError("selected weights scientific contract changed")
    else:
        transition = verify_recovery(selected_reader, metadata, recovery_compatibility)
    if transition:
        run.event("selected_policy_recovery_transition", **transition)
    selected_reader.load_model(model)
    selected_step = selected_reader.materialize(selected_reader.field("training"))["optimizer_steps"]
    selected_reader = None
    stage1_report = None
    if stage1:
        evaluate(full_dev=True, selection_eligible=False)
        stage1_report = {
            "schema": "stage1_training_diagnostics_v1",
            "component": metadata["component"],
            "initial": state["initial_validation"],
            "selected": state["selected_validation"],
            "full_development": last_validation,
            "selected_optimizer_step": selected_step,
            "completed_optimizer_steps": state["optimizer_steps"],
            "improved_on_initialization": state["selected_validation"]["selection_score"]
            < state["initial_validation"]["selection_score"],
            "selection_policy": metadata["selection_policy"],
            "precision_boundary": metadata["precision_boundary"],
            "reduction": stage1["reduction"],
            "loss_weighting": "valid-element normalized within each physical microbatch; equal actual microbatch objectives per update; generator physical batch one yields equal-example normalized losses",
            "unique_training_examples": len(train),
            "unique_training_answer_positions": unique_answers,
            "example_presentations": state["example_presentations"],
            "answer_position_exposures": state["answer_position_exposures"],
            "source_token_exposures": state["processed_source_tokens"],
            "initialization_is_not_a_trained_selected_checkpoint": True,
            **({"scheduler": state["scheduler"], "quality_policy": stage3["quality_policy"]} if stage3 else {}),
        }
    if generator:
        from ncm_lab.inference.export import export_generator

        artifact = export_generator(
            model,
            metadata,
            run.path / "artifacts/generator",
            run.meter,
            options["shard_bytes"],
            value["model"],
            selected_step=selected_step,
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
        if stage1:
            from ncm_lab.stage1.policy import codec_gate
            import math

            values = [v for row in scales.values() for v in row]
            valid_scales = all(math.isfinite(v) and v >= 1e-3 for v in values)
            stage1_report["scale_statistics"] = {
                **scale_statistics,
                "scale_floor_fraction": sum(v <= 1.000001e-3 for v in values) / len(values),
            }
            stage1_report["codec_qualification"] = codec_gate(
                state["initial_validation"],
                state["selected_validation"],
                stage1,
                artifacts_valid=valid_scales
                and math.isfinite(normalization["variance"])
                and normalization["variance"] > 0,
            )
            if stage3:
                gate = stage1_report["codec_qualification"]
                gate.update(schema="stage3_codec_qualification_v1", quality_policy=stage3["quality_policy"],
                            quality_warning=not gate["codec_qualified_for_generator"],
                            codec_qualified_for_generator=gate["artifacts_valid"],
                            status="qualified_with_warning" if not gate["codec_qualified_for_generator"] else "qualified")
            if stage3 and stage1_report["codec_qualification"].get("quality_warning"):
                print("WARNING: Stage 3 codec has finite quality non-improvement/regression; continuing under validity_required_quality_advisory_v1. Inspect the selected/initial component report.", file=sys.stderr, flush=True)
            if not stage1_report["codec_qualification"]["artifacts_valid"]:
                raise ValueError("invalid Stage 1 codec normalization/scales; cannot override")
        folder = run.path / "artifacts/codec"
        disk_path(folder).mkdir(parents=True, exist_ok=True)
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
                **({"stage1_diagnostics": stage1_report} if stage1 else {}),
            },
            exclusive=True,
        )
        diagnostics = {
            "latent_statistics": scale_statistics,
            "normalization": normalization,
            "head_frozen": not any(p.requires_grad for p in head.parameters()),
        }
    run.publish(artifact)
    if stage1:
        write_json(run.path / "stage1_diagnostics.json", stage1_report, exclusive=True)
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
        **optimizer_summary,
        "last_executed_phase": state.get("last_executed_phase"),
        "selection_policy": validation_plan,
        "processed_source_tokens": state["processed_source_tokens"],
        "answer_position_exposures": state["answer_position_exposures"],
        "allocated_canvas_positions": state["allocated_canvas_positions"],
        "backbone_answer_positions": state["backbone_answer_positions"],
        "unique_available_answer_positions": unique_answers,
        "best_dev_loss": state["best_dev_loss"],
        "stop_reason": stop_reason,
        "completion": {
            "code": "diagnostic_boundary" if diagnostic_stop else "objective_attained",
            "objective": objective_kind,
            "max_whole_update_tokens": max_update_tokens,
            "token_overshoot": max(0, state["processed_source_tokens"] - options["max_processed_tokens"])
            if objective_kind == "tokens"
            else None,
            "maximum_token_overshoot": max_update_tokens
            * (
                2
                if generator
                and 0
                < options["max_processed_tokens"] - options["warmup_processed_tokens"]
                < max_update_tokens
                else 1
            )
            - 1
            if objective_kind == "tokens"
            else None,
        },
        "status": "budget_stopped" if stop_reason else "completed",
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "restoration_peak_allocated_bytes": restored_peak,
        "checkpoint": str(run.path / "checkpoints/current.json"),
        "selected_checkpoint": state["selected"],
        "selected_optimizer_step": selected_step,
        **({"stage1_diagnostics": stage1_report} if stage1 else {}),
        "first_step_included_in_wall_budget": True,
        **diagnostics,
    }
