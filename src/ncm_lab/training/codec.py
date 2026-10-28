"""Bounded streaming codec fit, true held-out diagnostics and immutable selection."""

import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from ncm_lab.models.codecs import MultiscaleCodec
from ncm_lab.models.codecs import pool
from ncm_lab.training.records import anchors, disjoint, admitted_hash
from ncm_lab.training.latent_supervision import teacher_block
from ncm_lab.training.checkpoints import (
    atomic_checkpoint,
    read_checkpoint,
    preflight,
    tensor_bytes,
    selected_payload,
    rng_state,
    restore_rng,
)
from ncm_lab.runtime.io import canonical_hash, write_json, digest, read


def codec_objective(
    codec,
    head,
    states,
    valid,
    labels,
    prediction_weight=0.1,
    temperature=1.0,
    chunk_positions=8,
    normalization=1.0,
):
    if temperature <= 0 or chunk_positions <= 0:
        raise ValueError("invalid head diagnostic budget")
    codes = codec.encode(states, valid)
    metrics, total = {}, states.new_zeros(())
    teacher = states[valid].detach()
    for level, code in codes.items():
        reconstruction = codec.reconstruct(code, level)[valid]
        mse = (reconstruction.float() - teacher.float()).square().mean() / normalization
        kl, agreement, nll = states.new_zeros((), dtype=torch.float32), 0, 0.0
        for start in range(0, teacher.shape[0], chunk_positions):
            stop = start + chunk_positions
            # Full vocabulary, bounded by position chunks; target head remains frozen in autograd.
            teacher_logits = head(teacher[start:stop]).float()
            recovered_logits = head(reconstruction[start:stop]).float()
            target_logp = F.log_softmax(teacher_logits / temperature, -1).detach()
            logp = F.log_softmax(recovered_logits / temperature, -1)
            kl = kl + F.kl_div(logp, target_logp, log_target=True, reduction="sum") * temperature**2
            agreement += int((teacher_logits.argmax(-1) == recovered_logits.argmax(-1)).sum())
            nll += float(
                F.cross_entropy(recovered_logits.detach(), labels[valid][start:stop], reduction="sum")
            )
        kl = kl / teacher.shape[0]
        total = total + mse + prediction_weight * kl
        metrics[str(level)] = {
            "normalized_mse": float(mse.detach()),
            "raw_mse": float(mse.detach()) * normalization,
            "head_kl_teacher_to_reconstruction": float(kl.detach()),
            "top_token_agreement": agreement / teacher.shape[0],
            "reference_nll": nll / teacher.shape[0],
            "valid_positions": teacher.shape[0],
        }
    return total, metrics


@torch.no_grad()
def evaluate_codec(codec, target, records, meter, normalization=1.0, selection=True):
    allowed = {"dev"} if selection else {"dev", "test"}
    if not records or any(r["split"] not in allowed for r in records):
        raise ValueError("codec selection requires dev-only records; test is diagnostic-only")
    codec.eval()
    totals, count = {}, 0
    for row in records:
        available = anchors(row)
        if not available:
            continue
        meter.check(len(row["input_ids"]), 0, step=False)
        meter.source_tokens += len(row["input_ids"])
        states, valid, labels = teacher_block(target, row, available[0])
        _, observations = codec_objective(
            codec, target.get_output_embeddings(), states, valid, labels, normalization=normalization
        )
        size = int(valid.sum())
        count += size
        for level, metrics in observations.items():
            totals.setdefault(level, {k: 0.0 for k in metrics if k != "valid_positions"})
            for key in totals[level]:
                totals[level][key] += metrics[key] * size
    if not count:
        raise ValueError("no usable codec development labels")
    return {
        level: {**{k: v / count for k, v in metrics.items()}, "valid_positions": count}
        for level, metrics in totals.items()
    }


def fit_codec(
    target,
    train_rows,
    dev_rows,
    meter,
    candidates,
    steps=2,
    seed=0,
    width=256,
    kind="nested",
    levels=(4, 8, 15),
    dependencies=None,
    resume=None,
    source_contract=None,
    objective=None,
    data_tier=None,
):
    disjoint(train_rows, dev_rows)
    started = time.perf_counter()
    if not candidates or steps <= 0:
        raise ValueError("finite nonempty codec candidate schedule required")
    import math

    if any(not math.isfinite(w) or w < 0 for w in candidates):
        raise ValueError("codec candidates must be finite nonnegative coefficients")
    inferred = (
        "reconstruction_only"
        if all(w == 0 for w in candidates)
        else "prediction_aware"
        if all(w > 0 for w in candidates)
        else "adaptive_selection"
    )
    objective = objective or inferred
    if objective != inferred:
        raise ValueError("codec objective/search contradicts its coefficient candidates")
    from ncm_lab.training.source_contracts import STATE, validate_row, model_identity

    if source_contract is None:
        if not all(r.get("origin") == "software_fixture" for r in train_rows + dev_rows):
            raise ValueError("codec fitting requires admitted teacher source contract")
        source_contract = {
            "schema_version": 3,
            "fixture": True,
            "state_definition": STATE,
            "hidden_size": target.config.hidden_size,
        }
    for row in train_rows + dev_rows:
        if row.get("origin") != "software_fixture":
            validate_row(row, source_contract, legacy=bool(row.get("legacy_source_migrated")))
    device, dtype = target.get_input_embeddings().weight.device, target.get_input_embeddings().weight.dtype
    head = target.get_output_embeddings().requires_grad_(False)
    usable = [r for r in train_rows if anchors(r)]
    if not usable:
        raise ValueError("no codec training anchors")
    observations, best = [], float("inf")
    optimizer_steps = 0
    from ncm_lab.runtime.provenance import implementation_hash
    from ncm_lab.models.factory import execution_settings

    signature = canonical_hash(
        {
            "source": model_identity(dependencies)
            if dependencies and "target_id" in dependencies
            else source_contract,
            "train": train_rows,
            "dev": dev_rows,
            "implementation_sha256": implementation_hash(),
            "architecture_version": 3 if kind == "independent" else 2,
            "source_contract": source_contract,
            "objective": objective,
            "data_tier": data_tier,
            "steps_per_candidate": steps,
            "execution": execution_settings(),
            "dtype": str(dtype),
            "device": str(device),
            "width": width,
            "kind": kind,
            "levels": levels,
            "candidates": candidates,
            "seed": seed,
            "state": "post_final_norm_prediction_position",
            "criterion": "mean_mse_plus_0.1_head_kl",
        }
    )
    # Training-only scale statistic, bounded one pass; no dev/test statistics are fitted.
    recovered = None
    inherited_wall = 0.0
    inherited_optimizer_steps = 0
    sampling = []
    if resume:
        resume = Path(resume)
        recovered = read_checkpoint(resume, signature)
        if recovered.get("training_state_version") != 2 or recovered.get("rng") is None:
            raise ValueError("codec resume requires a current recovery checkpoint with RNG/sampler state")
        normalization = recovered["normalization"]
        consumption = recovered["consumption"]
        meter.steps = meter.inherited_steps = consumption["steps"]
        meter.source_tokens = meter.inherited_source_tokens = consumption["source_tokens"]
        meter.supervised_positions = meter.inherited_supervised_positions = consumption[
            "supervised_positions"
        ]
        parent_metrics = resume.parent / "metrics.json"
        if parent_metrics.exists():
            charged = read(parent_metrics).get("resources", {})
            meter.steps = max(meter.steps, charged.get("charged_operations", meter.steps))
            meter.source_tokens = max(
                meter.source_tokens, charged.get("charged_source_positions", meter.source_tokens)
            )
            meter.inherited_steps = meter.steps
            meter.inherited_source_tokens = meter.source_tokens
        inherited_wall = consumption["wall_seconds"]
        meter.started -= inherited_wall
        optimizer_steps = recovered["optimizer_steps"]
        inherited_optimizer_steps = optimizer_steps
        sampling = list(recovered["sampling_order"])
        observations = list(recovered["previous_observations"]) + [
            {**recovered["current_observation"], "checkpoint": "sha256:" + digest(resume)}
        ]
        chosen = min(observations, key=lambda r: (r["criterion"], r["candidate"], r["step"]))
        best = chosen["criterion"]
        if (chosen["candidate"], chosen["step"]) == (recovered["candidate"], recovered["step"]):
            best_payload = recovered
        else:
            best_payload = read_checkpoint(resume.parent / "codec-selected.pt", signature)
            if (best_payload["candidate"], best_payload["step"]) != (chosen["candidate"], chosen["step"]):
                raise ValueError("retained codec selection does not match evaluated best candidate")
        atomic_checkpoint(meter.artifact_dir / "codec-selected.pt", selected_payload(best_payload), meter)
    else:
        energy, positions = 0.0, 0
        for row in usable:
            meter.check(len(row["input_ids"]), 0, step=False)
            meter.source_tokens += len(row["input_ids"])
            state, mask, _ = teacher_block(target, row, anchors(row)[0])
            energy += float(state[mask].float().square().sum())
            positions += int(mask.sum()) * state.shape[-1]
        normalization = max(energy / positions, 1e-6)
    for candidate, weight in enumerate(candidates):
        if recovered and candidate < recovered["candidate"]:
            continue
        if weight < 0:
            raise ValueError("negative prediction weight")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            codec = MultiscaleCodec(target.config.hidden_size, width, kind, levels).to(
                device=device, dtype=dtype
            )
        trainable = [p for p in codec.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3) if trainable else None
        sampler = random.Random(seed)
        first_step = 1
        if recovered and candidate == recovered["candidate"]:
            codec.load_state_dict(recovered["model"])
            if optimizer:
                optimizer.load_state_dict(recovered["optimizer"])
            restore_rng(recovered["rng"], sampler)
            first_step = recovered["step"] + 1
        for step in range(first_step, steps + 1):
            row = usable[sampler.randrange(len(usable))]
            anchor = sampler.choice(anchors(row))
            meter.check(len(row["input_ids"]), 15)
            preflight(meter, 4 * tensor_bytes(codec.state_dict()) + 2 * 1024 * 1024)
            meter.source_tokens += len(row["input_ids"])
            states, valid, labels = teacher_block(target, row, anchor)
            codec.train()
            loss, _ = codec_objective(codec, head, states, valid, labels, weight, normalization=normalization)
            if not torch.isfinite(loss):
                raise ValueError("non-finite codec training loss")
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            meter.record(0, int(valid.sum()))
            if optimizer:
                optimizer.step()
                optimizer_steps += 1
            metrics = evaluate_codec(codec, target, dev_rows, meter, normalization)
            criterion = sum(
                v["normalized_mse"] + 0.1 * v["head_kl_teacher_to_reconstruction"] for v in metrics.values()
            ) / len(metrics)
            sampling.append([candidate, row.get("original_id"), anchor])
            observation = {
                "candidate": candidate,
                "prediction_weight": weight,
                "step": step,
                "criterion": criterion,
                "metrics": metrics,
                "split": "dev",
            }
            payload = {
                "schema_version": 2,
                "identity": signature,
                "spec": codec.spec,
                "model": codec.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "seed": seed,
                "candidate": candidate,
                "step": step,
                "normalization": normalization,
                "dependencies": dependencies,
                "source_contract": source_contract,
                "objective": objective,
                "data_tier": data_tier,
                "train_hash": admitted_hash(train_rows),
                "dev_hash": canonical_hash(dev_rows),
                "training_state_version": 2,
                "rng": rng_state(sampler),
                "sampling_order": list(sampling),
                "optimizer_steps": optimizer_steps,
                "previous_observations": list(observations),
                "current_observation": observation,
                "consumption": {
                    "steps": meter.steps,
                    "source_tokens": meter.source_tokens,
                    "supervised_positions": meter.supervised_positions,
                    "wall_seconds": inherited_wall + time.perf_counter() - started,
                },
            }
            path = meter.artifact_dir / "codec-latest.pt"
            checkpoint = atomic_checkpoint(path, payload, meter)
            observation = {**observation, "checkpoint": checkpoint}
            write_json(meter.artifact_dir / f"codec-dev-{candidate}-{step}.json", observation, exclusive=True)
            observations.append(observation)
            if criterion < best:
                best = criterion
                atomic_checkpoint(meter.artifact_dir / "codec-selected.pt", selected_payload(payload), meter)
    path = meter.artifact_dir / "codec-selected.pt"
    payload = read_checkpoint(path)
    codec.load_state_dict(payload["model"])
    codec.normalization = normalization
    scales = fit_latent_scales(codec, target, usable, meter)
    selection = {
        "schema_version": 3,
        "identity": signature,
        "criterion": "mean_mse_plus_0.1_head_kl",
        "source_contract": source_contract,
        "objective": objective,
        "data_tier": data_tier,
        "codec_seed": seed,
        "admitted_eligible_label_positions": sum(
            sum(row["response_mask"][row["prompt_length"] + 1 :]) for row in train_rows
        ),
        "statistics_training_subset_hash": admitted_hash(train_rows),
        "codec_spec": codec.spec,
        "tie_break": "earliest candidate then step",
        "candidates": observations,
        "selected": min(observations, key=lambda r: (r["criterion"], r["candidate"], r["step"])),
        "weights": path.name,
        "weights_sha256": digest(path),
        "dev_hash": canonical_hash(dev_rows),
        "normalization": normalization,
        "rates": codec.rates(),
        "latent_scales": {str(n): scale.tolist() for n, scale in scales.items()},
        "latent_scale_source": "selected codec, train-only first eligible anchor per source",
        "search_cost": {
            "steps": meter.steps,
            "target_processed_positions": meter.source_tokens,
            "optimizer_steps": optimizer_steps,
            "new_invocation_optimizer_steps": optimizer_steps - inherited_optimizer_steps,
            "new_invocation_steps": meter.new_steps,
            "new_invocation_target_positions": meter.new_source_tokens,
            "wall_seconds": inherited_wall + time.perf_counter() - started,
        },
    }
    write_json(meter.artifact_dir / "codec-selection.json", selection, exclusive=True)
    return selection


@torch.no_grad()
def fit_latent_scales(codec, target, rows, meter):
    sums, squares, counts = {}, {}, {}
    for row in rows:
        meter.check(len(row["input_ids"]), 0, step=False)
        meter.source_tokens += len(row["input_ids"])
        state, valid, _ = teacher_block(target, row, anchors(row)[0])
        codes = codec.encode(state, valid)
        for level, code in codes.items():
            mask = pool(state, level, valid)[1]
            values = code[mask].double()
            sums[level] = sums.get(level, 0) + values.sum(0)
            squares[level] = squares.get(level, 0) + values.square().sum(0)
            counts[level] = counts.get(level, 0) + values.shape[0]
    return {
        n: (squares[n] / counts[n] - (sums[n] / counts[n]).square()).clamp_min(1e-6).sqrt().cpu().float()
        for n in counts
    }


def load_selected_codec(
    selection_path,
    device="cpu",
    dtype=torch.float32,
    expected_hash=None,
    expected_model=None,
    diagnostic=False,
):
    selection_path = Path(selection_path)
    selection = read(selection_path)
    path = selection_path.parent / selection["weights"]
    if digest(path) != selection["weights_sha256"] or expected_hash and expected_hash != digest(path):
        raise ValueError("selected codec replaced or incompatible")
    payload = read_checkpoint(path, selection["identity"])
    migration = selection.get("migration")
    if migration:
        from ncm_lab.runtime.provenance import validate_run

        if (
            digest(migration["selection_path"]) != migration["selection_sha256"]
            or digest(migration["manifest_path"]) != migration["manifest_sha256"]
            or digest(migration["teacher_contract_path"]) != migration["teacher_contract_sha256"]
        ):
            raise ValueError("legacy codec migration provenance changed")
        validate_run(Path(migration["manifest_path"]).parent)
        if migration.get("fixture") and not diagnostic:
            raise ValueError("fixture migration is diagnostic-only")
    spec = dict(payload["spec"])
    spec.setdefault("architecture_version", 1)
    codec = MultiscaleCodec(**spec).to(device=device, dtype=dtype)
    codec.load_state_dict(payload["model"])
    codec.normalization = selection["normalization"]
    codec.training_data_hash = payload["train_hash"]
    codec.source_contract = payload.get("source_contract", selection.get("recovered_source_contract"))
    if codec.source_contract is None and not diagnostic:
        raise ValueError(
            "legacy codec lacks source contract; migrate from verified original provenance or use diagnostic-only legacy loading"
        )
    codec.objective = payload.get(
        "objective", selection.get("recovered_objective", "legacy_adaptive_selection")
    )
    chosen = selection["selected"]["prediction_weight"]
    if (
        codec.objective == "prediction_aware"
        and chosen <= 0
        or codec.objective == "reconstruction_only"
        and chosen != 0
    ):
        raise ValueError("selected coefficient contradicts codec objective")
    if expected_model is not None:
        from ncm_lab.training.source_contracts import validate_codec_source

        if codec.source_contract and codec.source_contract.get("fixture") and not diagnostic:
            raise ValueError("fixture codec cannot enter a research treatment")
        validate_codec_source(codec, expected_model)
    codec.dependency_contract = {
        "source": codec.source_contract,
        "spec": codec.spec,
        "train_hash": payload["train_hash"],
        "development_hash": payload["dev_hash"],
        "normalization": selection["normalization"],
        "latent_scales_sha256": canonical_hash(selection.get("latent_scales", {})),
        "selection_identity": selection["identity"],
        "weights_sha256": selection["weights_sha256"],
        "objective": codec.objective,
        "selected_prediction_weight": chosen,
        "data_tier": selection.get("data_tier"),
    }
    codec.latent_scales = {
        int(n): torch.tensor(scale) for n, scale in selection.get("latent_scales", {}).items()
    }
    return codec.eval().requires_grad_(False)


def migrate_codec(selection_path, teacher_contract_path, output, diagnostic_fixture=False):
    """Derive missing dependencies from original hash-bound evidence; never rewrite it."""
    from ncm_lab.runtime.provenance import validate_run
    from ncm_lab.training.source_contracts import validate_contract, validate_migration, validate_row
    from ncm_lab.training.records import load_records, admitted_hash

    selection_path = Path(selection_path)
    manifest = validate_run(selection_path.parent)
    if (
        manifest["run_kind"] != "codec"
        or manifest["status"] != "completed"
        or manifest["result_origin"] != "measured"
        and not diagnostic_fixture
    ):
        raise ValueError(
            "codec migration requires original completed measured provenance or explicit diagnostic fixture mode"
        )
    original = read(selection_path)
    payload = read_checkpoint(selection_path.parent / original["weights"], original["identity"])
    config = read(selection_path.parent / "config.json")
    recovered = read(teacher_contract_path)
    if recovered.get("migration_version") == 3:
        validate_migration(recovered)
    if not recovered.get("complete") or canonical_hash(recovered["contract"]) != recovered["contract_sha256"]:
        raise ValueError("cannot recover codec from incomplete/changed teacher contract")
    if recovered.get("fixture") and not diagnostic_fixture:
        raise ValueError("fixture teacher migration is diagnostic-only")
    contract = recovered["contract"]
    validate_contract(contract, config["model"])
    from ncm_lab.training.source_contracts import model_identity

    if model_identity(payload["dependencies"]) != model_identity(config["model"]):
        raise ValueError("original codec dependencies contradict original configuration")
    rows = load_records(config["training_data"], "train", allow_fixture=diagnostic_fixture)
    for row in rows:
        validate_row(row, contract, legacy=True)
    if config.get("data_tier"):
        from ncm_lab.data.ladder import resolve_ladder

        old = resolve_ladder(
            rows,
            [config["data_tier"]],
            config.get("data_seed", 0),
            eligibility_version=config.get("data_ladder_version", 1),
        )[0]
        rows = [row for row in rows if row["original_id"] in old["original_ids"]]
    if (
        admitted_hash(rows) != payload["train_hash"]
        or digest(config["training_data"]) != recovered["data_sha256"]
    ):
        raise ValueError("cannot recover original codec training-subset provenance")
    contract = {**contract, "hidden_size": payload["spec"]["hidden_size"]}
    coefficients = [candidate["prediction_weight"] for candidate in original["candidates"]]
    objective = (
        "reconstruction_only"
        if all(w == 0 for w in coefficients)
        else "prediction_aware"
        if all(w > 0 for w in coefficients)
        else "adaptive_selection"
    )
    derived = {
        **original,
        "schema_version": 3,
        "weights": str((selection_path.parent / original["weights"]).resolve()),
        "recovered_source_contract": contract,
        "recovered_objective": objective,
        "migration": {
            "version": 3,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": digest(selection_path),
            "manifest_path": str((selection_path.parent / "manifest.json").resolve()),
            "manifest_sha256": digest(selection_path.parent / "manifest.json"),
            "teacher_contract_sha256": digest(teacher_contract_path),
            "teacher_contract_path": str(Path(teacher_contract_path).resolve()),
            "fixture": manifest["result_origin"] == "test_fixture",
            "recovered_fields": [
                "source model/tokenizer/policy",
                "teacher state width",
                "training subset",
                "objective/search identity",
            ],
        },
    }
    write_json(output, derived, exclusive=True)
    return {"path": str(output), "original_bytes_preserved": True, "objective": objective}
