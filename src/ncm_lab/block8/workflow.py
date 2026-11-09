"""Source, teacher, codec, targets and hierarchical generator in explicit bounded stages."""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import uuid
from ncm_lab.block8.data import BlockDataset, import_tulu, prepare
from ncm_lab.block8.runtime import Attempt, admission, checked, plan, seal_settings, verify_settings
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.canvas_pipeline import exclusive_writer
from ncm_lab.runtime.io import atomic_json, read, reference, root, write_json


def generate_one(run, settings, selected, *, prompt=None, messages=None):
    import torch
    from ncm_lab.block8.model import generate
    from ncm_lab.block8.training import load_export
    from ncm_lab.data.tokenization import prompt_tokens
    from ncm_lab.teacher.model import model_identity

    manifest_path = checked(selected)
    exported = read(manifest_path)
    metadata = exported.get("metadata", {})
    if (
        exported.get("schema") != "hierarchical_block8_generator_export_v1"
        or model_identity(metadata.get("model_spec", {})) != model_identity(settings["model"])
        or metadata.get("architecture") != settings["architecture"]
        or metadata.get("dtype") != settings["dtype"]
    ):
        raise ValueError("selected block8 export and execution configuration differ")
    admission(run, settings, training=False)
    model, tokenizer, manifest = load_export(manifest_path, settings["device"], check=run.meter.check)
    ids = prompt_tokens(tokenizer, messages if messages is not None else prompt)
    if len(ids) > settings["data"]["max_prompt_tokens"]:
        raise ValueError("block8 prompt exceeds declared prompt allowance")
    if settings["device"] == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = generate(
        model,
        ids,
        settings["decoding"],
        settings["model"]["eos_ids"],
        max_context_tokens=settings["data"]["max_context_tokens"],
        suppressed_ids=[*tokenizer.all_special_ids, *range(len(tokenizer), model.backbone.config.vocab_size)],
        charge=run.meter.charge,
    )
    if settings["device"] == "cuda":
        torch.cuda.synchronize()
    result.update(
        text=tokenizer.decode(result["token_ids"], skip_special_tokens=True),
        generation_seconds=time.perf_counter() - started,
        selected_export=selected,
        selected_step=manifest["selected_step"],
        timing_scope="prefill, native hierarchy traversal, causal text commit and token transfers; model/tokenizer load excluded",
    )
    write_json(run.path / "generation.json", result, exclusive=True)
    return dict(output=reference(run.path / "generation.json"), **result)


def evaluate(run, settings, dataset, selected):
    import torch
    from ncm_lab.block8.model import generate
    from ncm_lab.block8.training import load_export

    with closing(BlockDataset(checked(dataset), settings, run.meter.check)) as data:
        admission(run, settings, training=False)
        model, tokenizer, manifest = load_export(checked(selected), settings["device"], check=run.meter.check)
        rows = []
        for key in data.ids["test"][: settings["decoding"]["max_evaluation_examples"]]:
            parent = data.parent(key)
            if settings["device"] == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = generate(
                model,
                parent.prompt_ids,
                settings["decoding"],
                settings["model"]["eos_ids"],
                max_context_tokens=settings["data"]["max_context_tokens"],
                suppressed_ids=[
                    *tokenizer.all_special_ids,
                    *range(len(tokenizer), model.backbone.config.vocab_size),
                ],
                charge=run.meter.charge,
            )
            if settings["device"] == "cuda":
                torch.cuda.synchronize()
            rows.append(
                dict(
                    original_id=parent.original_id,
                    group_id=parent.group_id,
                    reference_ids=parent.answer_ids,
                    reference_provenance=parent.provenance,
                    prediction=tokenizer.decode(result["token_ids"], skip_special_tokens=True),
                    exact_token_match=result["token_ids"] == parent.answer_ids,
                    generation_seconds=time.perf_counter() - started,
                    **result,
                )
            )
            run.meter.check_disk()
        result = dict(
            schema="hierarchical_block8_locked_rollout_evaluation_v1",
            dataset=dataset,
            selected=selected,
            selected_step=manifest["selected_step"],
            rows=rows,
            exact_match_rate=sum(r["exact_token_match"] for r in rows) / len(rows),
            eos_rate=sum(r["eos_genuine"] for r in rows) / len(rows),
            selection_used_test=False,
            correctness_scope="token exact match only; free-form semantic correctness unmeasured",
            speedup_claim=None,
            teacher_or_verifier_used=False,
        )
        write_json(run.path / "evaluation.json", result, exclusive=True)
        return dict(
            report=reference(run.path / "evaluation.json"),
            evaluated=len(rows),
            exact_match_rate=result["exact_match_rate"],
            eos_rate=result["eos_rate"],
        )


def run_workflow(
    settings=None,
    *,
    resume=None,
    experiment_dir=None,
    execute=False,
    prepare_only=False,
    data_manifest=None,
    evaluation_manifest=None,
    checkpoint=None,
    prompt=None,
    messages=None,
    stop_after=None,
    stop_after_codec=None,
):
    if not execute:
        if resume:
            state = read(resume)
            settings = verify_settings(state)
        return plan(settings)
    if resume and any(
        x is not None for x in (settings, experiment_dir, data_manifest, evaluation_manifest, checkpoint)
    ):
        raise ValueError("resume uses sealed settings/dependencies; scientific overrides are forbidden")
    if resume:
        path = Path(resume).resolve()
        folder = path.parent
    else:
        if settings is None:
            raise ValueError("new block8 experiment needs resolved settings")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        folder = Path(experiment_dir or root() / "runs/hierarchical_block8" / stamp).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "manifest.json"
        state = seal_settings(folder, settings)
        state["supplied_data"] = reference(data_manifest) if data_manifest else None
        state["supplied_evaluation"] = reference(evaluation_manifest) if evaluation_manifest else None
        state["generation_export"] = reference(checkpoint) if checkpoint else None
        write_json(path, state, exclusive=True)
    with exclusive_writer(folder / "writer.lock"):
        state = read(path)
        settings = verify_settings(state)

        def save():
            atomic_json(path, state)

        def attempt(action, operation):
            if (
                action in state["stages"]
                and state["stages"][action]["status"] == "completed"
                and action != "generate"
            ):
                receipt = read(checked(state["stages"][action]["receipt"]))
                for value in receipt["result"].values():
                    if isinstance(value, dict) and value.get("schema") == "native_reference_v1":
                        checked(value)
                return receipt["result"]
            count = state["attempts"].get(action, 0)
            if count >= settings["resources"]["max_attempts_per_stage"]:
                raise ValueError("block8 stage retry allowance exhausted; retain this experiment and ledgers")
            state["attempts"][action] = count + 1
            state["status"] = "running"
            state["stages"][action] = dict(status="running")
            save()
            run = None
            try:
                run = Attempt(folder, action, settings)
                run.meter.check_disk()
                print(
                    json.dumps(dict(stage="block8_" + action, attempt=str(run.path)), allow_nan=False),
                    flush=True,
                )
                result = operation(run)
                run.meter.check_disk()
                status = result.get("status", "completed")
                receipt = run.close(status, result)
            except Exception as exc:
                status = "budget_stopped" if isinstance(exc, BudgetStop) else "failed"
                result = dict(error=type(exc).__name__ + ": " + str(exc))
                receipt = run.close(status, result) if run else None
                state["failure"] = dict(stage=action, **result, receipt=receipt)
            state["stages"][action] = dict(status=status, receipt=receipt)
            state["status"] = status
            # No stage-local model/optimizer may remain resident into the next stage.
            import gc

            gc.collect()
            import torch

            if settings["device"] == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status == "completed" and state.get("failure", {}).get("stage") == action:
                state.pop("failure")
            save()
            if status != "completed":
                return None
            return result

        if state.get("generation_export"):
            if (prompt is None) == (messages is None):
                raise ValueError("generation requires exactly one prompt or messages input")
            result = attempt(
                "generate",
                lambda run: generate_one(
                    run, settings, state["generation_export"], prompt=prompt, messages=messages
                ),
            )
            return dict(
                status=state["status"], experiment=str(path), result=result, failure=state.get("failure")
            )
        if state.get("supplied_data"):
            inputs = [checked(state["supplied_data"])]
            if state.get("supplied_evaluation"):
                inputs.append(checked(state["supplied_evaluation"]))
        else:
            imported = attempt("source", lambda run: import_tulu(run, settings))
            if imported is None:
                return dict(status=state["status"], experiment=str(path), failure=state.get("failure"))
            inputs = [checked(imported["data"]), checked(imported["evaluation"])]
        prepared = attempt("prepare", lambda run: dict(dataset=prepare(run, settings, inputs)))
        if prepared is None or prepare_only:
            return dict(
                status=state["status"],
                experiment=str(path),
                dataset=(prepared or {}).get("dataset"),
                failure=state.get("failure"),
            )

        def progress_for(action):
            def progress(recovery, completed):
                state["recoveries"][action] = recovery
                state.setdefault("completed_steps", {})[action] = completed
                save()

            return progress

        from ncm_lab.block8.targets import teacher_features, targets
        from ncm_lab.block8.training import fit_codec, train

        features = attempt("teacher", lambda run: teacher_features(run, settings, prepared["dataset"]))
        if features is None:
            return dict(status=state["status"], experiment=str(path), failure=state.get("failure"))
        codec = attempt(
            "codec",
            lambda run: fit_codec(
                run,
                settings,
                prepared["dataset"],
                features["features"],
                state["recoveries"]["codec"],
                progress_for("codec"),
                stop_after=stop_after_codec,
            ),
        )
        if codec is None:
            return dict(status=state["status"], experiment=str(path), failure=state.get("failure"))
        codes = attempt(
            "targets",
            lambda run: targets(run, settings, prepared["dataset"], features["features"], codec["codec"]),
        )
        if codes is None:
            return dict(status=state["status"], experiment=str(path), failure=state.get("failure"))
        trained = attempt(
            "generator",
            lambda run: train(
                run,
                settings,
                prepared["dataset"],
                codes["targets"],
                codec["codec"],
                state["recoveries"]["generator"],
                progress_for("generator"),
                stop_after=stop_after,
            ),
        )
        if trained is None:
            return dict(status=state["status"], experiment=str(path), failure=state.get("failure"))
        evaluated = attempt(
            "evaluate", lambda run: evaluate(run, settings, prepared["dataset"], trained["selected"])
        )
        if evaluated is not None:
            state["status"] = "completed"
            state.pop("failure", None)
            state["selected"] = trained["selected"]
            save()
        return dict(
            status=state["status"],
            experiment=str(path),
            selected=trained["selected"],
            completed_step=trained["completed_step"],
            codec_step=codec["completed_step"],
            codec=codec["codec"],
            evaluation=evaluated,
            failure=state.get("failure"),
        )
