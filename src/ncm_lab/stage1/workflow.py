"""Foreground, immutable-input Stage 1 state machine. Heavy work stays in children."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from ncm_lab.runtime.io import (
    root,
    read,
    write_json,
    atomic_json,
    reference,
    digest,
    canonical_hash,
    implementation_hash,
    disk_path,
)
from ncm_lab.runtime.canvas_pipeline import exclusive_writer
from ncm_lab.runtime.budget import Budget, Meter, BudgetStop
from ncm_lab.stage1.config import STAGES, ACQUIRE, ACQUIRE_TULU, STAGE2_SCHEMA, report_stem

SCHEMA = "stage1_experiment_v1"
STAGE2_MANIFEST = "stage2_experiment_v1"
DEPENDENCIES = {
    "prepare": (),
    "teacher": ("prepare",),
    "codec": ("teacher",),
    "targets": ("prepare", "teacher", "codec"),
    "generator": ("targets",),
    "evaluate": ("prepare", "generator"),
}


def identifier():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def checked(ref):
    path = Path(ref["manifest"])
    if digest(path) != ref["sha256"]:
        raise ValueError("Stage 1 immutable artifact/receipt changed: " + str(path))
    return path


def manifest_schema(settings):
    return STAGE2_MANIFEST if settings["schema"] == STAGE2_SCHEMA else SCHEMA


def plan(settings, experiment_dir=None):
    data = settings["stages"]["prepare"]["data"]
    if data.get("format") == "tulu_parquet":
        source_info = {
            "format": "tulu_parquet",
            "asset_ids": data["source_asset_ids"],
            "scan_cap_rows": data["max_source_rows"],
            "content_verified": False,
        }
    else:
        source = root() / data["source_path"]
        source_info = {
            "exists": source.is_file(),
            "declared_sha256": data.get("source_sha256"),
            "size_bytes": source.stat().st_size if source.is_file() else None,
            "content_verified": False,
        }
    proposal = {
        "schema": "stage2_plan_v1" if settings["schema"] == STAGE2_SCHEMA else "stage1_plan_v1",
        "canvas": settings["canvas"],
        "seed": settings["seed"],
        "contract_sha256": settings["contract_sha256"],
        "scientific_settings": settings,
        "qualification": "unknown until actual source scan/tokenization; not executable qualified evidence",
        "source": source_info,
        "work_disk_memory": "derived after qualification before teacher allocation; no full-model hashing or allocation in planning",
        "missing_asset_instruction": ACQUIRE_TULU if data.get("format") == "tulu_parquet" else ACQUIRE,
        "training_clock_limits": "disabled",
        "research_work_started": False,
        "stage_order": list(STAGES),
        "dtype_by_stage": {n: v["dtype"] for n, v in settings["stages"].items()},
    }
    if experiment_dir is None:
        return {"status": "planned", "plan": proposal}
    folder = Path(experiment_dir).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("new experiment directory must be empty; select its manifest to resume")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "configs/proposal.json", settings, exclusive=True)
    write_json(folder / "plan.json", proposal, exclusive=True)
    state = {
        "schema": manifest_schema(settings),
        "experiment_id": folder.name,
        "repository_root": str(root()),
        "contract": reference(folder / "configs/proposal.json"),
        "plan": reference(folder / "plan.json"),
        "status": "planned",
        "stages": {s: {"status": "pending", "attempts": []} for s in STAGES},
        "targets": {s: settings["stages"][s]["training"]["steps"] for s in ("codec", "generator")},
        "history": [],
        "resources": None,
        "codec_review_override": None,
        "locked_test_observed": False,
    }
    write_json(folder / "manifest.json", state, exclusive=True)
    return {"status": "planned", "experiment": str(folder / "manifest.json"), "plan": proposal}


def load(experiment):
    path = Path(experiment).resolve()
    state = read(path)
    if state.get("schema") not in {SCHEMA, STAGE2_MANIFEST} or set(state.get("stages", {})) != set(STAGES):
        raise ValueError("an explicit single-pass experiment manifest is required")
    settings = read(checked(state["contract"]))
    if manifest_schema(settings) != state["schema"]:
        raise ValueError("experiment manifest and recipe schema differ")
    checked(state["plan"])
    if settings["implementation_sha256"] != implementation_hash():
        raise ValueError(
            "Stage 1 implementation identity changed; no pilot compatibility bypass is authorized"
        )
    if (
        canonical_hash({k: v for k, v in settings.items() if k not in {"config_path", "contract_sha256"}})
        != settings["contract_sha256"]
    ):
        raise ValueError("Stage 1 proposed scientific contract changed")
    if state["schema"] == STAGE2_MANIFEST and (
        state.get("targets")
        != {n: settings["stages"][n]["training"]["steps"] for n in ("codec", "generator")}
        or state.get("codec_review_override") is not None
    ):
        raise ValueError("Stage 2 requires its codec gate and the original fixed training targets")
    return path, state, settings


def configuration(name, path, state, settings):
    folder = path.parent
    if name == "prepare":
        value = deepcopy(settings["stages"][name])
        rows = value["data"]["max_source_rows"]
        tulu = value["data"].get("format") == "tulu_parquet"
        if tulu:
            rows += value["data"]["overlap"].get("max_source_rows", 0)
        from dataclasses import asdict

        cap = int(
            (settings["qualification"]["max_candidate_bytes"] + rows * 2048)
            * settings["resources"]["work_margin"]
        )
        value["budget"] = asdict(
            Budget(
                "s1-" + settings["contract_sha256"][:40] + "-prepare",
                enforce_wall_time_limits=False,
                max_steps=rows + 1,
                max_phase_steps=(rows + 1) * settings["resources"]["retry_allowance"],
                max_source_tokens=cap,
                max_phase_source_tokens=cap * settings["resources"]["retry_allowance"],
                **(
                    {
                        "max_answer_positions": cap,
                        "max_phase_answer_positions": cap * settings["resources"]["retry_allowance"],
                    }
                    if tulu
                    else {}
                ),
                max_storage_bytes=settings["resources"]["max_retained_bytes"],
                max_phase_storage_bytes=settings["resources"]["max_phase_retained_bytes"],
            )
        )
    else:
        if not state.get("resources"):
            raise ValueError("qualified finite resource derivation required before " + name)
        resource_plan = read(checked(state["resources"]))
        value = deepcopy(resource_plan["stages"][name]["config"])
    receipts = {n: state["stages"][n].get("receipt") for n in STAGES}
    results = {n: read(checked(ref))["result"] for n, ref in receipts.items() if ref}
    inputs = {
        "prepare": {},
        "teacher": {"data": results.get("prepare", {}).get("artifact")},
        "codec": {"features": results.get("teacher", {}).get("artifact")},
        "targets": {
            "data": results.get("prepare", {}).get("artifact"),
            "features": results.get("teacher", {}).get("artifact"),
            "codec": results.get("codec", {}).get("artifact"),
        },
        "generator": {"targets": results.get("targets", {}).get("artifact")},
        "evaluate": {
            "generator": results.get("generator", {}).get("artifact"),
            "data": results.get("prepare", {}).get("artifact"),
            "evaluation": results.get("prepare", {}).get("evaluation_data", {}).get("manifest"),
        },
    }[name]
    if any(v is None for v in inputs.values()):
        raise ValueError("missing manifest-bound prerequisite for " + name)
    value["inputs"] = inputs
    value["run_root"] = str(folder / "native_runs")
    value["output_ref"] = str(folder / "handoffs" / (name + ".json"))
    value["stage1_context"] = {
        "schema": SCHEMA,
        "experiment": str(path),
        "contract_sha256": settings["contract_sha256"],
        "stage": name,
        "codec_review_override": state.get("codec_review_override")
        if name in {"targets", "generator", "evaluate"}
        else None,
    }
    if settings["schema"] == STAGE2_SCHEMA:
        value["stage1_context"]["experiment_kind"] = "stage2"
    if name in {"codec", "generator"}:
        value["training"]["steps"] = state["targets"][name]
    return value


def verify_artifact(name, result, value):
    from ncm_lab.runtime.io import resolve_reference

    artifact = Path(result["artifact"]).resolve()
    if resolve_reference(value["output_ref"]).resolve() != artifact:
        raise ValueError("Stage 1 handoff differs from the exact completed artifact")
    info = read(artifact)
    if name in {"prepare", "teacher", "targets"}:
        from ncm_lab.data.artifacts import Dataset, LOCKED

        dataset = Dataset(
            artifact,
            {"prepare": "native_data_v1", "teacher": "native_features_v1", "targets": "native_targets_v1"}[
                name
            ],
        )
        if (
            dataset.manifest["identity"].get(
                "canvas", dataset.manifest["identity"].get("architecture", {}).get("canvas")
            )
            != value["architecture"]["canvas"]
        ):
            raise ValueError("Stage 1 source geometry mismatch")
        for _, row in dataset.iter_metadata():
            if "tensor" in row:
                target = (dataset.path / row["tensor"]["file"]).resolve()
                if (
                    not target.is_relative_to(dataset.path.resolve())
                    or digest(target) != row["tensor"]["sha256"]
                ):
                    raise ValueError("Stage 1 tensor cache changed")
        if name == "prepare":
            qualification = read(checked(result["qualification"]))
            if qualification["status"] != "qualified":
                raise ValueError("unqualified Stage 1 data cannot publish success")
            locked = Dataset(checked(result["evaluation_data"]), LOCKED)
            if set(locked.manifest["groups"]["test"]) & (
                set(dataset.manifest["groups"]["train"]) | set(dataset.manifest["groups"]["dev"])
            ):
                raise ValueError("locked evaluation group overlaps training/development")
    elif name in {"codec", "generator"}:
        from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE
        from ncm_lab.training.recovery_compatibility import verify_recovery

        if (
            result.get("completion", {}).get("code") != "objective_attained"
            or result["optimizer_steps"] != value["training"]["steps"]
        ):
            raise ValueError("Stage 1 training has not reached its exact total-update objective")
        if info.get("schema") != ("native_codec_v1" if name == "codec" else "native_generator_export_v1"):
            raise ValueError("wrong independent Stage 1 training artifact")
        weights = (artifact.parent / info["weights_manifest"]).resolve()
        if (
            not weights.is_relative_to(artifact.parent.resolve())
            or digest(weights) != info["weights_manifest_sha256" if name == "codec" else "weights_sha256"]
        ):
            raise ValueError("Stage 1 selected weights manifest changed")
        reader = CheckpointReader(weights, expected_kind=INFERENCE, identity=info["identity"])
        metadata = reader.materialize(reader.field("metadata"))
        if (
            metadata["architecture"] != value["architecture"]
            or metadata["stage1_contract"] != value["stage1"]
        ):
            raise ValueError("Stage 1 selection/reduction/geometry differs")
        current = CheckpointReader(result["checkpoint"])
        verify_recovery(current, metadata)
        if name == "generator":
            for filename, sha in info["tokenizer_files"].items():
                target = (artifact.parent / "tokenizer" / filename).resolve()
                if target.parent != (artifact.parent / "tokenizer").resolve() or digest(target) != sha:
                    raise ValueError("independent Stage 1 tokenizer changed")
    else:
        if info.get("schema") != "stage1_selected_evaluation_v1" or not info["selection_frozen_before_test"]:
            raise ValueError("invalid final Stage 1 evaluation receipt")
        checked(info["raw_outputs"])


def verify_record(name, record, experiment):
    from ncm_lab.runtime.provenance import validate_run
    from ncm_lab.config.stages import bind_inputs

    result = record["result"]
    run = Path(result["path"])
    checked(record["run"])
    if validate_run(run)["status"] != "completed":
        raise ValueError("Stage 1 stage does not have a sealed completed run")
    value = read(checked(record["config"]))
    if value["stage1_context"]["experiment"] != str(experiment):
        raise ValueError("Stage 1 stage belongs to a different experiment")
    manifest = read(experiment)
    contract = read(checked(manifest["contract"]))
    if value["stage1_context"]["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("Stage 1 stage contract differs from the experiment")
    if value != configuration(name, experiment, manifest, contract):
        raise ValueError("Stage 1 resolved stage settings differ from the immutable experiment recipe")
    bindings, missing = bind_inputs(value)
    if missing or read(run / "resolved_config.json") != {**value, "resolved_inputs": bindings}:
        raise ValueError("Stage 1 stage scientific configuration/input binding changed")
    if any(result.get(k) != v for k, v in read(run / "summary.json").items()):
        raise ValueError("Stage 1 result differs from sealed summary")
    verify_artifact(name, result, value)
    return value


def ensure_resources(path, state, settings):
    from ncm_lab.stage1.resources import derive

    result = read(checked(state["stages"]["prepare"]["receipt"]))["result"]
    resource_path = path.parent / "qualification/resources.json"
    if not state.get("resources"):
        if resource_path.exists():
            expected, _ = derive(settings, result, path.parent, persist=False)
            if read(resource_path) != expected:
                raise ValueError("interrupted resource derivation differs from the exact qualified contract")
            state["resources"] = reference(resource_path)
        else:
            _, state["resources"] = derive(settings, result, path.parent)
        state["history"].append({"event": "qualified_resource_derivation", "record": state["resources"]})
        atomic_json(path, state)
    derived = read(checked(state["resources"]))
    if derived["contract_sha256"] != settings["contract_sha256"] or derived[
        "qualified_data_sha256"
    ] != digest(result["artifact"]):
        raise ValueError("resource derivation source contract changed")
    if (
        not settings["stages"]["generator"]["model"].get("fixture")
        and shutil.disk_usage(path.parent).free
        < derived["peak_disk_estimate_including_shared_model_cache_and_headroom_bytes"]
    ):
        raise BudgetStop(
            "qualified peak disk/headroom allowance exceeds current free disk; preparation stays completed"
        )


def recover_completed(name, entry, path, state):
    if entry.get("receipt") or not entry["attempts"]:
        return
    attempt = Path(entry["attempts"][-1]["path"])
    if not attempt.is_relative_to(path.parent / "stages" / name):
        raise ValueError("recorded attempt belongs outside this experiment stage")
    if name in {"codec", "generator"}:
        pointer = capture_checkpoint(attempt, path.parent, read(checked(state["contract"])), name)
        if pointer and pointer != entry.get("last_checkpoint"):
            entry["last_checkpoint"] = pointer
            state["history"].append(
                {"event": "recovered_owned_launch_checkpoint", "stage": name, "checkpoint": pointer}
            )
            atomic_json(path, state)
    record_file = attempt / "result.json"
    if record_file.exists():
        record = read(record_file)
    elif all(
        (attempt / filename).is_file() for filename in ("exit.json", "stdout.json", "resolved_config.json")
    ):
        if read(attempt / "exit.json")["exit_status"]:
            return
        result = read(attempt / "stdout.json")
        if result.get("status") != "completed":
            return
        record = {
            "result": result,
            "exit_status": 0,
            "config": reference(attempt / "resolved_config.json"),
            "run": reference(Path(result["path"]) / "manifest.json"),
        }
    else:
        return
    if record.get("exit_status") or record["result"].get("status") != "completed":
        return
    if name in {"codec", "generator"} and record["result"].get("optimizer_steps") != state["targets"][name]:
        return
    verify_record(name, record, path)
    if not record_file.exists():
        write_json(record_file, record, exclusive=True)
    entry.update(status="completed", receipt=reference(record_file))
    if (
        name == "codec"
        and not record["result"]["stage1_diagnostics"]["codec_qualification"]["codec_qualified_for_generator"]
    ):
        entry["status"] = "review_required"
    state["history"].append(
        {"event": "recovered_sealed_stage_after_interruption", "stage": name, "receipt": entry["receipt"]}
    )
    atomic_json(path, state)


def capture_checkpoint(attempt, folder, settings, stage):
    launch_file = attempt / "launch.json"
    if not disk_path(launch_file).exists():
        return None
    launched = read(launch_file)
    run = Path(launched["path"]).resolve()
    if not run.is_relative_to(folder / "native_runs"):
        raise ValueError("child launch belongs to a different experiment")
    if checked(launched["launch"]).resolve() != run / "launch.json":
        raise ValueError("child launch receipt differs from the exact recorded native run")
    value = read(run / "resolved_config.json")
    context = value["stage1_context"]
    if (
        context["contract_sha256"] != settings["contract_sha256"]
        or context["experiment"] != str(folder / "manifest.json")
        or context["stage"] != stage
    ):
        raise ValueError("child launch scientific contract changed")
    pointer = run / "checkpoints/current.json"
    if disk_path(pointer).exists():
        from ncm_lab.training.checkpoints import CheckpointReader

        reader = CheckpointReader(pointer)
        return reference(reader.path)
    return None


def execute_child(value, attempt, path, state, settings, resume_checkpoint=None):
    from ncm_lab.runtime.pilot import stop_child

    attempt.mkdir(parents=True)
    config_path = attempt / "resolved_config.json"
    write_json(config_path, value, exclusive=True)
    command = [sys.executable, "-u", "-m", "ncm_lab.stage1.worker", "--config", str(config_path)]
    if resume_checkpoint:
        command += ["--resume-checkpoint", str(checked(resume_checkpoint))]
    name = value["stage1_context"]["stage"]
    print(
        f"Stage {name}: canvas={settings['canvas']['positions']} target={value['training']['steps'] if name in {'codec', 'generator'} else 'bounded preprocessing/evaluation'} batch={value['training']['batch_size']} accumulation={value['training']['accumulation']} dtype={value['dtype']} clocks=disabled\nLogs: {attempt}\nHandoff: {value['output_ref']}",
        flush=True,
    )
    with (attempt / "stdout.json").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=state["repository_root"],
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "NCM_REPO_ROOT": state["repository_root"],
                "NCM_CHILD_LAUNCH_RECORD": str(attempt / "launch.json"),
            },
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            if os.name == "nt"
            else 0,
            start_new_session=os.name != "nt",
            shell=False,
        )
        write_json(
            attempt / "process.json",
            {"pid": child.pid, "command": command, "execution_deadline": None},
            exclusive=True,
        )
        offset = 0
        try:
            while child.poll() is None:
                with (attempt / "stderr.log").open("rb") as stream:
                    stream.seek(offset)
                    text = stream.read(65536)
                    offset += len(text)
                if text:
                    print(text.decode("utf-8", errors="replace"), end="", file=sys.stderr, flush=True)
                if (
                    sum(p.stat().st_size for p in (path.parent / "stages").rglob("*") if p.is_file())
                    > settings["resources"]["max_controller_bytes"]
                ):
                    raise BudgetStop("Stage 1 retained child-log allowance exhausted")
                time.sleep(0.2)
        finally:
            stop_child(child)
            write_json(attempt / "exit.json", {"exit_status": child.returncode}, exclusive=True)
            pointer = capture_checkpoint(attempt, path.parent, settings, name)
            if pointer:
                state["stages"][name]["last_checkpoint"] = pointer
                atomic_json(path, state)
    try:
        result = read(attempt / "stdout.json")
    except (ValueError, OSError, __import__("yaml").YAMLError):
        result = {"status": "failed", "error": "child did not return a complete result; see retained logs"}
    with (attempt / "stderr.log").open("rb") as stream:
        stream.seek(offset)
        trailing = stream.read(settings["resources"]["max_controller_bytes"])
    if trailing:
        print(trailing.decode("utf-8", errors="replace"), end="", file=sys.stderr, flush=True)
    if child.returncode and result.get("status") == "completed":
        result = {
            **result,
            "status": "failed",
            "error": "nonzero child exit cannot publish successful Stage 1 work",
        }
    if result.get("status") not in {
        "completed",
        "failed",
        "blocked",
        "incomplete",
        "budget_stopped",
        "cancelled",
    }:
        result = {**result, "status": "failed", "error": "unrecognized Stage 1 child completion"}
    return {
        "result": result,
        "exit_status": child.returncode,
        "config": reference(config_path),
        "run": reference(Path(result["path"]) / "manifest.json")
        if result.get("status") == "completed"
        else None,
    }


def reports(path, state, settings):
    values = {
        name: read(checked(entry["receipt"]))["result"]
        for name, entry in state["stages"].items()
        if entry.get("receipt")
    }
    complete = state["status"] == "completed" and all(
        v["status"] == "completed" for v in state["stages"].values()
    )
    stage2 = settings["schema"] == STAGE2_SCHEMA
    stem = report_stem(settings)
    diagnostic = settings["qualification"]["diagnostic"]
    report = {
        "schema": f"{stem}_v1",
        "experiment": str(path),
        "status": state["status"],
        "workflow_completed": complete,
        "settings": settings,
        "targets": state["targets"],
        "stage_statuses": state["stages"],
        "history": state["history"],
        "qualification": read(checked(values["prepare"]["qualification"])) if "prepare" in values else None,
        "resource_derivation": read(checked(state["resources"])) if state.get("resources") else None,
        "codec": values.get("codec", {}).get("stage1_diagnostics"),
        "generator": values.get("generator", {}).get("stage1_diagnostics"),
        "evaluation": values.get("evaluate", {}).get("evaluation"),
        "codec_review_override": state.get("codec_review_override"),
        "locked_test_observed": state.get("locked_test_observed", False),
        "resource_observations": {
            name: {
                k: result.get(k)
                for k in (
                    "charged_work",
                    "elapsed_seconds",
                    "memory",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "wall_policy",
                )
            }
            for name, result in values.items()
        },
        "checkpoint_and_cache_io": {p.name: read(p) for p in (path.parent / "ledgers").glob("*-writes.json")},
        "failure": state.get("failure"),
        "quality": "Raw predictions require human relevance/coherence assessment; operational completion is not convergence or useful quality.",
        "origin": "software_fixture"
        if settings["stages"]["generator"]["model"].get("fixture")
        else "real_model",
        ("diagnostic_not_standard_stage2" if stage2 else "diagnostic_not_standard_stage1"): diagnostic,
    }
    if not report["qualification"] and isinstance(state.get("failure"), dict):
        qualification = state["failure"].get("result", {}).get("qualification")
        if isinstance(qualification, dict):
            report["qualification"] = read(checked(qualification))
    samples = []
    for attempt in state["stages"]["generator"]["attempts"]:
        launch = Path(attempt["path"]) / "launch.json"
        if launch.is_file():
            launched = read(launch)
            checked(launched["launch"])
            sample_file = Path(launched["path"]) / "monitor_samples.jsonl"
            if sample_file.is_file():
                samples.append(reference(sample_file))
    report["all_fixed_development_samples"] = samples
    folder = path.parent / "reports"
    folder.mkdir(exist_ok=True)
    atomic_json(folder / f"{stem}.json", report)
    lines = [
        f"# Stage {2 if stage2 else 1} single-pass experiment",
        "",
        f"Status: **{state['status']}**. Workflow completed: **{complete}**.",
        "",
        f"Canvas: {settings['canvas']['preset']} / {settings['canvas']['positions']}; training seed {settings['seed']}. Origin: {report['origin']}.",
        "",
        "| Stage | Status | Completed updates | Selected update |",
        "|---|---|---:|---:|",
    ]
    if diagnostic:
        lines[4:4] = [
            f"**Reduced-data diagnostic: not the standard Stage {2 if stage2 else 1} experiment or generalization evidence.**",
            "",
        ]
    for name in STAGES:
        result = values.get(name, {})
        lines.append(
            f"| {name} | {state['stages'][name]['status']} | {result.get('optimizer_steps', 'n/a')} | {result.get('selected_optimizer_step', 'n/a')} |"
        )
    for name in ("codec", "generator"):
        if report[name]:
            detail = report[name]
            lines += [
                "",
                f"## {name.capitalize()} learning diagnostics",
                "",
                f"Completed updates: {detail['completed_optimizer_steps']}; selected update: {detail['selected_optimizer_step']}; presentations: {detail['example_presentations']}.",
                "",
                "| Fixed-plan component | Initialization | Trained selection | Delta |",
                "|---|---:|---:|---:|",
            ]
            for key in (
                ("loss", "mse_full", "mse_cutoff", "head_kl")
                if name == "codec"
                else ("text_loss", "latent_loss", "loss")
            ):
                initial, selected = detail["initial"][key], detail["selected"][key]
                lines.append(f"| {key} | {initial:.7g} | {selected:.7g} | {selected - initial:.7g} |")
            if name == "codec":
                lines += [
                    "",
                    "Codec gate: **"
                    + detail["codec_qualification"]["status"]
                    + "**; full per-cutoff results, tolerances and raw scale statistics are retained in the JSON companion.",
                ]
            else:
                lines += [
                    "",
                    "Text CE improved on initialization: **"
                    + str(detail["improved_on_initialization"])
                    + "**. Relevance/coherence still requires review of all retained samples.",
                ]
    if report["evaluation"]:
        lines += [
            "",
            "## Frozen selected evaluation",
            "",
            "```json",
            json.dumps(report["evaluation"]["splits"], indent=2),
            "```",
        ]
    lines += [
        "",
        "## Interpretation and resources",
        "",
        report["quality"],
        "Codec improvement uses a versioned operational component gate, not statistical significance. Test data never fits normalization, targets or checkpoint selection.",
        "Losses average actual microbatch objectives; effective batch is not a global-token reduction. Unique data and repeated presentations are separate in the JSON report.",
        "Cumulative checkpoint/cache I/O survives pruning and retry. Peak retained storage and baseline memory are different quantities; activations/kernel/recovery peaks require measured hardware evidence.",
        "Training clocks are disabled; elapsed observations and finite non-time safeguards remain active.",
    ]
    if state.get("failure"):
        lines += ["", "Failure/review details: " + str(state["failure"])]
    (folder / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": reference(folder / f"{stem}.json"),
        "markdown": reference(folder / f"{stem}.md"),
    }


def run(
    experiment,
    *,
    stage="all",
    resume_checkpoint=None,
    allow_codec_review=False,
    reason=None,
    extend_generator_steps=None,
):
    path, state, settings = load(experiment)
    if settings["schema"] == STAGE2_SCHEMA and (allow_codec_review or extend_generator_steps is not None):
        raise ValueError("Stage 2 requires its codec gate and the original fixed generator target")
    if stage not in {*STAGES, "all"}:
        raise ValueError("unknown Stage 1 stage")
    if allow_codec_review and not reason:
        raise ValueError("diagnostic codec review override requires an explicit recorded --reason")
    if resume_checkpoint is not None:
        if stage not in {"codec", "generator"} or not Path(resume_checkpoint).resolve().is_relative_to(
            path.parent / "native_runs"
        ):
            raise ValueError("explicit recovery requires this experiment's codec/generator checkpoint")
        from ncm_lab.training.checkpoints import CheckpointReader

        reader = CheckpointReader(resume_checkpoint)
        metadata = reader.materialize(reader.field("metadata"))
        if (
            metadata.get("component") != stage
            or metadata.get("stage1_contract") != settings["stages"][stage]["stage1"]
        ):
            raise ValueError("explicit Stage 1 recovery component/selection policy mismatch")
    with exclusive_writer(path.parent / "execution.lock"):
        _, state, settings = load(path)
        original_state = canonical_hash(state)
        if extend_generator_steps is not None:
            if (
                not reason
                or state["stages"]["evaluate"]["status"] != "pending"
                or state.get("locked_test_observed")
            ):
                raise ValueError("extend only before locked evaluation, with a recorded reason")
            if (
                type(extend_generator_steps) is not int
                or not state["targets"]["generator"]
                < extend_generator_steps
                <= settings["resources"]["max_generator_steps"]
            ):
                raise ValueError("extension must be a larger TOTAL target within the planned finite ceiling")
            state["history"].append(
                {
                    "event": "explicit_total_step_extension",
                    "from": state["targets"]["generator"],
                    "to": extend_generator_steps,
                    "reason": reason,
                }
            )
            state["targets"]["generator"] = extend_generator_steps
            entry = state["stages"]["generator"]
            if entry.get("receipt"):
                entry.setdefault("prior_receipts", []).append(entry.pop("receipt"))
            entry["status"] = "pending"
            atomic_json(path, state)
        requested = STAGES if stage == "all" else (stage,)
        invocation = path.parent / "controller" / identifier()
        invocation.mkdir(parents=True)
        meter = Meter(
            Budget(
                "s1-controller-" + settings["contract_sha256"][:40],
                enforce_wall_time_limits=False,
                max_steps=7,
                max_phase_steps=settings["resources"]["max_controller_actions"],
                max_storage_bytes=1000000,
                max_phase_storage_bytes=settings["resources"]["max_controller_bytes"],
            ),
            invocation,
            path.parent / "ledgers",
        )
        already = []
        try:
            meter.charge(steps=1)
            for name in requested:
                entry = state["stages"][name]
                recover_completed(name, entry, path, state)
                if entry["status"] == "completed":
                    verify_record(name, read(checked(entry["receipt"])), path)
                    if name == "prepare":
                        ensure_resources(path, state, settings)
                    already.append(name)
                    continue
                if name == "codec" and entry["status"] == "review_required":
                    record = read(checked(entry["receipt"]))
                    verify_record(name, record, path)
                    gate = record["result"]["stage1_diagnostics"]["codec_qualification"]
                    if not allow_codec_review or not gate["artifacts_valid"]:
                        state.update(
                            status="review_required",
                            failure="Codec completed but did not pass the fixed initialization/component comparison; inspect its report.",
                        )
                        break
                    override = path.parent / "reports" / ("codec-override-" + identifier() + ".json")
                    write_json(
                        override,
                        {"reason": reason, "gate": gate, "diagnostic_override": True},
                        exclusive=True,
                    )
                    state["codec_review_override"] = reference(override)
                    entry["status"] = "completed"
                    state["history"].append(
                        {"event": "diagnostic_codec_gate_override", "record": reference(override)}
                    )
                    atomic_json(path, state)
                    continue
                for dependency in DEPENDENCIES[name]:
                    dependency_entry = state["stages"][dependency]
                    if dependency_entry["status"] != "completed":
                        raise ValueError(
                            f"{name} requires completed, qualified {dependency}; explicit wrappers do not run prerequisites"
                        )
                    verify_record(dependency, read(checked(dependency_entry["receipt"])), path)
                if name != "prepare":
                    ensure_resources(path, state, settings)
                meter.charge(steps=1)
                value = configuration(name, path, state, settings)
                pointer = entry.get("last_checkpoint")
                if resume_checkpoint is not None:
                    if name not in {"codec", "generator"}:
                        raise ValueError("--resume-checkpoint applies only to codec/generator")
                    from ncm_lab.training.checkpoints import CheckpointReader

                    candidate = Path(resume_checkpoint).resolve()
                    if not candidate.is_relative_to(path.parent / "native_runs"):
                        raise ValueError("explicit recovery checkpoint belongs outside this experiment")
                    reader = CheckpointReader(candidate)
                    pointer = reference(reader.path)
                if pointer:
                    checked(pointer)
                attempt = path.parent / "stages" / name / identifier()
                entry["attempts"].append(
                    {
                        "path": str(attempt),
                        "resume": pointer,
                        "restart_from_recorded_initialization": not pointer and bool(entry["attempts"]),
                    }
                )
                entry["status"] = "running"
                if name == "evaluate":
                    state["locked_test_observed"] = True
                state.update(status="running", active_stage=name, failure=None)
                atomic_json(path, state)
                record = execute_child(value, attempt, path, state, settings, pointer)
                write_json(attempt / "result.json", record, exclusive=True)
                result = record["result"]
                if result["status"] != "completed" or record["exit_status"]:
                    entry["status"] = result["status"] if result["status"] != "completed" else "failed"
                    state.update(status=entry["status"], failure={"stage": name, "result": result})
                    break
                verify_record(name, record, path)
                entry.update(status="completed", receipt=reference(attempt / "result.json"))
                if name in {"codec", "generator"}:
                    from ncm_lab.training.checkpoints import CheckpointReader

                    entry["last_checkpoint"] = reference(CheckpointReader(result["checkpoint"]).path)
                if name == "prepare":
                    atomic_json(path, state)
                    ensure_resources(path, state, settings)
                if (
                    name == "codec"
                    and not result["stage1_diagnostics"]["codec_qualification"][
                        "codec_qualified_for_generator"
                    ]
                ):
                    entry["status"] = "review_required"
                    state.update(
                        status="review_required",
                        failure="Codec training completed; component gate needs review. No targets or generator were started.",
                    )
                    break
                state["history"].append(
                    {"event": "stage_completed", "stage": name, "receipt": entry["receipt"]}
                )
                atomic_json(path, state)
            else:
                if len(already) != len(requested) or all(
                    e["status"] == "completed" for e in state["stages"].values()
                ):
                    state.update(
                        status="completed"
                        if all(e["status"] == "completed" for e in state["stages"].values())
                        else "ready",
                        active_stage=None,
                        failure=None,
                    )
        except KeyboardInterrupt:
            state.update(status="cancelled", failure="operator cancellation; only owned child tree stopped")
            if state.get("active_stage"):
                state["stages"][state["active_stage"]]["status"] = "cancelled"
        except Exception as exc:
            state.update(
                status="budget_stopped" if isinstance(exc, BudgetStop) else "failed",
                failure=type(exc).__name__ + ": " + str(exc),
            )
            if state.get("active_stage") and state["stages"][state["active_stage"]]["status"] == "running":
                state["stages"][state["active_stage"]]["status"] = state["status"]
        finally:
            meter.close()
            if canonical_hash(state) != original_state or not state.get("reports"):
                state["reports"] = reports(path, state, settings)
                atomic_json(path, state)
            else:
                for report in state["reports"].values():
                    checked(report)
            write_json(
                invocation / "result.json",
                {"status": state["status"], "requested": list(requested), "already_completed": already},
                exclusive=True,
            )
        return {
            "status": "already_completed" if len(already) == len(requested) else state["status"],
            "experiment": str(path),
            "stages": {n: e["status"] for n, e in state["stages"].items()},
            "report": str(path.parent / "reports" / f"{report_stem(settings)}.md"),
            **(
                {"asset_acquisition_if_missing": read(checked(state["plan"]))["missing_asset_instruction"]}
                if state.get("failure")
                else {}
            ),
            "failure": (
                {
                    "stage": state["failure"].get("stage"),
                    **{
                        k: state["failure"].get("result", {}).get(k)
                        for k in ("status", "error", "reason", "blockers", "checkpoint")
                        if state["failure"].get("result", {}).get(k) is not None
                    },
                }
                if isinstance(state.get("failure"), dict)
                else state.get("failure")
            ),
        }


def run_ar_reference(experiment, max_new_tokens=None):
    """Separate explicit inference-only action, excluded from default workload."""
    from dataclasses import asdict

    path, state, settings = load(experiment)
    cap = settings["canvas"]["positions"] if max_new_tokens is None else max_new_tokens
    if type(cap) is not int or not 1 <= cap <= settings["canvas"]["positions"]:
        raise ValueError("AR reference cap must be positive and at most the selected answer canvas")
    with exclusive_writer(path.parent / "execution.lock"):
        for name in ("prepare", "generator"):
            if state["stages"][name]["status"] != "completed":
                raise ValueError("AR reference requires prepared locked data and a frozen selected generator")
            verify_record(name, read(checked(state["stages"][name]["receipt"])), path)
        value = configuration("evaluate", path, state, settings)
        count = settings["qualification"]["test_examples"]
        work = count * cap * (512 + cap)
        value["budget"] = asdict(
            Budget(
                "s1-ar-" + settings["contract_sha256"][:32] + "-" + str(cap),
                enforce_wall_time_limits=False,
                max_steps=count,
                max_phase_steps=count * 3,
                max_source_tokens=work,
                max_phase_source_tokens=work * 3,
                max_answer_positions=count * cap,
                max_phase_answer_positions=count * cap * 3,
                max_canvas_positions=count * cap,
                max_phase_canvas_positions=count * cap * 3,
                max_storage_bytes=settings["resources"]["max_controller_bytes"],
                max_phase_storage_bytes=settings["resources"]["max_controller_bytes"] * 3,
            )
        )
        value["stage1_context"].update(action="ar_reference", reference_max_new_tokens=cap)
        tag = identifier()
        value["output_ref"] = str(path.parent / "handoffs" / ("ar_reference-" + tag + ".json"))
        attempt = path.parent / "stages/ar_reference" / tag
        # Any locked-test inference attempt closes the training-extension window,
        # including when it fails before the ordinary NCM evaluation is run.
        state["locked_test_observed"] = True
        atomic_json(path, state)
        record = execute_child(value, attempt, path, state, settings)
        write_json(attempt / "result.json", record, exclusive=True)
        if record["exit_status"] or record["result"].get("status") != "completed":
            return {"status": record["result"].get("status", "failed"), "logs": str(attempt)}
        from ncm_lab.runtime.provenance import validate_run

        if validate_run(record["result"]["path"])["status"] != "completed":
            raise ValueError("AR reference did not seal a completed inference run")
        artifact = read(record["result"]["artifact"])
        if artifact.get("schema") != "stage1_optional_ar_reference_v1" or artifact["used_to_repair_ncm"]:
            raise ValueError("invalid separate AR reference artifact")
        state.setdefault("optional_ar_references", []).append(reference(attempt / "result.json"))
        atomic_json(path, state)
        return {
            "status": "completed",
            "reference": record["result"]["artifact"],
            "experiment_status": state["status"],
        }
