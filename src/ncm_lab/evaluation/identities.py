"""Separate immutable observation, varied treatment, and shared condition identities."""

from pathlib import Path
from ncm_lab.runtime.io import canonical_hash, digest, read


def model_identity(model):
    return {
        k: model.get(k)
        for k in ("target_id", "target_revision", "tokenizer_id", "tokenizer_revision", "thinking")
    }


def run_identity(run, config, policy_hash):
    inputs = read(run.path / "inputs.json") if (run.path / "inputs.json").exists() else []
    selected = next(
        (
            i["identity"]
            for i in inputs
            if Path(i["identity"]["path"]).resolve() == Path(config["data_path"]).resolve()
        ),
        None,
    )
    data_sha = selected["sha256"] if selected else digest(config["data_path"])
    request_set = (
        selected["selected_ids"][: config["max_requests"]]
        if selected
        else {"file": data_sha, "cap": config["max_requests"]}
    )
    environment = read(run.path / "environment.json")
    runtime = {
        k: environment.get(k)
        for k in ("profile", "platform", "cpu_architecture", "logical_cpu_count", "gpus", "packages", "torch")
    }
    runtime.update(
        cpu_threads=config.get("cpu_threads"),
        precision=config["precision"],
        device=config["device"],
        backend="transformers_incremental_sdpa",
        warmup_requests=config.get("warmup_requests", 0),
        concurrency=1,
    )
    treatment = {
        "method": config["method"],
        "architecture": read(run.path / "resolved_method.json"),
        "checkpoint_sha256": config.get("checkpoint_sha256"),
        "codec_sha256": config.get("codec_sha256"),
        "codec_objective": config.get("codec_objective"),
        "training_seed": config.get("seed"),
        "codec_seed": config.get("codec_seed"),
        "data_seed": config.get("data_seed"),
        "treatment_id": config.get("treatment_id", config["method"]),
        "lineage": [
            {key: parent[key] for key in ("run_id", "manifest_sha256")}
            for parent in read(run.path / "dependencies.json")
        ]
        if (run.path / "dependencies.json").exists()
        else [],
    }
    condition = {
        "target": model_identity(config["model"]),
        "dataset": config["dataset"],
        "data_sha256": data_sha,
        "split": config["split"],
        "request_set_sha256": canonical_hash(request_set),
        "policy_hash": policy_hash,
        "prompt_policy": config["prompt_policy"],
        "runtime": runtime,
        "data_tier": config.get("data_tier"),
        "data_ladder_version": config.get("data_ladder_version", 3),
        "training_subset_sha256": read(config["checkpoint_selection"]).get("admitted_training_hash")
        if config.get("checkpoint_selection")
        else None,
        "comparison_regime": config.get("comparison", "equal_data"),
        "budget_cap": config.get("total_new_training_wall_seconds"),
        "scope": "full_test" if config.get("full_official_split") else "bounded_subset",
        "result_origin": config["result_origin"],
    }
    return {
        "schema_version": 3,
        "run_id": run.id,
        "launch_sha256": digest(run.path / "launch.json"),
        "treatment": treatment,
        "treatment_sha256": canonical_hash(treatment),
        "condition": condition,
        "condition_sha256": canonical_hash(condition),
    }


def observation_identity(identity, original_id, repetition):
    return {
        **identity,
        "observation": {
            "run_id": identity["run_id"],
            "launch_sha256": identity["launch_sha256"],
            "request_id": original_id,
            "timing_block": repetition,
            "measurement": "complete_request",
        },
    }


def validate_record_identity(row, expected):
    actual = row.get("identities")
    if actual is None:
        raise ValueError("prediction lacks versioned identities; use explicit diagnostic legacy reanalysis")
    for key in ("run_id", "launch_sha256", "treatment", "condition", "treatment_sha256", "condition_sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError("prediction/manifest identity disagrees for " + key)
    if actual["observation"]["request_id"] != row["original_id"]:
        raise ValueError("observation request identity mismatch")
    if any(actual["observation"].get(k) != expected[k] for k in ("run_id", "launch_sha256")):
        raise ValueError("observation launch identity mismatch")
