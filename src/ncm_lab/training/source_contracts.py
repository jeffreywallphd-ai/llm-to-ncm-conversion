"""Content-bound teacher semantics, distinct from location and execution precision."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from ncm_lab.runtime.io import canonical_hash, read, digest, write_json
from ncm_lab.decoding.policy import GenerationPolicy, resolve_policy

STATE = "post_final_norm_prediction_position"
BOUNDARY = "response_only_known_anchor_v2"


def model_identity(model):
    return {
        "target_id": model["target_id"],
        "target_revision": model["target_revision"],
        "tokenizer_id": model.get("tokenizer_id", model["target_id"]),
        "tokenizer_revision": model["tokenizer_revision"],
        "thinking": model.get("thinking", False),
    }


def policy_record(value):
    value = dict(value)
    value["eos_ids"] = tuple(value.get("eos_ids", ()))
    value["stop_sequences"] = tuple(tuple(s) for s in value.get("stop_sequences", ()))
    policy = GenerationPolicy(**value)
    return asdict(policy), policy.digest


def semantic_policy(value):
    policy, _ = policy_record(value)
    return {k: v for k, v in policy.items() if k not in {"max_new_tokens", "precision"}}


def make_contract(model, policy, split, hidden_size=None, assets=None):
    policy, sha = policy_record(policy)
    return {
        "schema_version": 3,
        "model": model_identity(model),
        "generation_policy": policy,
        "policy_sha256": sha,
        "template_hash": policy["template_hash"],
        "thinking": False,
        "state_definition": STATE,
        "horizon": 15,
        "partition_levels": [4, 8, 15],
        "hidden_size": hidden_size,
        "head_transform": "frozen_linear_head_on_post_final_norm_no_second_norm",
        "boundary_convention": BOUNDARY,
        "tokenization": "pinned_chat_template_user_then_generation_prompt",
        "split": split,
        "schema": "teacher",
        "asset_receipts": assets or [],
        "fixture": False,
    }


def expected_policy(config):
    from transformers import AutoTokenizer, GenerationConfig

    model = config["model"]
    location = model.get("target_path", model["target_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        location, revision=model["tokenizer_revision"], local_files_only=True, trust_remote_code=False
    )
    generation = GenerationConfig.from_pretrained(
        location, revision=model["target_revision"], local_files_only=True
    )
    return asdict(
        resolve_policy(
            SimpleNamespace(generation_config=generation),
            tokenizer,
            {"tokenizer_revision": model["tokenizer_revision"], **config.get("source_generation_policy", {})},
        )
    )


def validate_contract(contract, model, policy=None):
    if contract.get("schema_version") != 3 or contract.get("fixture"):
        raise ValueError(
            "research source requires a current non-fixture teacher contract; migrate trustworthy legacy evidence explicitly"
        )
    if contract["model"] != model_identity(model):
        raise ValueError("teacher/tokenizer source identity mismatch")
    for field, expected in {
        "state_definition": STATE,
        "boundary_convention": BOUNDARY,
        "horizon": 15,
        "partition_levels": [4, 8, 15],
        "thinking": False,
    }.items():
        if contract.get(field) != expected:
            raise ValueError("teacher source contract mismatch: " + field)
    _, sha = policy_record(contract["generation_policy"])
    if sha != contract["policy_sha256"]:
        raise ValueError("invalid canonical teacher policy digest")
    if policy is not None and semantic_policy(policy) != semantic_policy(contract["generation_policy"]):
        raise ValueError("teacher template or semantic generation policy mismatch")


def admit(path, split, config):
    from ncm_lab.training.records import load_records

    path = Path(path)
    fixture = config.get("result_origin") == "test_fixture"
    rows = load_records(path, split, allow_fixture=fixture)
    contract_path = path.with_suffix(".contract.json")
    supplied = config.get("teacher_contracts", {}).get(split)
    if supplied:
        contract_path = Path(supplied)
    if not contract_path.exists():
        if fixture and all(r["origin"] == "software_fixture" for r in rows):
            return rows, {"schema_version": 3, "fixture": True, "model": model_identity(config["model"])}
        raise ValueError(
            "teacher contract missing; ncm data migrate-teacher recovers only verified original provenance"
        )
    artifact = read(contract_path)
    if artifact.get("migration_version") == 3:
        validate_migration(artifact)
    if artifact.get("fixture") and not fixture:
        raise ValueError("fixture teacher migration is diagnostic-only")
    if artifact["data_sha256"] != digest(path) or not artifact.get("complete"):
        raise ValueError("teacher contract data identity/incomplete artifact mismatch")
    contract = artifact["contract"]
    validate_contract(contract, config["model"])
    if contract["split"] != split or canonical_hash(contract) != artifact["contract_sha256"]:
        raise ValueError("teacher contract split/digest mismatch")
    expected = expected_policy(config)
    validate_contract(contract, config["model"], expected)
    for row in rows:
        validate_row(row, contract, legacy=artifact.get("migration_version") == 3)
        if artifact.get("migration_version") == 3:
            row["legacy_source_migrated"] = True
    return rows, contract


def validate_row(row, contract, legacy=False):
    _, sha = policy_record(row["generation_policy"])
    if sha != row.get("policy_hash"):
        raise ValueError("invalid recorded canonical teacher policy hash")
    if (
        row.get("teacher_revision") != contract["model"]["target_revision"]
        or row.get("tokenizer_revision") != contract["model"]["tokenizer_revision"]
        or row.get("thinking") is not False
    ):
        raise ValueError("teacher/tokenizer/thinking record source mismatch")
    if semantic_policy(row["generation_policy"]) != semantic_policy(contract["generation_policy"]):
        raise ValueError("recorded teacher template/policy differs from source contract")
    recorded_policy, _ = policy_record(row["generation_policy"])
    reference = {
        **contract,
        "split": row["split"],
        "generation_policy": recorded_policy,
        "policy_sha256": sha,
        "template_hash": recorded_policy["template_hash"],
    }
    if not legacy and row.get("source_contract_sha256") != canonical_hash(reference):
        raise ValueError("record source contract reference mismatch")


def migrate_teacher(path, output, diagnostic_fixture=False):
    """Recover only from the immutable original teacher run, never current settings."""
    from ncm_lab.runtime.provenance import validate_run
    from ncm_lab.training.records import load_records

    path = Path(path)
    original = validate_run(path.parent)
    config = read(path.parent / "config.json")
    if (
        original["run_kind"] != "teacher"
        or original["status"] != "completed"
        or original["result_origin"] != "measured"
        and not diagnostic_fixture
    ):
        raise ValueError("legacy migration requires completed original measured teacher provenance")
    split = "test" if config.get("purpose") == "test_evaluation" else config["purpose"]
    rows = load_records(path, split, allow_fixture=diagnostic_fixture)
    contract = make_contract(config["model"], rows[0]["generation_policy"], split)
    validate_contract(contract, config["model"])
    for row in rows:
        validate_row(row, contract, legacy=True)
    artifact = {
        "fixture": original["result_origin"] == "test_fixture",
        "migration_version": 3,
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "data_sha256": digest(path),
        "complete": True,
        "original_manifest_sha256": digest(path.parent / "manifest.json"),
        "original_manifest_path": str((path.parent / "manifest.json").resolve()),
        "recovered_from": "original immutable config and recorded canonical policies; original bytes preserved",
    }
    write_json(output, artifact, exclusive=True)
    return artifact


def validate_migration(artifact):
    from ncm_lab.runtime.provenance import validate_run

    manifest = Path(artifact["original_manifest_path"])
    if digest(manifest) != artifact["original_manifest_sha256"]:
        raise ValueError("legacy teacher migration original provenance changed")
    run = validate_run(manifest.parent)
    if run["run_kind"] != "teacher" or run["status"] != "completed":
        raise ValueError("legacy teacher migration requires completed original provenance")


def validate_codec_source(codec, model=None, target=None):
    contract = getattr(codec, "source_contract", None)
    if contract is None:
        raise ValueError(
            "selected codec lacks source dependency contract; explicit legacy migration or diagnostic-only loading required"
        )
    if model is not None and not contract.get("fixture"):
        validate_contract(contract, model)
    if target is not None:
        if contract.get("hidden_size") not in {None, target.config.hidden_size}:
            raise ValueError("codec teacher-state width mismatch")
        expected = getattr(target, "ncm_source_model", None)
        if expected and not contract.get("fixture"):
            validate_contract(contract, expected)
