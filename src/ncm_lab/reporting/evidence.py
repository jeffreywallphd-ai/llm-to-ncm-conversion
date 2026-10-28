"""Append-only assessments, distinct from immutable runs and author selection."""

from datetime import datetime, timezone
from pathlib import Path
import uuid
import json
from ncm_lab.runtime.io import read, write_json, digest, root
from ncm_lab.runtime.provenance import validate_run


def attest(path, claim_class="correctness", analysis_receipt=None):
    path = Path(path)
    manifest = validate_run(path)
    config, metrics = read(path / "config.json"), read(path / "metrics.json")
    reasons = []
    if manifest["status"] != "completed":
        reasons.append("run incomplete")
    if manifest["result_origin"] != "measured":
        reasons.append("fixtures and diagnostics are not research evidence")
    if not (path / "inputs.json").exists() or not read(path / "inputs.json"):
        reasons.append("consumed input receipts missing")
    else:
        from ncm_lab.data.artifacts import verify_input

        for item in read(path / "inputs.json"):
            verify_input(item["receipt"])
    if claim_class in {"performance", "full_test"} and config.get("parity_run"):
        from ncm_lab.experiments.runner import prerequisites

        reasons.extend(prerequisites(config))
        parity_path = Path(config["parity_run"])
        parity = validate_run(parity_path, require_current_code=True)
        parity_metrics = read(parity_path / "metrics.json")
        if (
            parity["status"] != "completed"
            or parity["result_origin"] != "measured"
            or not parity_metrics.get("all_equal")
            or parity_metrics.get("generated_tokens_compared", 0) <= 0
        ):
            reasons.append("unverified real-model parity")
        if metrics.get("policy_hash") != parity_metrics.get("policy_hash"):
            reasons.append("evaluation and parity policies differ")
    if claim_class == "correctness":
        if not metrics.get("all_equal") or not metrics.get("policy_hash"):
            reasons.append("common-policy parity evidence missing")
        if manifest["run_kind"] == "verification" and metrics.get("generated_tokens_compared", 0) <= 0:
            reasons.append("zero-output equality is not meaningful model parity evidence")
    elif claim_class == "representation":
        if (
            manifest["run_kind"] != "codec_evaluate"
            or metrics.get("diagnostics_schema_version") != 3
            or metrics.get("purpose") != "locked_test_diagnostic"
            or config.get("split") != "test"
        ):
            reasons.append(
                "held-out selected-codec test diagnostics required; development selection is not final representation evidence"
            )
        dependencies = metrics.get("codec_dependencies", {})
        if not dependencies.get("weights_sha256") or not dependencies.get("normalization"):
            reasons.append("codec search/statistics provenance incomplete")
    elif claim_class == "mechanism":
        if (
            manifest["run_kind"] != "latent_diagnostics"
            or metrics.get("diagnostic_requests", 0) <= 0
            or metrics.get("oracle_available_in_generation") is not False
            or config.get("split") != "test"
        ):
            reasons.append("held-out causal/oracle diagnostics and intervention records missing")
        if not config.get("checkpoint_selection") or not config.get("codec_selection"):
            reasons.append("selected codec/student provenance missing")
    elif claim_class == "performance":
        if not config.get("parity_run"):
            reasons.append("same-configuration released-model parity missing")
        predictions = path / "predictions.jsonl"
        rows = (
            [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            if predictions.is_file()
            else []
        )
        warmups = read(path / "warmup.json") if (path / "warmup.json").exists() else []
        if (
            not rows
            or not config.get("warmup_requests")
            or not any(w.get("seconds") is not None and not w.get("error") for w in warmups)
        ):
            reasons.append("recorded warm steady-state observations missing")
        if any(
            r.get("generation_seconds") is None or r.get("target_processed_positions", 0) <= 0
            for r in rows
            if not r.get("error")
        ):
            reasons.append("decoder resource/timing telemetry missing")
        if not metrics.get("resources"):
            reasons.append("execution resource record missing")
    elif claim_class == "full_test":
        if config.get("split") != "test" or not config.get("full_official_split"):
            reasons.append("bounded subset is not a full official test")
        if not config.get("parity_run"):
            reasons.append("same-configuration correctness prerequisite missing")
        predictions = path / "predictions.jsonl"
        if not predictions.exists() or not metrics.get("requests"):
            reasons.append("official endpoint observations missing")
        elif (path / "inputs.json").exists():
            inputs = [i["identity"] for i in read(path / "inputs.json") if i["identity"]["split"] == "test"]
            rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            official = [
                i
                for i in inputs
                if i["source"].get("scope") == "full_official_split"
                and i["source"].get("official_split_record_count") == len(i["selected_ids"])
            ]
            if len(official) != 1 or len(rows) != len(official[0]["selected_ids"]):
                reasons.append("complete official split identity/cardinality not established")
            elif [r["original_id"] for r in rows] != official[0]["selected_ids"]:
                reasons.append("official split request identities/order differ")
            if not any(r.get("official_scores") and not r.get("error") for r in rows):
                reasons.append("no successfully scored official endpoint observations")
    else:
        raise ValueError("unknown evidence claim class")
    analysis = None
    if claim_class in {"performance", "full_test"} and config.get("checkpoint"):
        training_path = Path(config["checkpoint"]).parent / "config.json"
        training = read(training_path) if training_path.exists() else {}
        audit = config.get("overlap_audit") or training.get("overlap_audit")
        if audit:
            from ncm_lab.data.overlap import validate_audit

            identities = read(path / "inputs.json") if (path / "inputs.json").exists() else []
            for item in identities:
                if Path(item["identity"]["path"]).resolve() == Path(config["data_path"]).resolve():
                    validate_audit(audit, item["identity"]["sha256"])
        else:
            reasons.append("new adaptation corpus has no declared benchmark overlap audit")
    if claim_class in {"performance", "full_test", "representation", "mechanism"}:
        analysis_path = Path(analysis_receipt) if analysis_receipt else None
        if analysis_path is not None and analysis_path.exists():
            analysis_record = read(analysis_path)
            if claim_class in {"performance", "full_test"} and (
                not analysis_record.get("binding") or analysis_record.get("legacy_diagnostic")
            ):
                reasons.append(
                    "final research requires locked contrast/run bindings, not exploratory or legacy analysis"
                )
            if not any(
                i["run_id"] == manifest["run_id"] and i["manifest_sha256"] == digest(path / "manifest.json")
                for i in analysis_record["inputs"]
            ):
                reasons.append("analysis receipt does not bind this run")
            for name, sha in analysis_record["files"].items():
                artifact = (analysis_path.parent / name).resolve()
                if not artifact.is_relative_to(analysis_path.parent.resolve()) or digest(artifact) != sha:
                    raise ValueError("analysis output changed")
            derived = analysis_record.get("derived_tables_manifest")
            if derived:
                if digest(derived["path"]) != derived["sha256"]:
                    raise ValueError("derived analysis receipt changed")
                details = read(derived["path"])
                for name, sha in details["files"].items():
                    artifact = (Path(derived["path"]).parent / name).resolve()
                    if (
                        not artifact.is_relative_to(Path(derived["path"]).parent.resolve())
                        or digest(artifact) != sha
                    ):
                        raise ValueError("derived analysis output changed")
            analysis = {"path": str(analysis_path), "sha256": digest(analysis_path)}
        else:
            reasons.append("explicit immutable analysis receipt missing; provide --analysis")
    assessment = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "manifest_sha256": digest(path / "manifest.json"),
        "claim_class": claim_class,
        "eligible": not reasons,
        "reasons": reasons,
        "assessor": "automated_gate_v1",
        "human_decision": None,
        "utc": datetime.now(timezone.utc).isoformat(),
        "method": config.get("method"),
        "policy": metrics.get("policy_hash"),
        "scope": config.get("split"),
        "analysis": analysis,
        "protocol_sha256": manifest["protocol_hash"],
        "checkpoint_sha256": config.get("checkpoint_sha256"),
        "data_dependencies": [i["identity"]["sha256"] for i in read(path / "inputs.json")]
        if (path / "inputs.json").exists()
        else [],
        "limitations": [
            "eligibility applies only to this evidence class",
            "Transformers incremental SDPA reference; no serving-engine or concurrency claim",
        ],
    }
    folder = "attestations" if manifest["result_origin"] == "measured" else "engineering_attestations"
    output = root() / "results/generated" / folder / (uuid.uuid4().hex + ".json")
    write_json(output, assessment, exclusive=True)
    return {"path": str(output), **assessment}


def select_evidence(study, run_path, attestation_path, claim_ids):
    from ncm_lab.runtime.io import canonical_hash

    run_path, attestation_path = Path(run_path), Path(attestation_path)
    manifest = validate_run(run_path)
    assessment = read(attestation_path)
    study_path = root() / "publications" / study / "study.yaml"
    spec = read(study_path)
    config = read(run_path / "config.json")
    if manifest["experiment_id"] not in set(spec["required_experiments"] + spec["shared_controls"]):
        raise ValueError("experiment outside selected study")
    if (
        config.get("method") not in spec["eligible_methods"]
        or config.get("dataset") not in spec["eligible_datasets"]
    ):
        raise ValueError("method/dataset outside selected study")
    claims = {c["id"]: c for c in read(root() / "publications" / study / "claims.yaml")["claims"]}
    if not claim_ids or any(
        c not in claims or manifest["experiment_id"] not in claims[c]["experiment_ids"] for c in claim_ids
    ):
        raise ValueError("selection requires compatible explicit claim IDs")
    if any(
        assessment["claim_class"]
        not in claims[c].get("required_evidence_classes", ["full_test", "performance"])
        for c in claim_ids
    ):
        raise ValueError("attestation class does not support the selected claim")
    entry = {
        "run_id": manifest["run_id"],
        "manifest_sha256": digest(run_path / "manifest.json"),
        "attestation": str(attestation_path),
        "attestation_sha256": digest(attestation_path),
        "claim_ids": claim_ids,
        "method": config.get("method"),
        "dataset": config.get("dataset"),
        "split": config.get("split"),
        "contribution": "shared_control"
        if manifest["experiment_id"] in spec["shared_controls"]
        else "new_study_evidence",
        "selection_origin": "explicit CLI request; no human reviewer identity inferred",
    }
    validate_selection(entry, run_path)
    identity = canonical_hash(entry)
    record = root() / "results/generated/selections" / (identity + ".json")
    write_json(record, {"entry": entry, "assessment": assessment}, exclusive=True)
    spec["selected_runs"].append(entry)
    write_json(study_path, spec)
    return {"selection": str(record), "entry": entry}


def validate_selection(selection, path):
    assessment = read(selection["attestation"])
    if digest(selection["attestation"]) != selection["attestation_sha256"]:
        raise ValueError("changed eligibility attestation")
    if not assessment["eligible"] or assessment["manifest_sha256"] != digest(Path(path) / "manifest.json"):
        raise ValueError("ineligible or changed evidence")
    manifest = validate_run(path)
    if manifest["result_origin"] != "measured":
        raise ValueError("fixture evidence cannot be selected")
    config = read(Path(path) / "config.json")
    for field in ("method", "dataset", "split"):
        if field in selection and selection[field] != config.get(field):
            raise ValueError("study selection contradicts resolved run configuration")
    if assessment.get("method") != config.get("method"):
        raise ValueError("attestation method contradicts run")
    for dependency in (
        read(Path(path) / "dependencies.json") if (Path(path) / "dependencies.json").exists() else []
    ):
        validate_run(dependency["path"])
        if digest(Path(dependency["path"]) / "manifest.json") != dependency["manifest_sha256"]:
            raise ValueError("selected evidence dependency changed")
    return assessment
