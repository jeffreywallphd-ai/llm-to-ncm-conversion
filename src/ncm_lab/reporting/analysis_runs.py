"""Explicit immutable analysis inclusions, independent of the historical run index."""

import json
import uuid
from pathlib import Path
from ncm_lab.runtime.io import root, read, write_json, digest, canonical_hash
from ncm_lab.runtime.provenance import validate_run, implementation_hash
from ncm_lab.evaluation.tables import analyze_records


def load_observations(path, allow_legacy=False):
    path = Path(path)
    manifest = validate_run(path)
    expected = read(path / "analysis_identity.json") if (path / "analysis_identity.json").exists() else None
    rows = []
    if (path / "predictions.jsonl").exists():
        for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if expected:
                from ncm_lab.evaluation.identities import validate_record_identity

                validate_record_identity(row, expected)
            elif not allow_legacy and manifest["result_origin"] != "test_fixture":
                raise ValueError(
                    "legacy predictions lack treatment/condition identities; request explicit diagnostic-only reanalysis"
                )
            from ncm_lab.evaluation.endpoints import ENDPOINTS, endpoint_result

            if row.get("dataset") in ENDPOINTS and "endpoint" not in row:
                row["endpoint"] = endpoint_result(
                    row["dataset"], row.get("official_scores", {}), row.get("error")
                )
                row["correct"] = row["endpoint"]["correct"]
            rows.append(
                {
                    **row,
                    "run_id": manifest["run_id"],
                    "manifest_sha256": digest(path / "manifest.json"),
                    "result_origin": manifest["result_origin"],
                }
            )
    return rows


def analyze(experiment_id, run_ids=None, bindings=None, legacy_diagnostic=False):
    from ncm_lab.reporting.reports import find_run
    from ncm_lab.experiments.registry import registry

    if experiment_id not in registry():
        raise ValueError("unregistered experiment")
    if bool(run_ids) == bool(bindings):
        raise ValueError(
            "analysis requires either explicit --runs or locked --bindings; use results index for historical discovery"
        )
    contrasts = None
    binding_ref = None
    excluded = []
    if bindings:
        binding = read(bindings)
        for key in ("plan", "results"):
            reference = binding[key]
            if digest(reference["path"]) != reference["sha256"]:
                raise ValueError("locked analysis " + key + " reference changed")
        plan = read(binding["plan"]["path"])
        results = read(binding["results"]["path"])
        if binding["plan_sha256"] != plan["sha256"]:
            raise ValueError("locked analysis plan identity mismatch")
        original = {c["id"]: c for c in plan["contrast_instances"]}
        if sorted(c["id"] for c in binding["contrasts"]) != sorted(original):
            raise ValueError("locked analysis must retain every declared contrast exactly once")
        contrasts = []
        for instance in binding["contrasts"]:
            declared = original[instance["id"]]
            for key, value in declared.items():
                if key != "roles" and instance.get(key) != value:
                    raise ValueError("locked analysis contrast declaration changed: " + key)
            if set(instance["roles"]) != set(declared["roles"]):
                raise ValueError("locked analysis role declaration changed")
            roles = {}
            for role, references in instance["roles"].items():
                allowed = {
                    results[j]["run_id"]
                    for j in declared["roles"][role]
                    if results[j]["status"] == "completed"
                }
                if {r["run_id"] for r in references} != allowed:
                    raise ValueError("analysis retry/inclusion differs from locked plan outcomes")
                for reference in references:
                    directory = Path(reference["path"])
                    source = next(
                        results[j]
                        for j in declared["roles"][role]
                        if results[j].get("run_id") == reference["run_id"]
                    )
                    if (
                        directory.resolve() != Path(source["path"]).resolve()
                        or digest(directory / "manifest.json") != reference["manifest_sha256"]
                    ):
                        raise ValueError("locked analysis selected run identity changed")
                roles[role] = [r["run_id"] for r in references]
            excluded.extend(instance["excluded"])
            contrasts.append({**instance, "roles": roles})
        paths = [Path(r["path"]) for r in results.values() if r.get("path")]
        binding_ref = {
            "path": str(Path(bindings).resolve()),
            "sha256": digest(bindings),
            "plan_sha256": plan["sha256"],
        }
    else:
        paths = [find_run(value) for value in run_ids]
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("analysis inclusion is empty")
    observations = []
    records = []
    for path in paths:
        manifest = validate_run(path)
        records.append(
            {
                "run_id": manifest["run_id"],
                "manifest_sha256": digest(path / "manifest.json"),
                "status": manifest["status"],
                "origin": manifest["result_origin"],
                "path": str(path),
            }
        )
        observations.extend(load_observations(path, legacy_diagnostic))
    analysis = analyze_records(observations, contrasts=contrasts)
    output = root() / "results/generated/analyses" / (experiment_id + "-" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    from ncm_lab.evaluation.codec_analysis import analyze_codec_runs

    codec = analyze_codec_runs(paths, output / "codec-tables")
    write_json(
        output / "analysis.json",
        {
            "experiment_id": experiment_id,
            "runs": records,
            "analysis": analysis,
            "codec_analysis": codec,
            "excluded": excluded,
            "findings": [],
        },
        exclusive=True,
    )
    receipt = {
        "schema_version": 3,
        "analysis": "explicit_contrasts_v3",
        "implementation_sha256": implementation_hash(),
        "inputs": records,
        "binding": binding_ref,
        "contrasts_sha256": canonical_hash(contrasts),
        "legacy_diagnostic": legacy_diagnostic,
        "inclusion_policy": "locked bindings" if bindings else "explicit exploratory runs",
        "parameters": {
            "bootstrap_iterations": 500,
            "duplicate_policy": "reject",
            "failure_policy": "retain all attempts",
        },
        "files": {"analysis.json": digest(output / "analysis.json")},
        "derived_tables_manifest": {
            "path": codec["analysis_manifest"],
            "sha256": digest(codec["analysis_manifest"]),
        },
    }
    write_json(output / "analysis_manifest.json", receipt, exclusive=True)
    return {
        "path": str(output / "analysis.json"),
        "analysis_manifest": str(output / "analysis_manifest.json"),
        "runs": len(records),
        "research_findings": [],
        "binding": binding_ref,
    }


def robustness_stage(run, config, meter):
    boundaries = config["context_length_strata"]
    if boundaries[0] != 0 or boundaries[-1] is not None or sorted(boundaries[:-1]) != boundaries[:-1]:
        raise ValueError("lock increasing context-length strata before final outcomes")
    strata = [[] for _ in boundaries[:-1]]
    sources = []
    for path in map(Path, config["shared_evaluation_runs"]):
        manifest = validate_run(path)
        if manifest["experiment_id"] != "ACL-002" or read(path / "config.json")["dataset"] != "longbench":
            raise ValueError("robustness reuses actual ACL-002 LongBench evidence only")
        sources.append({"run_id": manifest["run_id"], "manifest_sha256": digest(path / "manifest.json")})
        for row in load_observations(path):
            meter.check()
            length = row["input_tokens"]
            index = next(
                i
                for i in range(len(strata))
                if length >= boundaries[i] and (boundaries[i + 1] is None or length < boundaries[i + 1])
            )
            strata[index].append(row)
            meter.record(0, 0)
    result = {
        "schema_version": 3,
        "factor": "input token length",
        "boundaries": boundaries,
        "shared_evidence": True,
        "new_independent_generalization_runs": False,
        "sources": sources,
        "strata": [],
    }
    for index, rows in enumerate(strata):
        result["strata"].append(
            {
                "lower": boundaries[index],
                "upper": boundaries[index + 1],
                "requests": len(rows),
                "missing_or_failed": sum(bool(r.get("error")) for r in rows),
                "analysis": analyze_records(rows),
            }
        )
    write_json(run.path / "strata.json", result, exclusive=True)
    return {
        "status": "completed",
        "shared_source_runs": len(sources),
        "strata": result["strata"],
        "independent_runs_added": 0,
    }
