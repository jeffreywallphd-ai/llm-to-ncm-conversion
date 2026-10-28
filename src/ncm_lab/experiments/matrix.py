"""Offline, hash-bound DAG plans with finite aggregate reservations."""

import copy
import random
from pathlib import Path
from dataclasses import asdict
from ncm_lab.runtime.io import root, read, digest, canonical_hash, write_json
from ncm_lab.runtime.budget import Budget
from ncm_lab.experiments.configuration import resolve_config


def resolve_plan(spec_path, output, tiers=None):
    spec_path = Path(spec_path).resolve()
    from ncm_lab.experiments.study import materialize

    spec, sources = materialize(spec_path, tiers)
    from ncm_lab.experiments.registry import registry

    protocols = registry()
    jobs, known = [], set()
    totals = {"steps": 0, "wall_seconds": 0, "source_positions": 0, "storage_bytes": 0, "download_bytes": 0}
    dependencies = {str(path): digest(path) for path in sources}
    input_dependencies = {}
    phases = {}
    for index, job in enumerate(spec["jobs"]):
        job = copy.deepcopy(job)
        if job["id"] in known or not set(job.get("depends_on", [])) <= known:
            raise ValueError("jobs must have unique IDs and topologically ordered dependencies")
        known.add(job["id"])
        budget_path = root() / job["budget"]
        budget = Budget(**read(budget_path))
        phase = phases.setdefault(
            budget.phase_budget_id,
            {
                "caps": {k: v for k, v in asdict(budget).items() if k.startswith("max_phase_")},
                "steps": 0,
                "wall_seconds": 0,
                "source_token_passes": 0,
                "storage_bytes": 0,
                "download_bytes": 0,
            },
        )
        if phase["caps"] != {k: v for k, v in asdict(budget).items() if k.startswith("max_phase_")}:
            raise ValueError("one plan cannot change caps under the same phase ID")
        for name, key in [
            ("steps", "max_steps"),
            ("wall_seconds", "max_wall_seconds"),
            ("source_token_passes", "max_training_source_token_passes"),
            ("storage_bytes", "max_storage_bytes"),
            ("download_bytes", "max_download_bytes"),
        ]:
            phase[name] += getattr(budget, key)
            if phase[name] > phase["caps"]["max_phase_" + name]:
                raise ValueError("resolved plan exceeds its aggregate phase " + name + " cap")
        dependencies[str(budget_path)] = digest(budget_path)
        config = read(root() / job["config"]) if isinstance(job["config"], str) else job["config"]
        if isinstance(job["config"], str):
            dependencies[str(root() / job["config"])] = digest(root() / job["config"])
        config = resolve_config(config, planning=True, job_id=job["id"])

        def artifact_references(value):
            if isinstance(value, dict):
                return [ref for item in value.values() for ref in artifact_references(item)]
            if isinstance(value, list):
                return [ref for item in value for ref in artifact_references(item)]
            return [value[1:].split("/", 1)[0]] if isinstance(value, str) and value.startswith("@") else []

        ancestors = set(job.get("depends_on", []))
        pending = list(ancestors)
        by_id = {j["id"]: j for j in jobs}
        while pending:
            for parent in by_id[pending.pop()].get("depends_on", []):
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        if any(ref not in ancestors for ref in artifact_references(config)):
            raise ValueError(job["id"] + ": symbolic artifact owner outside dependency closure")
        if config.get("codec_selection", "").startswith("@"):
            owner = config["codec_selection"][1:].split("/", 1)[0]
            source = by_id[owner]["config"]
            method = config.get("scientific_resolution", {})
            if method.get("codec_kind") and source.get("codec_kind", "nested") != method["codec_kind"]:
                raise ValueError(job["id"] + ": selected codec kind contradicts treatment")

        def bind_input(value):
            if not value or value.startswith("@"):
                return
            path = (root() / value).resolve()
            input_dependencies[str(path)] = digest(path) if path.is_file() else None

        for field in (
            "training_data",
            "development_data",
            "data_path",
            "prompts_path",
            "evaluation_data",
            "checkpoint",
            "checkpoint_selection",
            "codec_selection",
            "resume_checkpoint",
            "overlap_audit",
            "latent_cache",
            "training_data_reference",
            "development_data_reference",
            "training_prompt_receipt",
        ):
            bind_input(config.get(field))
        for field in ("input_receipts", "benchmark_receipts", "required_files"):
            for value in config.get(field, []):
                bind_input(value)
        blocked = []
        declared = protocols[config["experiment_id"]]["methods"]
        methods = [config["method"]] if config.get("method") else config.get("methods", [])
        if any(method not in declared for method in methods):
            raise ValueError("job method is not registered in its experiment protocol")
        if config.get("method") and config.get("executor") not in {
            "acceptance",
            "codec",
            "codec_evaluate",
            "latent_cache",
        }:
            try:
                job["resolved_method"] = config.get("scientific_resolution")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                blocked.append(str(exc))
        protocol = root() / protocols[config["experiment_id"]]["directory"] / "protocol.yaml"
        dependencies[str(protocol)] = digest(protocol)
        from ncm_lab.experiments.runner import prerequisites

        blocked.extend(prerequisites(config))
        totals["steps"] += budget.max_steps
        totals["wall_seconds"] += budget.max_wall_seconds
        totals["source_positions"] += budget.max_training_source_token_passes
        totals["storage_bytes"] += budget.max_storage_bytes
        totals["download_bytes"] += budget.max_download_bytes
        job.update(config=config, budget=asdict(budget), blocked=blocked, order=index)
        jobs.append(job)
    caps = spec["aggregate_limits"]
    if set(caps) != set(totals) or any(totals[k] > caps[k] for k in totals):
        raise ValueError("resolved jobs exceed finite aggregate plan caps")
    order, completed = [], set()
    rng = random.Random(spec.get("timing_seed", 0))
    while len(completed) < len(jobs):
        ready = [
            j["id"] for j in jobs if j["id"] not in completed and set(j.get("depends_on", [])) <= completed
        ]
        if not ready:
            raise ValueError("cyclic job dependencies")
        rng.shuffle(ready)
        order.extend(ready)
        completed.update(ready)
    plan = {
        "schema_version": 3,
        "jobs": jobs,
        "order": order,
        "dependencies": dependencies,
        "contrast_instances": spec.get("contrast_instances", []),
        "robustness": spec.get("robustness"),
        "coverage": spec.get("coverage"),
        "selected_tiers": spec.get("selected_tiers"),
        "input_dependencies": input_dependencies,
        "requires_calibration": spec.get("requires_calibration", False),
        "calibration_run": spec.get("calibration_run"),
        "aggregate_limits": caps,
        "reserved_upper_bounds": totals,
        "timing_seed": spec.get("timing_seed", 0),
        "unknown_costs": ["released-checkpoint step rate, peak memory, codec/drafter convergence"],
        "offline_only": True,
    }
    if spec.get("kind") == "first_paper_v3":
        from ncm_lab.experiments.study import validate_coverage

        plan["coverage"] = validate_coverage(jobs, spec.get("contrast_instances", []))
    from ncm_lab.runtime.provenance import implementation_hash

    plan["implementation_sha256"] = implementation_hash()
    plan["sha256"] = canonical_hash(plan)
    write_json(output, plan, exclusive=True)
    return plan


def launch_plan(path):
    plan = read(path)
    check = dict(plan)
    expected = check.pop("sha256")
    if canonical_hash(check) != expected:
        raise ValueError("resolved plan changed")
    if any(digest(p) != sha for p, sha in plan["dependencies"].items()):
        raise ValueError("plan dependency changed; resolve a new plan")
    if any(
        (digest(p) if Path(p).is_file() else None) != sha
        for p, sha in plan.get("input_dependencies", {}).items()
    ):
        raise ValueError("plan input dependency changed or became available; resolve a new plan")
    from ncm_lab.runtime.provenance import implementation_hash

    if plan["implementation_sha256"] != implementation_hash():
        raise ValueError("plan implementation changed; resolve a new plan")
    from ncm_lab.experiments.runner import run
    import uuid

    launch = root() / "results/generated/plans" / (expected[:12] + "-" + uuid.uuid4().hex[:8])
    launch.mkdir(parents=True)
    write_json(launch / "resolved-plan.json", plan, exclusive=True)
    results = {}
    if plan.get("requires_calibration"):
        from ncm_lab.runtime.provenance import validate_run

        calibration = plan.get("calibration_run")
        valid = False
        if calibration:
            manifest = validate_run(calibration, require_current_code=True)
            measured = read(Path(calibration) / "metrics.json")
            valid = (
                manifest["status"] == "completed"
                and manifest["result_origin"] == "measured"
                and measured.get("optimizer_steps", 0) > 0
                and measured.get("storage_bytes", 0) > 0
            )
        if not valid:
            results = {
                j["id"]: {"status": "blocked", "reason": "real bounded training/storage calibration required"}
                for j in plan["jobs"]
            }
            write_json(launch / "results.json", results, exclusive=True)
            return {
                "path": str(launch),
                "status": "blocked",
                "jobs": len(results),
                "reason": "calibration prerequisite missing; no research job launched",
            }

    def references(value):
        if isinstance(value, dict):
            return {k: references(v) for k, v in value.items()}
        if isinstance(value, list):
            return [references(v) for v in value]
        if isinstance(value, str) and value.startswith("@"):
            if "/" not in value[1:]:
                directory = Path(results[value[1:]]["path"])
                if not (directory / "manifest.json").is_file():
                    raise ValueError("upstream run manifest missing")
                return str(directory)
            job_id, artifact = value[1:].split("/", 1)
            hashed = artifact.endswith("#sha256")
            artifact = artifact.removesuffix("#sha256")
            field = None
            if "#json:" in artifact:
                artifact, field = artifact.split("#json:", 1)
            base = Path(results[job_id]["path"]).resolve()
            path = (base / artifact).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise ValueError("missing/unsafe upstream job artifact")
            return read(path)[field] if field else digest(path) if hashed else str(path)
        return value

    for job_id in plan["order"]:
        job = next(j for j in plan["jobs"] if j["id"] == job_id)

        def usable(result):
            if result["status"] == "completed":
                return True
            if result["status"] != "cancelled" or not result.get("path"):
                return False
            metrics = read(Path(result["path"]) / "metrics.json")
            return metrics.get("stop_reason") == "budget" and metrics.get("endpoint_eligible") is True

        if any(not usable(results[d]) for d in job.get("depends_on", [])):
            results[job_id] = {"status": "blocked", "reason": "dependency did not complete"}
            continue
        config_path, budget_path = launch / (job_id + ".json"), launch / (job_id + ".budget.json")
        try:
            resolved_config = references(job["config"])
        except (ValueError, KeyError, OSError) as exc:
            results[job_id] = {"status": "blocked", "reason": str(exc)}
            write_json(launch / (job_id + ".result.json"), results[job_id], exclusive=True)
            continue
        write_json(
            config_path,
            {
                **resolved_config,
                "resolved_plan": expected,
                "job_id": job_id,
                "resolved_plan_artifact": str(launch / "resolved-plan.json"),
                "job_order": plan["order"].index(job_id),
            },
            exclusive=True,
        )
        write_json(budget_path, job["budget"], exclusive=True)
        try:
            results[job_id] = run(job["config"]["experiment_id"], config_path, budget_path)
        except (ValueError, RuntimeError, OSError) as exc:
            # Preserve pre-launch/phase-reservation failures and continue independent jobs.
            results[job_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json(launch / (job_id + ".result.json"), results[job_id], exclusive=True)
    write_json(launch / "results.json", results, exclusive=True)
    bindings = []
    for instance in plan.get("contrast_instances", []):
        roles = {}
        excluded = []
        for role, identifiers in instance["roles"].items():
            roles[role] = []
            for identifier in identifiers:
                result = results[identifier]
                if result["status"] == "completed":
                    directory = Path(result["path"])
                    roles[role].append(
                        {
                            "run_id": result["run_id"],
                            "path": str(directory),
                            "manifest_sha256": digest(directory / "manifest.json"),
                        }
                    )
                else:
                    excluded.append(
                        {
                            "job_id": identifier,
                            "status": result["status"],
                            "reason": "not an eligible completed final observation",
                        }
                    )
        bindings.append({**instance, "roles": roles, "excluded": excluded})
    write_json(
        launch / "analysis-bindings.json",
        {
            "schema_version": 3,
            "plan_sha256": expected,
            "contrasts": bindings,
            "plan": {
                "path": str(launch / "resolved-plan.json"),
                "sha256": digest(launch / "resolved-plan.json"),
            },
            "results": {"path": str(launch / "results.json"), "sha256": digest(launch / "results.json")},
            "selection_rule": "one recorded attempt per job; failures retained; retries require a new explicit predeclared binding",
        },
        exclusive=True,
    )
    return {"path": str(launch), "results": results}
