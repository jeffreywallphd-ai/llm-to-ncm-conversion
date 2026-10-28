import statistics
import math
from dataclasses import replace
from pathlib import Path
from ncm_lab.runtime.io import read, digest, write_json
from ncm_lab.runtime.provenance import validate_run


class InfeasibleBudget(ValueError):
    pass


def training_cost_budget(config, budget):
    receipts = list(config.get("input_receipts", []))
    for key in ("training_data_reference", "development_data_reference"):
        if config.get(key):
            reference = read(config[key])
            if reference.get("input_receipt"):
                receipts.append(reference["input_receipt"])
    teacher_parents = [Path(read(receipt)["path"]).parent for receipt in receipts]
    if config.get("teacher_cost_run"):
        path = Path(config["teacher_cost_run"])
    else:
        candidates = [
            p
            for p in teacher_parents
            if (p / "manifest.json").is_file() and read(p / "manifest.json").get("run_kind") == "teacher"
        ]
        if not candidates:
            raise ValueError(
                "total cost requires canonical consumed teacher artifacts with immutable original run cost records"
            )
        path = candidates[0]
    manifest = validate_run(path)
    if (
        config.get("teacher_cost_manifest_sha256")
        and digest(path / "manifest.json") != config["teacher_cost_manifest_sha256"]
    ):
        raise ValueError("teacher cost evidence changed")
    origin = "test_fixture" if config.get("result_origin") == "test_fixture" else "measured"
    if (
        manifest["run_kind"] != "teacher"
        or manifest["status"] != "completed"
        or manifest["result_origin"] != origin
    ):
        raise ValueError("total-cost matching requires real completed teacher evidence")
    from ncm_lab.reporting.costs import cost_closure

    initial_closure = cost_closure(path)
    rows = {r["run_id"]: r for r in initial_closure["runs"]}
    preparations = dict(initial_closure["preparations"])
    for receipt_path in receipts:
        directory = Path(read(receipt_path)["path"]).parent
        if (directory / "manifest.json").exists():
            closure = cost_closure(directory)
            rows.update({r["run_id"]: r for r in closure["runs"]})
            preparations.update(closure["preparations"])
    if config.get("codec_selection"):
        codec_path = Path(config["codec_selection"]).parent
        codec_manifest = validate_run(codec_path)
        if (
            codec_manifest["run_kind"] != "codec"
            or codec_manifest["status"] != "completed"
            or codec_manifest["result_origin"] != origin
        ):
            raise ValueError("total-cost matching requires completed measured codec-search evidence")
        closure = cost_closure(codec_path)
        rows.update({r["run_id"]: r for r in closure["runs"]})
        preparations.update(closure["preparations"])
    for key in ("latent_cache", "resume_checkpoint"):
        if config.get(key):
            directory = Path(config[key]).parent
            closure = cost_closure(directory)
            rows.update({r["run_id"]: r for r in closure["runs"]})
            preparations.update(closure["preparations"])
    if not preparations or any(p.get("preprocessing_wall_seconds") is None for p in preparations.values()):
        raise ValueError(
            "total adaptation budget requires measured source preprocessing receipts; legacy unknown preparation is not zero"
        )
    preprocessing = sum(p["preprocessing_wall_seconds"] for p in preparations.values())
    preparation = [r for r in rows.values() if r["kind"] in {"teacher", "codec", "latent_cache"}]
    if any(r["origin"] != origin or r.get("executor_wall_seconds") is None for r in preparation):
        raise ValueError("complete measured preparation costs, including retries/resumes, are required")
    teacher = sum(r["executor_wall_seconds"] for r in preparation if r["kind"] == "teacher")
    codec = sum(r["executor_wall_seconds"] for r in preparation if r["kind"] in {"codec", "latent_cache"})
    previous_student = sum(r["executor_wall_seconds"] for r in rows.values() if r["kind"] == "train")
    total = config["total_new_training_wall_seconds"]
    remaining = min(budget.max_wall_seconds, total - teacher - codec - previous_student - preprocessing)
    if remaining <= 0:
        raise InfeasibleBudget(
            "preparation/resume expenditure exhausted the declared total training wall budget"
        )
    return replace(budget, max_wall_seconds=remaining), {
        "teacher_seconds": teacher,
        "codec_search_seconds": codec,
        "charged_runs": [r["run_id"] for r in preparation],
        "previous_student_seconds": previous_student,
        "source_preprocessing_seconds": preprocessing,
        "preprocessing_references": sorted(preparations),
        "actual_pre_student_expenditure": teacher + codec + previous_student + preprocessing,
        "total_wall_cap": total,
        "remaining_student_wall_cap": remaining,
        "units": "measured wall seconds on declared hardware",
        "limitation": "released pretraining, asset retrieval and benchmark scoring excluded; source preprocessing included once per receipt",
    }


def select_cost(run_paths, output, tolerance=0.1, reference="HIER_SKIP"):
    if not 0 <= tolerance <= 1:
        raise ValueError("predeclare a finite relative cost tolerance in [0,1]")
    candidates, references, manifests = [], {}, []
    for path in map(Path, run_paths):
        run = validate_run(path)
        if run["status"] != "completed":
            raise ValueError("cost selection requires completed development profiles")
        manifests.append({"run_id": run["run_id"], "manifest_sha256": digest(path / "manifest.json")})
        for row in read(path / "component_timings.json"):
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError("development timing must have positive high-resolution observations")
            key = (row["batch"], row["prefix_length"])
            if row["method"] == reference:
                references.setdefault(key, []).append(row["seconds"])
            if row["method"] == "FLAT_COST":
                candidates.append(row)
    if not candidates or not references:
        raise ValueError("profile actual reduced-depth candidates and the declared reference first")
    grouped = {}
    for row in candidates:
        layers = tuple(row["resolved_method"]["layer_indices"])
        key = (row["batch"], row["prefix_length"])
        if key not in references:
            raise ValueError("unpaired cost-profile condition")
        grouped.setdefault(layers, []).append(abs(row["seconds"] / statistics.median(references[key]) - 1))
    matched = [
        (statistics.mean(values), layers)
        for layers, values in grouped.items()
        if statistics.mean(values) <= tolerance
    ]
    if not matched:
        raise ValueError("no reduced-depth candidate meets the predeclared tolerance")
    error, layers = min(matched)
    result = {
        "schema_version": 1,
        "split": "dev",
        "layer_indices": list(layers),
        "relative_cost_error": error,
        "tolerance": tolerance,
        "reference": reference,
        "observations": manifests,
        "weighting": "uniform recorded timing observations; component cost only",
        "performance_claim": False,
    }
    write_json(output, result, exclusive=True)
    return result
