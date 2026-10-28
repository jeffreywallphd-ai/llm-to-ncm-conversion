"""Versioned codec measurements, matched error units and explicit cost coordinates."""

import csv
import json
from pathlib import Path
from ncm_lab.runtime.io import read, write_json, digest
from ncm_lab.runtime.provenance import validate_run, implementation_hash
from ncm_lab.reporting.costs import cost_closure


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def analyze_codec_runs(paths, output, figures=False):
    paths = list(map(Path, paths))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rates = []
    curves = []
    interventions = []
    gaps = []
    costs = []
    selected_costs = []
    inputs = []
    for base in paths:
        manifest = validate_run(base)
        config = read(base / "config.json")
        metrics = read(base / "metrics.json")
        common = {
            "run_id": manifest["run_id"],
            "manifest_sha256": digest(base / "manifest.json"),
            "origin": manifest["result_origin"],
            "method": config.get("method", config.get("executor")),
            "treatment_id": config.get("treatment_id", config.get("method")),
            "training_seed": config.get("seed"),
            "codec_seed": config.get("codec_seed"),
            "data_tier": config.get("data_tier"),
            "comparison_regime": config.get("comparison", "equal_data"),
            "split": config.get("split"),
            "objective": config.get("codec_objective"),
            "codec_sha256": config.get("codec_sha256"),
            "student_sha256": config.get("checkpoint_sha256"),
        }
        inputs.append({"run_id": manifest["run_id"], "manifest_sha256": common["manifest_sha256"]})
        closure = cost_closure(base)
        adaptation = [
            row
            for row in closure["runs"]
            if row["kind"] in {"teacher", "codec", "latent_cache", "train", "acceptance"}
        ]
        parents = [row for row in adaptation if row["run_id"] != manifest["run_id"]]

        def total(rows, key):
            return sum(row[key] for row in rows) if all(row.get(key) is not None for row in rows) else None

        preprocessing = closure["total_once_per_dependency"]["source_preprocessing_seconds"]

        def adaptation_wall(rows):
            observed = total(rows, "executor_wall_seconds")
            return observed + preprocessing if observed is not None and preprocessing is not None else None

        parent_wall = adaptation_wall(parents)
        costs.append(
            {
                **common,
                **closure["total_once_per_dependency"],
                "adaptation_wall_seconds": adaptation_wall(adaptation),
                "missing": closure["missing"],
                "dependency_count": len(closure["runs"]),
                "cost_provenance": closure["runs"],
            }
        )
        if metrics.get("diagnostics_schema_version") == 3:
            dependency = metrics["codec_dependencies"]
            for level, observation in metrics["diagnostics"].items():
                rates.append(
                    {
                        **common,
                        "purpose": metrics["purpose"],
                        "split": metrics["scope"],
                        "level": int(level),
                        "objective": dependency["objective"],
                        "selected_prediction_weight": dependency["selected_prediction_weight"],
                        "codec_sha256": dependency["weights_sha256"],
                        "codec_architecture_version": dependency["spec"]["architecture_version"],
                        "standalone_scalars": metrics["rates"]["standalone_scalars"][level],
                        "standalone_bytes": metrics["rates"]["standalone_bytes"][level],
                        "all_level_scalars": metrics["rates"]["all_levels_scalars"],
                        "rate_definition": "standalone independent code",
                        "normalization": metrics["normalization"],
                        "positions_sha256": metrics["evaluated_positions_sha256"],
                        **observation,
                    }
                )
        for file in base.rglob("codec-selection.json"):
            selection = read(file)
            for observation in selection["candidates"]:
                for level, values in observation["metrics"].items():
                    rates.append(
                        {
                            **common,
                            "subprocedure": file.parent.relative_to(base).as_posix(),
                            "purpose": "development_candidate_selection",
                            "split": "dev",
                            "level": int(level),
                            "standalone_scalars": selection["rates"]["standalone_scalars"][level],
                            "standalone_bytes": selection["rates"]["standalone_bytes"][level],
                            "all_level_scalars": selection["rates"]["all_levels_scalars"],
                            "rate_definition": "standalone independent code",
                            "objective": selection.get("objective", "legacy_search_unspecified"),
                            "candidate": observation["candidate"],
                            "step": observation["step"],
                            "prediction_weight": observation["prediction_weight"],
                            "normalization": selection["normalization"],
                            **values,
                        }
                    )
        for file in base.rglob("dev-*.json"):
            observation = read(file)
            if "dev_loss" not in observation:
                continue
            elapsed = observation.get("cumulative_student_wall_seconds")
            curves.append(
                {
                    **common,
                    "subprocedure": file.parent.relative_to(base).as_posix(),
                    "step": observation["step"],
                    "development_loss": observation["dev_loss"],
                    "valid_positions": observation["valid_positions"],
                    "actual_target_positions": observation.get("source_token_passes"),
                    "distinct_source_positions": observation.get("distinct_source_positions"),
                    "distinct_eligible_response_positions": observation.get(
                        "distinct_eligible_response_positions"
                    ),
                    "cumulative_student_wall_seconds": elapsed,
                    "completed_preparation_wall_seconds": parent_wall,
                    "cumulative_adaptation_wall_seconds": parent_wall + elapsed
                    if parent_wall is not None and elapsed is not None
                    else None,
                    "selected_total_procedure_cost": None,
                    "cost_missing_reason": None
                    if elapsed is not None and parent_wall is not None
                    else "no contemporaneous cumulative cost observation; final-run time is not substituted",
                    "cost_dependency_run_ids": [row["run_id"] for row in parents],
                }
            )
        if (base / "selection.json").exists():
            selection = read(base / "selection.json")
            selected_costs.append(
                {
                    **common,
                    "selected_checkpoint_sha256": selection["weights_sha256"],
                    "selected_step": selection["selected"]["step"],
                    "full_procedure_adaptation_wall_seconds": adaptation_wall(adaptation),
                    "includes_later_discarded_training_and_selection": True,
                    "cost_provenance": closure["runs"],
                    "missing": closure["missing"],
                }
            )
        for file in base.rglob("latent-diagnostics.json"):
            diagnostic = read(file)
            for kind, field in [
                ("bridge_removal", "interventions"),
                ("zero_predicted_code", "level_zeroing"),
            ]:
                for level, observation in diagnostic.get(field, {}).items():
                    interventions.append(
                        {
                            **common,
                            "level": int(level),
                            "disabled_level": int(level),
                            "intervention": kind,
                            "gate": diagnostic["gates"][level],
                            **observation,
                        }
                    )
            for level, predicted in diagnostic["causal_predictions"].items():
                oracle = diagnostic["oracle_codec"][level]
                gaps.append(
                    {
                        **common,
                        "level": int(level),
                        "error_space": "reconstructed target prediction state",
                        "oracle_normalized_mse": oracle["normalized_mse"],
                        "predicted_normalized_mse": predicted["normalized_mse"],
                        "oracle_raw_mse": oracle.get("raw_mse"),
                        "predicted_raw_mse": predicted["mse"],
                        "normalization": diagnostic.get("normalization"),
                        "positions": diagnostic.get("evaluated_positions"),
                        "valid_positions": oracle["valid_positions"],
                        "scope": "oracle diagnostic versus causal predictor; no oracle at inference",
                    }
                )
    tables = {
        "rate_distortion": rates,
        "learning_curves": curves,
        "interventions": interventions,
        "oracle_predicted": gaps,
        "total_dependency_cost": costs,
        "selected_procedure_cost": selected_costs,
    }
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    write_json(output / "tables.json", tables, exclusive=True)
    if figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name, x, y in [
            ("rate_distortion", "standalone_scalars", "normalized_mse"),
            ("learning_curves", "actual_target_positions", "development_loss"),
            ("interventions", "disabled_level", "max_logit_difference"),
            ("oracle_predicted", "oracle_normalized_mse", "predicted_normalized_mse"),
            ("total_dependency_cost", "target_processed_positions", "executor_wall_seconds"),
        ]:
            values = [row for row in tables[name] if row.get(x) is not None and row.get(y) is not None]
            if not values:
                continue
            fig, axis = plt.subplots()
            labels = sorted(
                {
                    str(
                        (
                            row["origin"],
                            row["method"],
                            row["data_tier"],
                            row["comparison_regime"],
                            row.get("objective"),
                            row.get("purpose"),
                        )
                    )
                    for row in values
                }
            )
            for label in labels:
                subset = [
                    row
                    for row in values
                    if str(
                        (
                            row["origin"],
                            row["method"],
                            row["data_tier"],
                            row["comparison_regime"],
                            row.get("objective"),
                            row.get("purpose"),
                        )
                    )
                    == label
                ]
                axis.scatter([row[x] for row in subset], [row[y] for row in subset], label=label)
            axis.set(
                xlabel=x.replace("_", " "),
                ylabel=y.replace("_", " "),
                title="Recorded measurements; fixtures are engineering only",
            )
            axis.legend(fontsize=6)
            fig.savefig(output / (name + ".png"), dpi=160, bbox_inches="tight")
            plt.close(fig)
    receipt = {
        "schema_version": 3,
        "analysis": "codec_measurements_v3",
        "implementation_sha256": implementation_hash(),
        "inputs": inputs,
        "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "parameters": {
            "figures": figures,
            "reconstruction_units": "normalized target-state MSE for both oracle and prediction",
        },
        "research_claims": [],
    }
    write_json(output / "analysis_manifest.json", receipt, exclusive=True)
    return {
        "output": str(output),
        "rows": {key: len(value) for key, value in tables.items()},
        "analysis_manifest": str(output / "analysis_manifest.json"),
        "research_claims": [],
    }
