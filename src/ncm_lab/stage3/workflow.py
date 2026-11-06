"""Stage 3 contracts over the shared foreground state machine."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from ncm_lab.runtime.io import read, resolve_reference
from ncm_lab.runtime.budget import Budget
from ncm_lab.stage1.workflow import checked

DEPENDENCIES = {
    "source": (),
    "distill": ("source",),
    "prepare": ("source", "distill"),
    "teacher": ("prepare",),
    "codec": ("teacher",),
    "targets": ("prepare", "teacher", "codec"),
    "generator": ("targets",),
    "evaluate": ("prepare", "generator"),
}


def configuration(name, path, state, settings):
    value = deepcopy(settings["stages"][name])
    resources = settings["resources"]
    folder = path.parent
    results = {
        n: read(checked(e["receipt"]))["result"] for n, e in state["stages"].items() if e.get("receipt")
    }
    if name in {"source", "distill", "prepare"}:
        source = settings["stages"]["source"]["data"]
        count = (
            source["max_source_rows"] + source.get("overlap", {}).get("max_source_rows", 0)
            if name == "source"
            else value["stage3"]["distillation"]["max_candidates"]
        )
        canvas = settings["canvas"]["positions"]
        tokens = count * (
            source.get("max_row_chars", 131072) if name == "source" else canvas * (512 + canvas)
        )
        kwargs = {
            "max_steps": count + 1,
            "max_source_tokens": tokens,
            "max_answer_positions": tokens,
            "max_canvas_positions": tokens,
            "max_storage_bytes": resources["max_retained_bytes"],
        }
        kwargs.update(
            {
                k.replace("max_", "max_phase_", 1): v * resources["retry_allowance"]
                for k, v in list(kwargs.items())
            }
        )
        kwargs["max_phase_storage_bytes"] = resources["max_phase_retained_bytes"]
        value["budget"] = asdict(
            Budget(
                "s3-" + settings["contract_sha256"][:40] + "-" + name,
                enforce_wall_time_limits=False,
                **kwargs,
            )
        )
    else:
        if not state.get("resources"):
            raise ValueError("final qualified data/resource receipt required")
        value = deepcopy(read(checked(state["resources"]))["stages"][name]["config"])

    def artifact(n):
        return results.get(n, {}).get("artifact")

    inputs = {
        "source": {},
        "distill": {"source": artifact("source")},
        "prepare": {"source": artifact("source"), "distill": artifact("distill")},
        "teacher": {"data": artifact("prepare")},
        "codec": {"features": artifact("teacher")},
        "targets": {"data": artifact("prepare"), "features": artifact("teacher"), "codec": artifact("codec")},
        "generator": {"targets": artifact("targets")},
        "evaluate": {
            "generator": artifact("generator"),
            "data": artifact("prepare"),
            "evaluation": results.get("prepare", {}).get("evaluation_data", {}).get("manifest"),
        },
    }[name]
    if any(p is None for p in inputs.values()):
        raise ValueError("missing exact Stage 3 prerequisites")
    value.update(
        inputs=inputs,
        run_root=str(folder / "native_runs"),
        output_ref=str(folder / "handoffs" / (name + ".json")),
    )
    value["stage1_context"] = dict(
        schema="stage3_experiment_v1",
        experiment=str(path),
        contract_sha256=settings["contract_sha256"],
        stage=name,
        experiment_kind="stage3",
    )
    return value


def verify_artifact(name, result, value):
    artifact = Path(result["artifact"]).resolve()
    if resolve_reference(value["output_ref"]).resolve() != artifact:
        raise ValueError("Stage 3 handoff differs from sealed artifact")
    from ncm_lab.data.artifacts import Dataset, LOCKED

    if name in {"source", "distill", "prepare"}:
        data = Dataset(
            artifact,
            {"source": "stage3_candidates_v1", "distill": "stage3_distilled_v1", "prepare": "native_data_v1"}[
                name
            ],
        )
        if data.manifest["identity"]["canvas"] != value["architecture"]["canvas"]:
            raise ValueError("Stage 3 data geometry differs")
        for i in range(len(data)):
            data[i]  # Verify per-example identities, not just the index file.
        if name in {"source", "prepare"}:
            locked = Dataset(checked(result["evaluation_data"]), LOCKED)
            if set(locked.manifest["groups"]["test"]) & (
                set(data.manifest["groups"]["train"]) | set(data.manifest["groups"]["dev"])
            ):
                raise ValueError("locked group leaked across splits")
        if name == "prepare" and read(checked(result["qualification"]))["status"] != "qualified":
            raise ValueError("final data qualification failed")
    else:
        report = read(artifact)
        if (
            report.get("schema") != "stage3_selected_evaluation_v1"
            or not report["selection_frozen_before_test"]
        ):
            raise ValueError("invalid Stage 3 evaluation")
        checked(report["raw_outputs"])
        checked(report["ar_reference"])
        checked(report["ar_raw_outputs"])
