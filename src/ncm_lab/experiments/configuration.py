"""Model-free executor contracts, shared by planning and execution."""

import copy
import math
import re
from dataclasses import asdict
from ncm_lab.models.factory import resolve_method

EXECUTORS = {
    "fixture",
    "acceptance",
    "profile",
    "train",
    "teacher",
    "verification",
    "evaluate",
    "development",
    "codec",
    "codec_evaluate",
    "latent_cache",
    "latent_diagnostics",
    "overlap_audit",
    "cost_selection",
    "workflow",
    "workload",
    "external_engine",
    "robustness",
}
MODEL_EXECUTORS = {
    "train",
    "teacher",
    "verification",
    "evaluate",
    "development",
    "codec",
    "codec_evaluate",
    "latent_cache",
    "latent_diagnostics",
    "profile",
}


def symbolic(value):
    return isinstance(value, str) and value.startswith("@")


def resolve_config(config, planning=False, job_id="direct"):
    c = copy.deepcopy(config)

    def fail(field, reason):
        raise ValueError(f"{job_id}: {field}: {reason}")

    def require(field, kind):
        if field not in c or not isinstance(c[field], kind) or isinstance(c[field], bool) and kind is int:
            fail(field, "required " + kind.__name__)

    for field in ("experiment_id", "executor", "result_origin"):
        require(field, str)
    if c["executor"] not in EXECUTORS:
        fail("executor", "unsupported executor")
    fixture = c["result_origin"] == "test_fixture"
    if c.get("data_ladder_version", 3) != 3:
        fail("data_ladder_version", "new execution requires version 3; legacy versions are migration-only")
    if c.get("fixed_codec_transfer") and c.get("comparison") != "fixed_codec_transfer":
        fail(
            "fixed_codec_transfer",
            "larger-codec reuse requires separately named comparison=fixed_codec_transfer",
        )
    if "anchor_policy" in c and c["anchor_policy"] != "first_eligible_for_dev_seeded_train":
        fail("anchor_policy", "supported first_eligible_for_dev_seeded_train")
    c.setdefault("cpu_threads", 1)
    if c["executor"] in MODEL_EXECUTORS:
        for field in ("device", "precision"):
            require(field, str)
        if not fixture:
            require("model", dict)
            for field in ("target_id", "target_revision", "tokenizer_revision"):
                if not isinstance(c["model"].get(field), str):
                    fail("model." + field, "required pinned identity")
    for field in (
        "max_requests",
        "warmup_requests",
        "timing_repetition",
        "steps",
        "max_examples",
        "cpu_threads",
        "max_development_examples",
        "repetitions",
        "evaluation_interval",
        "checkpoint_interval",
        "data_tier",
    ):
        if field in c and (isinstance(c[field], bool) or not isinstance(c[field], int) or c[field] < 0):
            fail(field, "finite nonnegative integer required")
    for field in (
        "cpu_threads",
        "max_development_examples",
        "repetitions",
        "evaluation_interval",
        "checkpoint_interval",
        "data_tier",
    ):
        if field in c and c[field] == 0:
            fail(field, "positive integer required")
    if "total_new_training_wall_seconds" in c:
        value = c["total_new_training_wall_seconds"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            fail("total_new_training_wall_seconds", "positive finite seconds required")
    if c.get("budget_endpoint_policy") not in {None, "best_completed_dev_before_budget"}:
        fail("budget_endpoint_policy", "unsupported selection policy")
    if c["executor"] in {"evaluate", "verification"}:
        for field in ("method", "dataset", "split", "data_path"):
            require(field, str)
        for field in ("max_requests", "max_new_tokens"):
            require(field, int)
        if c["max_requests"] <= 0 or c["max_new_tokens"] < 0:
            fail("max_requests/max_new_tokens", "invalid finite request cap")
        if c["executor"] == "evaluate":
            require("output_instruction", str)
            c.setdefault("warmup_requests", 0)
            c.setdefault("timing_repetition", 0)
            if c["dataset"] in {"ifeval", "longbench"}:
                require("official_evaluator", dict)
                for field in ("root", "revision", "files_sha256"):
                    if field not in c["official_evaluator"]:
                        fail("official_evaluator." + field, "required scorer identity")
            if c["method"] not in {"AR", "FLAT_RELEASED", "STRONG_EXTERNAL"}:
                for field in ("checkpoint", "checkpoint_sha256", "checkpoint_selection"):
                    require(field, str)
        c["prompt_policy"] = {
            "schema_version": 1,
            "style": "official" if c["dataset"] in {"ifeval", "longbench"} else "allowlisted_json",
            "output_instruction": c.get("output_instruction", ""),
            "thinking": False,
        }
    if c["executor"] in {"train", "codec"}:
        if not c.get("training_data_reference"):
            require("training_data", str)
        if not c.get("development_data_reference"):
            require("development_data", str)
    if c["executor"] in {"codec", "codec_evaluate"}:
        if c.get("codec_kind", "nested") not in {"nested", "linear", "independent", "channel", "pooling"}:
            fail("codec_kind", "unsupported codec")
        levels = c.get("levels", [4, 8, 15])
        width = c.get("latent_width", 256)
        if not levels or len(levels) != len(set(levels)) or not set(levels) <= {4, 8, 15}:
            fail("levels", "unique declared codec levels required")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            fail("latent_width", "positive integer required")
        weights = c.get("prediction_weights", [0.0, 0.1])
        if not weights or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
            for w in weights
        ):
            fail("prediction_weights", "finite nonnegative coefficients required")
        inferred = (
            "reconstruction_only"
            if all(w == 0 for w in weights)
            else "prediction_aware"
            if all(w > 0 for w in weights)
            else "adaptive_selection"
        )
        if c.get("codec_objective", inferred) != inferred:
            fail("codec_objective", "contradicts coefficient candidate set")
        c["codec_objective"] = inferred
    if c["executor"] == "profile":
        from pathlib import Path
        from ncm_lab.runtime.io import read, root
        from ncm_lab.models.factory import LATENT

        overrides = c.get("method_overrides", {})
        for name in c.get("methods", ["FLAT_ADAPTED", "HIER_SKIP", "HIER_NO_SKIP"]):
            effective = {**c, "method": name, **overrides.get(name, {})}
            for field in ("device", "precision", "model"):
                if effective.get(field) != c.get(field):
                    fail(
                        "method_overrides." + name + "." + field,
                        "profiler requires common placement/upstream identity",
                    )
            if name == "FLAT_COST" and c.get("cost_candidate"):
                effective["cost_selection"] = "unselected-development-profile-candidate"
            if (
                fixture
                and name in LATENT
                and name != "HIER_LATENT_CE_ONLY"
                and not effective.get("codec_selection")
            ):
                effective["codec_sha256"] = "software-fixture-untrained-codec"
                effective.setdefault("latent_width", 4)
            resolved = resolve_method(effective, "plan")
            selected = effective.get("codec_selection")
            if selected and not symbolic(selected) and (root() / Path(selected)).is_file():
                spec = read(root() / selected).get("codec_spec")
                if spec:
                    if resolved.codec_kind != spec["kind"]:
                        fail(
                            "method_overrides." + name + ".codec_selection",
                            "selected codec kind contradicts treatment",
                        )
                    for field, key in [
                        ("latent_width", "width"),
                        ("levels", "levels"),
                        ("codec_kind", "kind"),
                        ("codec_architecture_version", "architecture_version"),
                    ]:
                        if field in effective and effective[field] != spec[key]:
                            fail(
                                "method_overrides." + name + "." + field,
                                "conflicts with selected codec specification",
                            )
    if c.get("method") and not c["method"].startswith("CODEC_"):
        effective = copy.deepcopy(c)
        deferred = []
        if symbolic(effective.get("layer_indices")):
            deferred.append("layer_indices")
            effective["layer_indices"] = [0, 2, 4]
        try:
            method = resolve_method(effective, "plan" if planning else c["executor"])
        except (ValueError, OSError) as exc:
            fail("method", str(exc))
        c["scientific_resolution"] = {**asdict(method), "deferred_fields": deferred}
    for key, value in c.items():
        if symbolic(value) and not re.fullmatch(r"@[a-zA-Z0-9_.-]+(?:/[^\\]+)?", value):
            fail(key, "invalid symbolic artifact reference")
        if symbolic(value) and ".." in value.split("/"):
            fail(key, "unsafe artifact reference")
    return c
