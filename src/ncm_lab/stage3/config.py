"""Independent Stage 3 recipe, resolved without scanning assets or loading models."""

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from ncm_lab.config.canvas import resolve_canvas, canvas_metadata, validate_production
from ncm_lab.config.stages import config
from ncm_lab.runtime.io import root, read, canonical_hash, implementation_hash
from ncm_lab.stage1.config import Resources
from ncm_lab.stage3.schedule import Schedule

SCHEMA = "stage3_single_pass_v1"
MANIFEST = "stage3_experiment_v1"
STAGES = ("source", "distill", "prepare", "teacher", "codec", "targets", "generator", "evaluate")
GATE = "validity_required_quality_advisory_v1"


@dataclass(frozen=True)
class Qualification:
    train_examples: int = 100000
    minimum_train_examples: int = 90000
    dev_examples: int = 2000
    test_examples: int = 2000
    split_seed: int = 17
    validation_seed: int = 20261029
    dev_fraction: float = 0.15
    test_fraction: float = 0.15
    max_candidate_bytes: int = 4000000000
    max_source_index_bytes: int = 32000000000
    diagnostic: bool = False

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k in {"dev_fraction", "test_fraction"}:
                if not 0 < v < 1:
                    raise ValueError("nonempty split fractions required")
            elif k == "diagnostic":
                if type(v) is not bool:
                    raise ValueError("explicit diagnostic flag required")
            elif type(v) is not int or v < (0 if k.endswith("seed") else 1):
                raise ValueError("finite positive qualification limits required")
        if self.dev_fraction + self.test_fraction >= 1 or self.minimum_train_examples > self.train_examples:
            raise ValueError("invalid partition/minimum training count")


def resolve(path=None, *, canvas_size=None, seed=None, data_config=None, target_policy=None, source=None):
    path = Path(path or root() / "configs/stage3/experiment.yaml").resolve()
    recipe = read(path)
    if (
        recipe.get("schema") != SCHEMA
        or set(recipe)
        - {
            "schema",
            "canvas_size",
            "seed",
            "stages",
            "qualification",
            "resources",
            "distillation",
            "target_policy",
            "notes",
            "previous_locked_manifests",
            "evaluation",
        }
        or set(recipe["stages"]) != set(STAGES)
    ):
        raise ValueError("independent typed Stage 3 recipe and all stages required")
    if source not in {None, "tulu"}:
        raise ValueError("Stage 3 supports pinned Tulu only")
    q = asdict(Qualification(**recipe["qualification"]))
    resources = asdict(Resources(**recipe["resources"]))
    selected = resolve_canvas(canvas_size or recipe["canvas_size"])
    selected_seed = seed if seed is not None else recipe["seed"]
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("nonnegative training seed required")
    policy = target_policy or recipe["target_policy"]
    if policy not in {"teacher-consistent", "reference-only"}:
        raise ValueError("explicit supported target policy required")
    distillation = recipe["distillation"]
    if set(distillation) != {
        "schema",
        "max_candidates",
        "max_record_attempts",
        "repetition_threshold",
        "authoritative_components",
    }:
        raise ValueError("typed distillation policy required")
    if (
        distillation["schema"] != "sequence_distillation_v1"
        or type(distillation["max_candidates"]) is not int
        or distillation["max_candidates"] <= 0
        or type(distillation["max_record_attempts"]) is not int
        or not 1 <= distillation["max_record_attempts"] <= 3
        or not 0 < distillation["repetition_threshold"] <= 1
    ):
        raise ValueError("finite candidate/retry and repetition limits required")
    evaluation = recipe.get("evaluation", {"warmup": 1, "repeats": 3})
    if (
        set(evaluation) != {"warmup", "repeats"}
        or any(type(evaluation[k]) is not int or not 0 <= evaluation[k] <= 10 for k in evaluation)
        or evaluation["repeats"] < 1
    ):
        raise ValueError("bounded declared latency warmup/repeats required")
    if not isinstance(distillation["authoritative_components"], list) or any(
        not isinstance(x, str) or not x for x in distillation["authoritative_components"]
    ):
        raise ValueError("explicit authoritative component allowlist required")
    from ncm_lab.stage3.source_index import SourceIndex

    source_index = asdict(
        SourceIndex(
            max_bytes=q["max_source_index_bytes"],
            candidate_bytes=q["max_candidate_bytes"],
            reserve_free_bytes=resources["reserve_free_bytes"],
        )
    )
    stages = {n: config(p, _bind_canvas=False) for n, p in recipe["stages"].items()}
    if data_config:
        stages["source"]["data"] = config(data_config, _bind_canvas=False)["data"]
    from ncm_lab.stage1.policy import CODEC_SELECTION, GENERATOR_SELECTION

    expected_roles = {
        "source": "stage3_source",
        "distill": "stage3_distill",
        "prepare": "stage3_qualify",
        "evaluate": "stage3_evaluate",
    }
    for name, cfg in stages.items():
        if cfg["stage"] != expected_roles.get(name, name):
            raise ValueError("Stage 3 stage/config role mismatch: " + name)
        if name in {"codec", "generator"}:
            expected_selection = GENERATOR_SELECTION if name == "generator" else CODEC_SELECTION
            if cfg.get("stage1", {}).get("selection") != expected_selection:
                raise ValueError("Stage 3 requires its fixed selection/reduction contract: " + name)
            training = cfg["training"]
            if (
                training["objective"] != "steps"
                or training["warmup_processed_tokens"]
                or training["recovery_probe"]
            ):
                raise ValueError(
                    "Stage 3 requires completed-update training without backbone-freezing warmup/probes"
                )
            if training["auxiliary_weight"] != 0.1 or training["prediction_weight"] != 0.1:
                raise ValueError("Stage 3 retains its existing codec and generator objectives")
    model = stages["generator"]["model"]
    from ncm_lab.config.architecture import Architecture

    architecture = selected.architecture(Architecture(**stages["generator"]["architecture"])).to_dict()
    if not model.get("fixture"):
        validate_production(architecture)
        if stages["source"]["data"].get("format") != "tulu_parquet":
            raise ValueError("production Stage 3 requires the approved pinned Tulu adapter")
        if q["diagnostic"]:
            raise ValueError("diagnostic reduced data is software-fixture only")
    for name, cfg in stages.items():
        cfg.update(model=deepcopy(model), architecture=deepcopy(architecture), inputs={}, output_ref=None)
        cfg["budget"]["enforce_wall_time_limits"] = False
        cfg["training"].update(seed=selected_seed, validation_seed=q["validation_seed"])
        cfg["stage1_data"] = q
        cfg["stage3"] = {
            "schema": SCHEMA,
            "quality_policy": GATE,
            "target_policy": policy,
            "cache_retained_bytes": q["max_candidate_bytes"],
            "source_index": source_index,
            "payload_write_caps": {
                "per_run": q["max_candidate_bytes"] * 10,
                "aggregate": q["max_candidate_bytes"] * 10 * resources["retry_allowance"],
            },
            "distillation": distillation,
            "previous_locked_manifests": recipe.get("previous_locked_manifests", []),
            "evaluation": recipe.get("evaluation", {"warmup": 1, "repeats": 3}),
        }
        if name == "source":
            source_writes = max(q["max_candidate_bytes"] * 10, q["max_source_index_bytes"] * 4)
            cfg["stage3"]["payload_write_caps"] = {
                "per_run": source_writes,
                "aggregate": source_writes * resources["retry_allowance"],
            }
        if name in {"codec", "generator"}:
            saved_schedule = cfg.pop("stage3_schedule")
            if saved_schedule["total"] != cfg["training"]["steps"]:
                raise ValueError("schedule horizon must equal completed-update target")
            cfg["stage3"]["schedule"] = asdict(Schedule(**saved_schedule))
            cfg["training"].update(objective="steps", warmup_processed_tokens=0, recovery_probe=False)
        elif name == "source":
            cfg["data"].update(max_answer_tokens=selected.positions, seed=q["split_seed"])
    if stages["generator"]["training"]["steps"] > resources["max_generator_steps"]:
        raise ValueError("generator target exceeds explicit ceiling")
    for name, dtype in (
        ("teacher", "bfloat16"),
        ("distill", "bfloat16"),
        ("codec", "float32"),
        ("targets", "float32"),
        ("generator", "bfloat16"),
    ):
        if not model.get("fixture") and stages[name]["dtype"] != dtype:
            raise ValueError("independent Stage 3 precision policies required")
    standard = (
        not model.get("fixture")
        and selected.positions == 256
        and selected_seed == 17
        and policy == "teacher-consistent"
        and (q["train_examples"], q["minimum_train_examples"], q["dev_examples"], q["test_examples"])
        == (100000, 90000, 2000, 2000)
        and all(
            (
                stages[n]["training"]["steps"],
                stages[n]["training"]["batch_size"],
                stages[n]["training"]["accumulation"],
            )
            == expected
            for n, expected in (("codec", (10000, 4, 8)), ("generator", (20000, 1, 16)))
        )
    )
    if standard:
        defaults = read(root() / "configs/stage3/experiment.yaml")
        standard = all(
            recipe.get(k) == defaults.get(k)
            for k in ("qualification", "resources", "distillation", "evaluation", "previous_locked_manifests")
        )
        for name in STAGES:
            baseline = config(root() / defaults["stages"][name], _bind_canvas=False)
            for key in ("training", "dtype", "device", "monitoring"):
                standard = standard and stages[name][key] == baseline[key]
            if name in {"codec", "generator"}:
                standard = standard and stages[name]["stage3"]["schedule"] == baseline["stage3_schedule"]
        default_data = config(root() / defaults["stages"]["source"], _bind_canvas=False)["data"]
        standard = standard and stages["source"]["data"] == default_data
    settings = dict(
        schema=SCHEMA,
        canvas=canvas_metadata(architecture),
        seed=selected_seed,
        qualification=q,
        resources=resources,
        stages=stages,
        target_policy=policy,
        recipe_label="standard_research_candidate" if standard else "explicit_variant",
        initialization="pinned pretrained backbone; seeded added modules and fresh codec",
        initialization_sha256=canonical_hash([model, architecture, selected_seed]),
        implementation_sha256=implementation_hash(),
        config_path=str(path),
    )
    settings["contract_sha256"] = canonical_hash({k: v for k, v in settings.items() if k != "config_path"})
    return settings
