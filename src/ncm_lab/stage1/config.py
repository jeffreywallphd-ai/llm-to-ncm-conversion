"""Typed independent Stage 1 settings and offline proposal resolution."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from ncm_lab.config.canvas import resolve_canvas, canvas_metadata, validate_production
from ncm_lab.config.stages import config
from ncm_lab.runtime.io import read, root, canonical_hash, implementation_hash
from ncm_lab.stage1.policy import SCHEMA, Policy, GENERATOR_SELECTION, CODEC_SELECTION

STAGES = ("prepare", "teacher", "codec", "targets", "generator", "evaluate")
ACQUIRE = "ncm prepare --config configs/validation/real_pilot.yaml"


@dataclass(frozen=True)
class DataSettings:
    train_examples: int = 2000
    dev_examples: int = 200
    test_examples: int = 200
    split_seed: int = 17
    validation_seed: int = 20261029
    dev_fraction: float = 0.15
    test_fraction: float = 0.15
    max_candidate_bytes: int = 134217728
    diagnostic: bool = False

    def __post_init__(self):
        if type(self.diagnostic) is not bool:
            raise ValueError("diagnostic must be an explicit boolean")
        for name in ("train_examples", "dev_examples", "test_examples", "max_candidate_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("positive finite data qualification allowance: " + name)
        for name in ("split_seed", "validation_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("nonnegative independent data/validation seed required")
        if not (
            0 < self.dev_fraction < 1
            and 0 < self.test_fraction < 1
            and self.dev_fraction + self.test_fraction < 1
        ):
            raise ValueError("nonempty group partition fractions required")
        if not self.diagnostic and (self.train_examples, self.dev_examples, self.test_examples) != (
            2000,
            200,
            200,
        ):
            raise ValueError("reduced data requires diagnostic: true; not the standard Stage 1 result")


@dataclass(frozen=True)
class Resources:
    reserve_free_bytes: int = 20000000000
    max_retained_bytes: int = 700000000000
    max_phase_retained_bytes: int = 2000000000000
    max_generator_steps: int = 1000
    retry_allowance: int = 3
    work_margin: float = 1.25
    max_controller_bytes: int = 100000000
    max_controller_actions: int = 100

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "work_margin":
                if type(value) not in {int, float} or not 1 <= value <= 4:
                    raise ValueError("finite resource allowance margin in 1..4 required")
            elif type(value) is not int or value <= 0:
                raise ValueError("positive finite Stage 1 resource allowance: " + name)
        if self.max_phase_retained_bytes < self.max_retained_bytes:
            raise ValueError("aggregate retained allowance below run allowance")


def resolve(
    path=None, *, canvas_size=None, seed=None, data_config=None, codec_steps=None, generator_steps=None
):
    path = Path(path or root() / "configs/stage1/experiment.yaml").resolve()
    value = read(path)
    allowed = {"schema", "canvas_size", "seed", "stages", "qualification", "resources", "notes"}
    if set(value) - allowed or value.get("schema") != SCHEMA or set(value.get("stages", {})) != set(STAGES):
        raise ValueError("typed Stage 1 experiment schema and all six stage configs required")
    selected = resolve_canvas(canvas_size if canvas_size is not None else value.get("canvas_size", "medium"))
    selected_seed = seed if seed is not None else value.get("seed", 17)
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    qualification = asdict(DataSettings(**value.get("qualification", {})))
    resources = asdict(Resources(**value.get("resources", {})))
    stages = {}
    for name, filename in value["stages"].items():
        cfg = config(filename, _bind_canvas=False)
        if cfg["stage"] != {"prepare": "stage1_prepare", "evaluate": "stage1_evaluate"}.get(name, name):
            raise ValueError("Stage 1 stage/config mismatch: " + name)
        stages[name] = cfg
    if data_config is not None:
        other = config(data_config, _bind_canvas=False)
        if other["stage"] not in {"data", "stage1_prepare"}:
            raise ValueError("--data-config must be an approved native data configuration")
        stages["prepare"]["data"] = other["data"]
    model = stages["generator"]["model"]
    base_architecture = __import__("ncm_lab.config.architecture", fromlist=["Architecture"]).Architecture(
        **stages["generator"]["architecture"]
    )
    architecture = selected.architecture(base_architecture).to_dict()
    if not model.get("fixture"):
        validate_production(architecture)
    for name, cfg in stages.items():
        cfg.update(model=deepcopy(model), architecture=deepcopy(architecture), inputs={}, output_ref=None)
        cfg["budget"]["enforce_wall_time_limits"] = False
        cfg.pop("experiment_context", None)
        cfg["training"]["seed"] = selected_seed
        cfg["training"]["validation_seed"] = qualification["validation_seed"]
        cfg.setdefault(
            "stage1",
            asdict(Policy(selection=GENERATOR_SELECTION if name == "generator" else CODEC_SELECTION)),
        )
        cfg["stage1_data"] = qualification
        if name in {"codec", "generator"}:
            cfg["training"].update(objective="steps", warmup_processed_tokens=0, recovery_probe=False)
    for name, override in (("codec", codec_steps), ("generator", generator_steps)):
        if override is not None:
            if type(override) is not int or override <= 0:
                raise ValueError("positive total optimizer-step override required")
            stages[name]["training"]["steps"] = override
    if stages["generator"]["training"]["steps"] > resources["max_generator_steps"]:
        raise ValueError("generator target exceeds the explicit finite extension ceiling")
    if any(
        stages[n]["dtype"] != dtype
        for n, dtype in (
            ("teacher", "bfloat16"),
            ("codec", "float32"),
            ("targets", "float32"),
            ("generator", "bfloat16"),
        )
    ):
        if not model.get("fixture"):
            raise ValueError(
                "Stage 1 requires independent BF16 teacher / FP32 codec+targets / BF16 generator precision"
            )
    if stages["prepare"]["data"].get("overlength", "exclude") != "exclude":
        raise ValueError("Stage 1 admits complete references only; overlength must be exclude")
    stages["prepare"]["data"]["seed"] = qualification["split_seed"]
    settings = {
        "schema": SCHEMA,
        "canvas": canvas_metadata(architecture),
        "seed": selected_seed,
        "qualification": qualification,
        "resources": resources,
        "stages": stages,
        "initialization": "pinned pretrained backbone and seeded added modules; no pilot checkpoint",
        "initialization_sha256": canonical_hash(
            {
                "model": model,
                "architecture": architecture,
                "seed": selected_seed,
                "implementation": implementation_hash(),
            }
        ),
        "implementation_sha256": implementation_hash(),
        "config_path": str(path),
    }
    settings["contract_sha256"] = canonical_hash({k: v for k, v in settings.items() if k != "config_path"})
    return settings
