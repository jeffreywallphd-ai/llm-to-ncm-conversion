"""Typed stage options, immutable configuration resolution, and explicit handoffs."""

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from ncm_lab.config.architecture import Architecture
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.io import read, root, resolve_reference, digest


@dataclass(frozen=True)
class Training:
    steps: int = 2
    epochs: int = 100
    batch_size: int = 1
    accumulation: int = 1
    learning_rate: float = 1e-5
    new_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    auxiliary_weight: float = 0.1
    gradient_clip: float = 1.0
    prediction_weight: float = 0.1
    head_positions: int = 32
    head_chunk: int = 8
    validation_interval: int = 1
    checkpoint_interval: int = 1
    shard_bytes: int = 67108864
    warmup_processed_tokens: int = 0
    max_validation_examples: int = 32
    minimum_unique_answer_positions: int = 0
    max_processed_tokens: int = 100000000
    recovery_probe: bool = False
    seed: int = 17

    def __post_init__(self):
        if self.recovery_probe and self.warmup_processed_tokens:
            raise ValueError("full-training recovery probe must skip module-only warmup")
        for key, value in asdict(self).items():
            if key == "recovery_probe":
                if not isinstance(value, bool):
                    raise ValueError("recovery_probe must be boolean")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("finite nonnegative training option required: " + key)
        for key in (
            "steps",
            "epochs",
            "batch_size",
            "accumulation",
            "validation_interval",
            "checkpoint_interval",
            "max_validation_examples",
            "max_processed_tokens",
            "head_positions",
            "head_chunk",
            "shard_bytes",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError("positive integer training limit required: " + key)
        if (
            not 1 <= self.head_positions <= 32
            or not 1 <= self.head_chunk <= 8
            or self.shard_bytes < 1024
            or self.learning_rate <= 0
            or self.new_learning_rate <= 0
            or self.gradient_clip <= 0
        ):
            raise ValueError("invalid learning rate or bounded head/checkpoint configuration")


def config(path):
    path = Path(path)
    if not path.is_absolute():
        path = root() / path
    value = read(path)
    allowed = {
        "schema",
        "stage",
        "model",
        "architecture",
        "budget",
        "device",
        "dtype",
        "inputs",
        "output_ref",
        "training",
        "data",
        "preparation",
        "notes",
        "threads",
        "run_root",
        "hardware_profile",
        "monitoring",
    }
    if set(value) - allowed:
        raise ValueError("unknown native stage fields: " + str(sorted(set(value) - allowed)))
    if value.get("schema") != "native_stage_v1" or value.get("stage") not in {
        "data",
        "teacher",
        "codec",
        "targets",
        "generator",
        "validation",
        "profile",
        "prepare",
    }:
        raise ValueError("native stage schema required; archived configurations are incompatible")

    def expand(field):
        if isinstance(value.get(field), str):
            value[field] = read(root() / value[field])

    for field in ("model", "architecture", "budget"):
        expand(field)
    value["architecture"] = Architecture(**value.get("architecture", {})).to_dict()
    value["budget"] = asdict(Budget(**value["budget"]))
    value["training"] = asdict(Training(**value.get("training", {})))
    value.setdefault("device", "cpu")
    value.setdefault("dtype", "float32")
    value.setdefault("threads", 1)
    value.setdefault("hardware_profile", "cpu" if value["device"] == "cpu" else "cuda")
    if value["hardware_profile"] not in {"cpu", "cuda", "spark"}:
        raise ValueError("unknown hardware profile")
    if value["dtype"] not in {"float32", "bfloat16"} or value["device"] not in {"cpu", "cuda"}:
        raise ValueError("native path supports explicit CPU/CUDA and FP32/BF16 only")
    if (
        isinstance(value["threads"], bool)
        or not isinstance(value["threads"], int)
        or not 1 <= value["threads"] <= 256
    ):
        raise ValueError("threads must be an integer in 1..256")
    value.setdefault("inputs", {})
    value.setdefault("data", {})
    value.setdefault("model", {})
    from ncm_lab.monitoring.records import Monitoring

    if isinstance(value.get("monitoring"), str):
        value["monitoring"] = read(root() / value["monitoring"])
    value["monitoring"] = asdict(Monitoring(**value.get("monitoring", {})))
    if value["model"] and not value["model"].get("fixture"):
        a = value["architecture"]
        if (a["early_blocks"], a["hierarchy_blocks"], a["late_blocks"], a["canvas"], a["latent_width"]) != (
            8,
            16,
            12,
            1024,
            384,
        ):
            raise ValueError(
                "real initial architecture requires 8/16/12 blocks, 32 updates, 1024 canvas and width 384"
            )
    return value


def bind_inputs(value):
    result = {}
    missing = []
    for key, path in value.get("inputs", {}).items():
        try:
            manifest = resolve_reference(path)
            result[key] = {"manifest": str(manifest), "sha256": digest(manifest)}
        except FileNotFoundError:
            missing.append(f"{key}: missing explicit artifact handoff {path}")
    return result, missing
