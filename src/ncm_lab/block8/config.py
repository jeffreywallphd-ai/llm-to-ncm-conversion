"""Offline resolution; the block decoder has no embedded-hierarchy/codec contract."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.io import canonical_hash, implementation_hash, read, reference, root
from ncm_lab.teacher.model import model_identity


def positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError("finite integer allowance required: " + name)


@dataclass(frozen=True)
class Architecture:
    schema: str = "qwen_context_causal_block8_v1"
    block_size: int = 8
    mask_id: int = -1
    attention: str = "causal_context_bidirectional_active_v1"
    alignment: str = "active_same_position_ar_previous_position_v1"
    cache: str = "committed_causal_text_only_v1"

    def __post_init__(self):
        defaults = asdict_default_architecture()
        if asdict(self) != defaults or any(
            type(getattr(self, k)) is not type(v) for k, v in defaults.items()
        ):
            raise ValueError("the separate block8 architecture requires its exact versioned contract")


def asdict_default_architecture():
    return dict(
        schema="qwen_context_causal_block8_v1",
        block_size=8,
        mask_id=-1,
        attention="causal_context_bidirectional_active_v1",
        alignment="active_same_position_ar_previous_position_v1",
        cache="committed_causal_text_only_v1",
    )


@dataclass(frozen=True)
class DataPolicy:
    max_prompt_tokens: int = 512
    max_answer_tokens: int = 1024
    max_context_tokens: int = 1536
    max_parents: int = 252000
    max_views: int = 32256000
    max_row_bytes: int = 16777216
    max_index_bytes: int = 32000000000
    minimum_train: int = 90000
    minimum_dev: int = 2000
    minimum_test: int = 2000
    target_policy: str = "sealed_record_targets_v1"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "target_policy":
                positive(value, name)
        if (
            self.max_context_tokens < self.max_prompt_tokens + self.max_answer_tokens
            or self.max_answer_tokens > 1024
            or self.max_prompt_tokens > 512
            or self.max_index_bytes < 65536
            or self.max_parents > 1000000
            or self.target_policy != "sealed_record_targets_v1"
        ):
            raise ValueError("invalid bounded block8 data policy")


@dataclass(frozen=True)
class Training:
    steps: int = 1000
    accumulation: int = 16
    learning_rate: float = 0.00001
    mask_learning_rate: float = 0.0001
    weight_decay: float = 0.01
    ar_weight: float = 0.5
    fully_masked_fraction: float = 0.5
    gradient_clip: float = 1.0
    validation_interval: int = 100
    checkpoint_interval: int = 250
    max_validation_views: int = 256
    head_chunk: int = 8
    shard_bytes: int = 67108864
    optimizer_tile_elements: int = 1048576
    optimizer_seed: int = 1729
    activation_checkpointing: bool = True
    curriculum_end_steps: tuple = (50, 100, 200, 1000)

    def __post_init__(self):
        object.__setattr__(self, "curriculum_end_steps", tuple(self.curriculum_end_steps))
        for name in (
            "steps",
            "accumulation",
            "validation_interval",
            "checkpoint_interval",
            "max_validation_views",
            "head_chunk",
            "shard_bytes",
            "optimizer_tile_elements",
        ):
            positive(getattr(self, name), name)
        positive(self.optimizer_seed, "optimizer_seed", zero=True)
        for name in (
            "learning_rate",
            "mask_learning_rate",
            "weight_decay",
            "ar_weight",
            "fully_masked_fraction",
            "gradient_clip",
        ):
            x = getattr(self, name)
            if type(x) not in {int, float} or not math.isfinite(x) or x < 0:
                raise ValueError("finite nonnegative training option required: " + name)
        if (
            min(self.learning_rate, self.mask_learning_rate, self.gradient_clip) <= 0
            or not 0 < self.fully_masked_fraction <= 1
            or self.head_chunk > 8
            or self.shard_bytes < 1024
            or type(self.activation_checkpointing) is not bool
            or len(self.curriculum_end_steps) != 4
            or any(type(x) is not int or x <= 0 for x in self.curriculum_end_steps)
            or list(self.curriculum_end_steps) != sorted(set(self.curriculum_end_steps))
            or self.curriculum_end_steps[-1] != self.steps
        ):
            raise ValueError("invalid training/curriculum; declare increasing endpoints for 1,2,4,8 tokens")

    def block_at(self, completed):
        return next(b for b, end in zip((1, 2, 4, 8), self.curriculum_end_steps) if completed < end)


@dataclass(frozen=True)
class Decoding:
    refinement_passes: int = 4
    max_new_tokens: int = 1024
    max_blocks: int = 128
    max_evaluation_examples: int = 32

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive(value, name)
        if self.refinement_passes > 8 or self.max_new_tokens % 8:
            raise ValueError("decode with 1..8 refinement passes and an eight-aligned output cap")


@dataclass(frozen=True)
class Resources:
    reserve_free_bytes: int = 20000000000
    activation_reserve_bytes: int = 8000000000
    max_attempts_per_stage: int = 3
    max_write_bytes: int = 2000000000000
    max_phase_write_bytes: int = 6000000000000

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive(value, name, zero=name.endswith("reserve_bytes") or name == "reserve_free_bytes")
        if self.max_phase_write_bytes < self.max_write_bytes:
            raise ValueError("aggregate writes below per-attempt allowance")


def resolve(path=None):
    path = Path(path or root() / "configs/block8/experiment.yaml")
    value = read(path)
    required = {
        "schema",
        "model",
        "device",
        "dtype",
        "hardware_profile",
        "seed",
        "architecture",
        "data",
        "training",
        "decoding",
        "resources",
        "budget",
        "source",
    }
    if set(value) != required or value["schema"] != "block8_workflow_v1":
        raise ValueError("versioned complete block8 workflow configuration required")
    if isinstance(value["model"], str):
        value["model"] = read(root() / value["model"])
    if value["device"] not in {"cpu", "cuda"} or value["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("explicit supported device/precision required")
    if value["hardware_profile"] not in {"cpu", "cuda", "spark"}:
        raise ValueError("invalid hardware profile")
    positive(value["seed"], "seed", zero=True)
    value["architecture"] = asdict(Architecture(**value["architecture"]))
    value["data"] = asdict(DataPolicy(**value["data"]))
    value["training"] = asdict(Training(**value["training"]))
    value["decoding"] = asdict(Decoding(**value["decoding"]))
    value["resources"] = asdict(Resources(**value["resources"]))
    value["budget"] = asdict(Budget(**value["budget"]))
    if (
        value["decoding"]["max_new_tokens"] > value["data"]["max_answer_tokens"]
        or value["data"]["max_index_bytes"] > value["budget"]["max_storage_bytes"]
    ):
        raise ValueError("decode/index allowance exceeds declared data/storage capacity")
    source = value["source"]
    if set(source) != {"config", "max_candidates", "max_source_index_bytes", "max_candidate_bytes"}:
        raise ValueError("explicit bounded source adapter settings required")
    for k in source.keys() - {"config"}:
        positive(source[k], k)
    model_identity(value["model"])
    from ncm_lab.config.architecture import PIN

    if not value["model"].get("fixture") and (
        value["model"]["model_id"] != "Qwen/Qwen3-8B"
        or value["model"]["revision"] != PIN
        or value["model"]["tokenizer_revision"] != PIN
        or value["model"].get("thinking", False)
    ):
        raise ValueError("pinned non-thinking Qwen3-8B required")
    # Bind source settings now; editing their file later cannot change a sealed experiment.
    value["source"]["resolved_data"] = read(root() / source["config"])["data"]
    source_data = value["source"]["resolved_data"]
    bindings = {
        key: root() / source_data[key] for key in ("asset_lock", "source_policy") if key in source_data
    }
    if source_data.get("overlap", {}).get("artifact"):
        bindings["overlap_artifact"] = root() / source_data["overlap"]["artifact"]
    value["source"]["policy_receipts"] = {key: reference(path) for key, path in bindings.items()}
    value["model_asset_policy"] = reference(root() / "configs/data/assets.lock.json")
    value["implementation_sha256"] = implementation_hash()
    value["identity"] = canonical_hash(value)
    return value
