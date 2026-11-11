"""Offline resolution of the native hierarchical block workflow and finite budgets."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from ncm_lab.runtime.budget import Budget
from ncm_lab.config.architecture import Architecture
from ncm_lab.config.canvas import resolve_canvas, validate_production
from ncm_lab.runtime.io import canonical_hash, implementation_hash, read, reference, root
from ncm_lab.teacher.model import model_identity


def positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError("finite integer allowance required: " + name)


BLOCK_CONTRACT = dict(
    schema="native_hierarchical_block8_v1",
    block_size=8,
    conditioning="prompt_and_committed_text_only_v1",
    cache="committed_causal_text_only_v1",
    prediction="one_native_hierarchy_traversal_per_block_v1",
    teacher_alignment="parent_P_minus_one_plus_answer_index_v1",
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
class Optimization:
    steps: int = 20000
    batch_size: int = 1
    accumulation: int = 16
    learning_rate: float = 0.00001
    new_learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    validation_interval: int = 250
    checkpoint_interval: int = 1000
    max_validation_views: int = 256
    head_chunk: int = 8
    shard_bytes: int = 67108864
    optimizer_tile_elements: int = 1048576
    optimizer_seed: int = 1729
    warmup_steps: int = 1000
    minimum_lr_ratio: float = 0.1

    def __post_init__(self):
        for name in (
            "steps",
            "batch_size",
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
        positive(self.warmup_steps, "warmup_steps", zero=True)
        for name in (
            "learning_rate",
            "new_learning_rate",
            "weight_decay",
            "gradient_clip",
            "minimum_lr_ratio",
        ):
            x = getattr(self, name)
            if type(x) not in {int, float} or not math.isfinite(x) or x < 0:
                raise ValueError("finite optimization option required: " + name)
        if (
            min(self.learning_rate, self.new_learning_rate, self.gradient_clip) <= 0
            or not 0 < self.minimum_lr_ratio <= 1
            or self.warmup_steps >= self.steps
            or self.head_chunk > 8
            or self.shard_bytes < 1024
        ):
            raise ValueError("invalid hierarchical optimization/schedule allowance")


@dataclass(frozen=True)
class Training(Optimization):
    auxiliary_weight: float = 0.1
    ar_weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size != 1:
            raise ValueError("hierarchical generator uses one context per microbatch")
        for x in (self.auxiliary_weight, self.ar_weight):
            if type(x) not in {int, float} or not math.isfinite(x) or x < 0:
                raise ValueError("finite hierarchical loss weights required")
        if self.auxiliary_weight <= 0:
            raise ValueError("hierarchical cumulative-code supervision must remain enabled")


@dataclass(frozen=True)
class CodecTraining(Optimization):
    steps: int = 10000
    batch_size: int = 4
    accumulation: int = 8
    learning_rate: float = 0.0001
    new_learning_rate: float = 0.0001
    warmup_steps: int = 500
    checkpoint_interval: int = 500
    prediction_weight: float = 0.1
    max_statistics_views: int = 4096

    def __post_init__(self):
        super().__post_init__()
        positive(self.max_statistics_views, "max_statistics_views")
        if (
            type(self.prediction_weight) not in {int, float}
            or not math.isfinite(self.prediction_weight)
            or self.prediction_weight < 0
        ):
            raise ValueError("finite codec consistency weight required")


@dataclass(frozen=True)
class Features:
    teacher_dtype: str = "bfloat16"
    target_train_positions: int = 10000000
    minimum_train_positions: int = 1000000
    max_dev_parents: int = 256

    def __post_init__(self):
        for name in ("target_train_positions", "minimum_train_positions", "max_dev_parents"):
            positive(getattr(self, name), name)
        if (
            self.teacher_dtype not in {"bfloat16", "float32"}
            or self.minimum_train_positions > self.target_train_positions
        ):
            raise ValueError("invalid frozen teacher subset/precision policy")


@dataclass(frozen=True)
class Decoding:
    max_new_tokens: int = 1024
    max_blocks: int = 128
    max_evaluation_examples: int = 32

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive(value, name)
        if self.max_new_tokens % 8:
            raise ValueError("hierarchical decoding uses an eight-aligned finite output cap")


@dataclass(frozen=True)
class Resources:
    reserve_free_bytes: int = 20000000000
    activation_reserve_bytes: int = 8000000000
    max_attempts_per_stage: int = 3
    max_write_bytes: int = 6000000000000
    max_phase_write_bytes: int = 18000000000000
    max_stall_seconds: int = 14400

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
        "codec",
        "features",
        "decoding",
        "resources",
        "budget",
        "source",
    }
    if set(value) != required or value["schema"] != "hierarchical_block8_workflow_v1":
        raise ValueError("versioned complete block8 workflow configuration required")
    if isinstance(value["model"], str):
        value["model"] = read(root() / value["model"])
    if value["device"] not in {"cpu", "cuda"} or value["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("explicit supported device/precision required")
    if value["hardware_profile"] not in {"cpu", "cuda", "spark"}:
        raise ValueError("invalid hardware profile")
    positive(value["seed"], "seed", zero=True)
    a = (
        Architecture(**value["architecture"])
        if value["architecture"]
        else resolve_canvas("xsmall").architecture()
    )
    if (
        a.canvas != 8
        or a.initial_queries != 4
        or (a.early_blocks, a.hierarchy_blocks, a.late_blocks) != (8, 16, 12)
    ):
        raise ValueError("hierarchical block8 requires native 8/16/12 layers, four queries and 32 updates")
    if not value["model"].get("fixture"):
        validate_production(a.to_dict())
    value["architecture"] = a.to_dict()
    value["block_contract"] = dict(BLOCK_CONTRACT)
    value["data"] = asdict(DataPolicy(**value["data"]))
    value["training"] = asdict(Training(**value["training"]))
    value["codec"] = asdict(CodecTraining(**value["codec"]))
    value["features"] = asdict(Features(**value["features"]))
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
    from ncm_lab.block8.budgets import source_work_budget, preparation_work_budget

    source_budget, value["source_work_budget"] = source_work_budget(value)
    prep_budget, value["preparation_work_budget"] = preparation_work_budget(value)
    value["stage_budgets"] = {"source": source_budget, "prepare": prep_budget}
    value["implementation_sha256"] = implementation_hash()
    value["identity"] = canonical_hash(value)
    return value
