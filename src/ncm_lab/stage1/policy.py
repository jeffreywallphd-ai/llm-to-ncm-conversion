"""Versioned scientific policies; importing these allocates no model tensors."""

from dataclasses import dataclass, asdict
import math

SCHEMA = "stage1_single_pass_v1"
REDUCTION = "equal_actual_microbatch_objectives_v1"
CODEC_SELECTION = "stage1_codec_fixed_components_v1"
GENERATOR_SELECTION = "stage1_mean_example_text_ce_v1"


@dataclass(frozen=True)
class Policy:
    schema: str = SCHEMA
    selection: str = CODEC_SELECTION
    reduction: str = REDUCTION
    progress_interval: int = 10
    comparison_atol: float = 1e-7
    comparison_rtol: float = 1e-5
    combined_min_delta: float = 0.0
    generator_target_cast: str = "explicit_target_to_generator_parameter_dtype_v1"

    def __post_init__(self):
        if self.schema != SCHEMA or self.reduction != REDUCTION:
            raise ValueError("unsupported Stage 1 scientific/reduction contract")
        if self.selection not in {CODEC_SELECTION, GENERATOR_SELECTION}:
            raise ValueError("unsupported Stage 1 selection policy")
        if type(self.progress_interval) is not int or self.progress_interval <= 0:
            raise ValueError("positive Stage 1 progress interval required")
        for value in (self.comparison_atol, self.comparison_rtol, self.combined_min_delta):
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative comparison tolerances required")
        if self.generator_target_cast != "explicit_target_to_generator_parameter_dtype_v1":
            raise ValueError("unsupported target precision boundary")


def codec_gate(initial, selected, policy, *, artifacts_valid=True):
    policy = Policy(**policy) if isinstance(policy, dict) else policy
    keys = ("loss", "mse_full", "mse_cutoff", "head_kl")
    valid = artifacts_valid and all(
        type(row.get(k)) in {int, float} and math.isfinite(row[k]) and row[k] >= 0
        for row in (initial, selected)
        for k in keys
    )
    deltas = {k: selected[k] - initial[k] for k in keys} if valid else {}
    tolerances = (
        {k: policy.comparison_atol + policy.comparison_rtol * abs(initial[k]) for k in keys} if valid else {}
    )
    improved = valid and deltas["loss"] < -policy.combined_min_delta
    components = valid and all(deltas[k] <= tolerances[k] for k in keys[1:])
    return {
        "schema": "stage1_codec_qualification_v1",
        "policy": asdict(policy),
        "artifacts_valid": bool(valid),
        "training_completed": True,
        "codec_qualified_for_generator": bool(improved and components),
        "status": "qualified" if improved and components else "review_required" if valid else "invalid",
        "initial": initial,
        "selected": selected,
        "component_deltas": deltas,
        "comparison_tolerances": tolerances,
        "combined_improved": bool(improved),
        "components_without_regression": bool(components),
        "interpretation": "operational comparison of the same fixed plan, not significance or convergence",
    }
