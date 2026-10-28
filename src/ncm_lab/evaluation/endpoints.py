"""Typed official endpoints; continuous scores never become fabricated correctness."""

import math

ENDPOINTS = {
    "gsm8k": ("exact_match", "binary", "fraction"),
    "ifeval": ("prompt_strict_accuracy", "binary", "fraction"),
    "longbench": ("rouge_l", "continuous", "official ROUGE-L fraction"),
    "humaneval_plus": (
        "pass_at_1",
        "binary",
        "single-completion official pass fraction; verified isolation required",
    ),
    "finqa": ("execution_accuracy", "binary", "fraction"),
    "tatqa": ("exact_match", "binary", "fraction"),
}


def endpoint_result(dataset, scores, error=None, scorer=None):
    name, kind, units = ENDPOINTS[dataset]
    value = scores.get(name)
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if valid and not 0 <= value <= 1:
        raise ValueError(f"{dataset}: {name} outside official fraction range")
    if valid and kind == "binary" and value not in (0, 1):
        raise ValueError(f"{dataset}: {name} must be binary")
    status = "error" if error else "scored" if valid else "missing"
    return {
        "schema_version": 1,
        "name": name,
        "kind": kind,
        "units": units,
        "value": value if valid and not error else None,
        "status": status,
        "scorer": scorer,
        "correct": (bool(value == 1) if valid and not error else False) if kind == "binary" else None,
        "error": error or (None if valid else f"missing required official metric {name}"),
        "failure_inclusive_policy": "zero for failed/unscored requests; report scored-only denominator separately",
    }
