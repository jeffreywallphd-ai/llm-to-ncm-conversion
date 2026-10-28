"""Order-invariant analysis; paper contrasts bind exact immutable runs."""

from collections import defaultdict
import json
import statistics
from ncm_lab.evaluation.analysis import summarize, paired_group_interval
from ncm_lab.runtime.io import canonical_hash


def identities(row):
    value = row.get("identities")
    if value:
        if (
            canonical_hash(value["condition"]) != value["condition_sha256"]
            or canonical_hash(value["treatment"]) != value["treatment_sha256"]
        ):
            raise ValueError("prediction identity hash mismatch")
        return value["condition"], value["treatment_sha256"], canonical_hash(value["observation"])
    condition = {
        key: row.get(key)
        for key in (
            "dataset",
            "split",
            "policy_hash",
            "data_tier",
            "comparison",
            "result_origin",
            "result_scope",
            "full_official_split",
            "runtime_condition",
            "target_revision",
            "request_set_sha256",
        )
    }
    condition["comparison"] = condition["comparison"] or "equal_data"
    treatment = canonical_hash(
        {
            key: row.get(key)
            for key in (
                "method",
                "training_seed",
                "codec_seed",
                "checkpoint_sha256",
                "codec_sha256",
                "codec_objective",
                "treatment_id",
                "training_run_id",
            )
        }
    )
    observation = canonical_hash(
        [row.get("run_id"), row["original_id"], row.get("repetition_id"), condition, treatment]
    )
    return condition, treatment, observation


def _table(method, seed, condition, treatment, records):
    summary = summarize(records)
    official = defaultdict(list)
    for row in records:
        for key, value in row.get("official_scores", {}).items():
            if isinstance(value, (int, float)):
                official[key].append(value)
    cycles = [c for row in records for c in (row.get("decoder") or {}).get("cycles", [])]
    generation = [r for r in records if r.get("generation_seconds") is not None]
    seconds = sum(r["generation_seconds"] for r in generation)
    tokens = sum(r.get("output_tokens", 0) for r in generation)
    table = {
        "method": method,
        "training_seed": seed,
        "condition": condition,
        "treatment_sha256": treatment,
        **summary,
        "official_metrics": {
            k: {
                "mean": statistics.mean(v),
                "denominator": len(v),
                "missing": len(records) - len(v),
                "all_requests_mean_with_failures_zero": sum(v) / len(records),
            }
            for k, v in official.items()
        },
        "failed_requests_in_denominator": summary["errors"],
        "accepted_prefix": [c["accepted"] for c in cycles],
        "mean_accepted_prefix": statistics.mean(c["accepted"] for c in cycles) if cycles else None,
        "committed_tokens_per_cycle": sum(1 + c["accepted"] for c in cycles) / len(cycles)
        if cycles
        else None,
        "decoder_missing_requests": sum(r.get("decoder") is None for r in records),
        "mean_output_tokens": statistics.mean(r.get("output_tokens", 0) for r in records),
        "generation_missing_requests": len(records) - len(generation),
        "warm_generation_tokens_per_second": tokens / seconds if seconds > 0 else None,
        "generation_seconds_per_token": seconds / tokens if tokens > 0 else None,
        "accepted_prefix_survival": [
            {
                "offset": j,
                "accepted": sum(c["accepted"] >= j for c in cycles),
                "eligible_cycles": sum(c.get("proposed", 0) >= j for c in cycles),
            }
            for j in range(1, 16)
        ],
    }
    count = sum(r.get("official_scores", {}).get("instruction_count", 0) for r in records)
    if count:
        for name in ("strict", "loose"):
            table["instruction_" + name + "_accuracy"] = (
                sum(r.get("official_scores", {}).get(name + "_instruction_correct", 0) for r in records)
                / count
            )
        table["instruction_denominator"] = count
    endpoints = [r["endpoint"] for r in records if r.get("endpoint")]
    if endpoints:
        if len({(e["name"], e["kind"], e["units"]) for e in endpoints}) != 1:
            raise ValueError("cannot combine incompatible primary endpoints")
        scored = [e["value"] for e in endpoints if e["status"] == "scored"]
        table["primary_endpoint"] = {
            **{key: endpoints[0][key] for key in ("name", "kind", "units")},
            "scored_mean": statistics.mean(scored) if scored else None,
            "scored_denominator": len(scored),
            "all_requests_denominator": len(records),
            "failure_inclusive_mean_with_zero": sum(scored) / len(records),
            "missing": sum(e["status"] == "missing" for e in endpoints),
            "errors": sum(e["status"] == "error" for e in endpoints),
        }
    return table


def _pair(left, right, contrast, condition, seed, shared=False):
    a = {(r["original_id"], r.get("repetition_id", 0)): r for r in left}
    b = {(r["original_id"], r.get("repetition_id", 0)): r for r in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError(f"{contrast}: multiple observations for a role/request/block; lock retry selection")
    latency = []
    endpoint = []
    unmatched = []
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            unmatched.append(
                {"contrast": contrast, "request": key, "missing_role": "left" if key not in a else "right"}
            )
            continue
        x, y = a[key], b[key]
        latency.append({"group_id": x["group_id"], "difference": y["request_seconds"] - x["request_seconds"]})
        ex, ey = x.get("endpoint"), y.get("endpoint")
        if ex and ey:
            if (ex["name"], ex["units"]) != (ey["name"], ey["units"]):
                raise ValueError("cannot pair incompatible endpoint semantics")
            endpoint.append(
                {"group_id": x["group_id"], "difference": (ey["value"] or 0) - (ex["value"] or 0)}
            )
    common = {
        "contrast_id": contrast,
        "training_seed": seed,
        "condition": condition,
        "paired_requests": len(latency),
        "left_requests": len(left),
        "right_requests": len(right),
        "unmatched_requests": len(unmatched),
        "failed_left": sum(bool(r.get("error")) for r in left),
        "failed_right": sum(bool(r.get("error")) for r in right),
        "shared_reference_reused": shared,
        "estimand": "all attempted paired requests; failures retained; endpoint failures zero",
        "timing_repetitions_are_training_replicates": False,
    }
    return (
        {**common, **paired_group_interval(latency, iterations=500)},
        ({**common, **paired_group_interval(endpoint, iterations=500)} if endpoint else None),
        unmatched,
    )


def analyze_records(rows, contrasts=None, duplicate_policy="reject"):
    unique = {}
    groups = defaultdict(list)
    for row in rows:
        condition, treatment, observation = identities(row)
        if observation in unique:
            if duplicate_policy == "deduplicate_identical" and unique[observation] == row:
                continue
            raise ValueError("duplicate observation identity; rows cannot silently overwrite: " + observation)
        unique[observation] = row
        groups[
            (
                row.get("method", "unknown"),
                row.get("training_seed"),
                json.dumps(condition, sort_keys=True),
                treatment,
            )
        ].append(row)
    tables = [
        _table(
            method,
            seed,
            json.loads(condition),
            treatment,
            sorted(values, key=lambda r: (r["original_id"], r.get("repetition_id", 0))),
        )
        for (method, seed, condition, treatment), values in sorted(
            groups.items(), key=lambda item: repr(item[0])
        )
    ]
    differences = []
    endpoints = []
    missing = []
    if contrasts is not None:
        for contrast in sorted(contrasts, key=lambda c: c["id"]):
            role_rows = {
                role: [
                    r
                    for r in unique.values()
                    if r.get("identities", {}).get("run_id", r.get("run_id")) in refs
                ]
                for role, refs in contrast["roles"].items()
            }
            names = sorted(role_rows)
            if len(names) != 2:
                raise ValueError("each contrast requires exactly two treatment roles")
            for field in contrast.get("held_constant", []):
                values = {
                    json.dumps(identities(r)[0].get(field), sort_keys=True)
                    for records in role_rows.values()
                    for r in records
                }
                if len(values) > 1:
                    raise ValueError(contrast["id"] + ": held-constant condition differs: " + field)
            result, endpoint, unmatched = _pair(
                role_rows[names[0]],
                role_rows[names[1]],
                contrast["id"],
                contrast.get("condition"),
                contrast.get("seed_pair"),
                bool(contrast.get("shared_reference_roles")),
            )
            result.update(left=names[0], right=names[1])
            differences.append(result)
            if endpoint:
                endpoints.append(endpoint)
            missing.extend(unmatched)
            for role, records in role_rows.items():
                if not records:
                    missing.append(
                        {
                            "contrast": contrast["id"],
                            "missing_role": role,
                            "reason": "no selected observations",
                        }
                    )
    else:
        by_condition = defaultdict(list)
        for key, records in groups.items():
            by_condition[key[2]].append((key, records))
        for condition, values in sorted(by_condition.items()):
            values = sorted(values, key=lambda pair: repr(pair[0]))
            for i, (ka, a) in enumerate(values):
                for kb, b in values[i + 1 :]:
                    if ka[1] != kb[1] and not any(r.get("shared_reference") for r in a + b):
                        continue
                    result, endpoint, unmatched = _pair(
                        a, b, canonical_hash([ka, kb]), json.loads(condition), ka[1]
                    )
                    result.update(left=ka[0], right=kb[0], left_treatment=ka[3], right_treatment=kb[3])
                    differences.append(result)
                    if endpoint:
                        endpoints.append(endpoint)
                    missing.extend(unmatched)
    return {
        "per_seed": tables,
        "paired_latency_differences": differences,
        "paired_endpoint_differences": endpoints,
        "unmatched_observations": missing,
        "duplicate_policy": duplicate_policy,
        "observation_count": len(unique),
        "weighting": "requests within independent codec/student seed pair; groups bootstrapped within seed",
        "across_seed_aggregation": None,
        "scope": "explicit input observations; claims require locked contrast bindings and evidence gates",
    }
