from collections import defaultdict
import math
import random
import statistics


def summarize(records, elapsed_seconds=None):
    rows = list(records)
    latencies = sorted(r["request_seconds"] for r in rows)
    if any(not math.isfinite(t) or t < 0 for t in latencies):
        raise ValueError("invalid request latency")
    binary = [
        r for r in rows if r.get("endpoint", {}).get("kind") == "binary" or isinstance(r.get("correct"), bool)
    ]
    if binary and len(binary) != len(rows):
        raise ValueError("cannot aggregate binary and continuous endpoints as one accuracy")
    correct = sum(r.get("correct") is True and not r.get("error") for r in binary) if binary else None
    elapsed = elapsed_seconds if elapsed_seconds is not None else sum(latencies)

    def percentile(p):
        if not latencies:
            return None
        return latencies[max(0, math.ceil(p * len(latencies)) - 1)]

    return {
        "requests": len(rows),
        "correct": correct,
        "errors": sum(bool(r.get("error")) for r in rows),
        "accuracy": correct / len(binary) if binary else None,
        "binary_requests": len(binary),
        "p50_seconds": percentile(0.50),
        "p95_seconds": percentile(0.95),
        "elapsed_seconds": elapsed,
        "correct_requests_per_hour": correct * 3600 / elapsed
        if elapsed > 0 and correct is not None
        else None,
        "throughput_scope": "elapsed workload" if elapsed_seconds is not None else "serial requests only",
    }


def paired_group_interval(rows, iterations=2000, seed=0):
    """Bootstrap complete report/tree groups; rows are paired method differences.

    Run independently per training seed and report that separation; repetitions do
    not become artificial independent training replicates.
    """
    if not 1 <= iterations <= 100000:
        raise ValueError("bootstrap repetitions must be bounded")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["difference"])
    if len(groups) < 2:
        return {
            "mean": statistics.mean(row["difference"] for row in rows) if rows else None,
            "ci95": None,
            "reason": "at least two independent groups required",
        }
    values, rng = [sorted(groups[key]) for key in sorted(groups)], random.Random(seed)
    estimates = []
    for _ in range(iterations):
        selected = [rng.choice(values) for _ in values]
        estimates.append(statistics.mean(x for group in selected for x in group))
    estimates.sort()
    return {
        "mean": statistics.mean(x for group in values for x in group),
        "ci95": [estimates[int(0.025 * iterations)], estimates[min(iterations - 1, int(0.975 * iterations))]],
        "independent_groups": len(groups),
        "seed": seed,
        "iterations": iterations,
    }
