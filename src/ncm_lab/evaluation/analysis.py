from collections import defaultdict
import math
import random
import statistics


def summarize(records, elapsed_seconds=None):
    rows = list(records)
    latencies = sorted(r["request_seconds"] for r in rows)
    if any(not math.isfinite(t) or t < 0 for t in latencies):
        raise ValueError("invalid request latency")
    correct = sum(r.get("correct") is True and not r.get("error") for r in rows)
    elapsed = elapsed_seconds if elapsed_seconds is not None else sum(latencies)
    def percentile(p):
        if not latencies:
            return None
        return latencies[max(0, math.ceil(p * len(latencies)) - 1)]
    return {"requests": len(rows), "correct": correct, "errors": sum(bool(r.get("error")) for r in rows),
            "accuracy": correct / len(rows) if rows else None, "p50_seconds": percentile(.50),
            "p95_seconds": percentile(.95), "elapsed_seconds": elapsed,
            "correct_requests_per_hour": correct * 3600 / elapsed if elapsed > 0 else None,
            "throughput_scope": "elapsed workload" if elapsed_seconds is not None else "serial requests only"}


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
        return {"mean": None, "ci95": None, "reason": "at least two independent groups required"}
    values, rng = list(groups.values()), random.Random(seed)
    estimates = []
    for _ in range(iterations):
        selected = [rng.choice(values) for _ in values]
        estimates.append(statistics.mean(x for group in selected for x in group))
    estimates.sort()
    return {"mean": statistics.mean(x for group in values for x in group),
            "ci95": [estimates[int(.025 * iterations)], estimates[min(iterations - 1, int(.975 * iterations))]],
            "independent_groups": len(groups), "seed": seed, "iterations": iterations}
