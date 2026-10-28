from collections import defaultdict
import statistics
from ncm_lab.evaluation.analysis import summarize, paired_group_interval


def analyze_records(rows):
    groups, paired, shared = defaultdict(list), defaultdict(dict),defaultdict(dict)
    for row in rows:
        method, seed = row.get("method", "unknown"), row.get("training_seed")
        condition = (row.get("dataset"),row.get("comparison","equal_data"),row.get("policy_hash"),row.get('result_origin','unspecified'))
        groups[(method, seed, condition)].append(row)
        if row.get('shared_reference'):
            shared[(row.get('original_id'),row.get('repetition_id'),condition)][method]=row
        else:
            paired[(row.get("original_id"), seed, row.get("repetition_id"),condition)][method] = row
    tables = []
    for (method, seed, condition), records in groups.items():
        summary = summarize(records)
        cycles = [c for r in records for c in (r.get("decoder") or {}).get("cycles", [])]
        measured = [r for r in records if r.get("decoder") is not None]
        official = defaultdict(list)
        for row in records:
            for key, value in row.get("official_scores", {}).items():
                if isinstance(value, (int, float)):
                    official[key].append(value)
        tables.append({"method": method, "training_seed": seed, "condition":list(condition), **summary,
            "official_metrics": {k: {"mean": statistics.mean(v), "denominator": len(v),
                                     "missing": len(records)-len(v)} for k, v in official.items()},
            "accepted_prefix": [c["accepted"] for c in cycles],
            "mean_accepted_prefix": statistics.mean(c["accepted"] for c in cycles) if cycles else None,
            "decoder_missing_requests": len(records)-len(measured),
            "committed_tokens_per_cycle": sum(1+c["accepted"] for c in cycles)/len(cycles) if cycles else None,
            "mean_output_tokens": statistics.mean(r.get("output_tokens", 0) for r in records)})
        if any("instruction_count" in r.get("official_scores",{}) for r in records):
            count = sum(r.get("official_scores",{}).get("instruction_count",0) for r in records)
            tables[-1]["instruction_strict_accuracy"] = sum(r.get("official_scores",{}).get("strict_instruction_correct",0)
                for r in records)/count if count else None
        tables[-1]["failed_requests_in_denominator"] = summary["errors"]
        generation=[r for r in records if r.get('generation_seconds') is not None]
        seconds=sum(r['generation_seconds'] for r in generation)
        tokens=sum(r.get('output_tokens',0) for r in generation)
        tables[-1]['warm_generation_tokens_per_second']=tokens/seconds if seconds>0 else None
        tables[-1]['generation_seconds_per_token']=seconds/tokens if tokens>0 else None
        tables[-1]['generation_missing_requests']=len(records)-len(generation)
        tables[-1]['accepted_prefix_survival']=[{'offset':j,'accepted':sum(c['accepted']>=j for c in cycles),
            'eligible_cycles':sum(c.get('proposed',0)>=j for c in cycles)} for j in range(1,16)]
        for name, values in tables[-1]["official_metrics"].items():
            values["all_requests_mean_with_failures_zero"] = sum(official[name])/len(records) if records else None
    differences = defaultdict(list)
    for (original_id, seed, repetition, condition), original_methods in paired.items():
        methods={**shared.get((original_id,repetition,condition),{}),**original_methods}
        names = sorted(methods)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                a, b = methods[left], methods[right]
                differences[(left, right, seed, condition)].append({"group_id": a["group_id"],
                    "difference": b["request_seconds"]-a["request_seconds"],
                    'shared_reference_reused':bool(a.get('shared_reference') or b.get('shared_reference'))})
    return {"per_seed": tables, "paired_latency_differences": [
        {"left": a, "right": b, "training_seed": seed, "condition":list(condition),
         'shared_reference_reused':any(v['shared_reference_reused'] for v in values),**paired_group_interval(values, iterations=500)}
        for (a, b, seed, condition), values in differences.items()], "weighting": "requests within training seed and condition",
        "scope": "descriptive observations; research claims require separate eligibility and selection"}
