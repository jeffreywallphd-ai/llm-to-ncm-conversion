"""Nested whole-group subsets, never enlarged with repeated source positions."""
import hashlib
from ncm_lab.runtime.io import canonical_hash


def resolve_ladder(records, tiers, seed=0):
    if any(r["split"] != "train" for r in records) or sorted(set(tiers)) != list(tiers):
        raise ValueError("data ladder needs train-only data and increasing unique tiers")
    groups = {}
    identities = set()
    for row in records:
        identity = row.get("original_id")
        if not identity or identity in identities:
            raise ValueError("data ladder requires unique original source IDs")
        identities.add(identity)
        groups.setdefault(row["group_id"], []).append(row)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    counts = {g: sum(sum(r["response_mask"]) for r in rows) for g, rows in groups.items()}
    available = sum(counts.values())
    results = []
    for tier in tiers:
        if tier <= 0:
            raise ValueError("positive finite ladder tiers required")
        selected, positions = [], 0
        if tier <= available:
            for group in order:
                if positions + counts[group] > tier:
                    break
                selected.extend(groups[group])
                positions += counts[group]
        results.append({"requested_positions": tier, "available_positions": available,
            "admitted_positions": positions, "status": "available" if selected else "unavailable",
            "original_ids": [r["original_id"] for r in selected],
            "subset_hash": canonical_hash(selected) if selected else None,
            "rule": "seeded whole-group prefix bounded above by tier; no repeated fill"})
    return results
