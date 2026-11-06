"""Fixed development-only review examples spanning source/length buckets."""

from collections import defaultdict, deque
from ncm_lab.runtime.io import canonical_hash


def fixed_prompts(dataset, indices, count, seed):
    buckets = defaultdict(list)
    for index in indices:
        e = dataset.metadata(index)["example"]
        length = len(e["answer_ids"])
        band = 0 if length <= 64 else 1 if length <= 128 else 2
        buckets[(band, e["provenance"].get("component_source", "unknown"))].append(e)
    queues = [
        deque(sorted(rows, key=lambda e: canonical_hash([seed, e["original_id"]])))
        for _, rows in sorted(buckets.items())
    ]
    selected = []
    while queues and len(selected) < count:
        for queue in queues[:]:
            e = queue.popleft()
            selected.append(
                dict(
                    id=e["original_id"],
                    prompt_ids=e["prompt_ids"],
                    group_id=e["group_id"],
                    reference=e["provenance"].get("reference_text"),
                    target_origin=e["provenance"].get("target_origin", e["source_type"]),
                    example_sha256=canonical_hash(e),
                )
            )
            if not queue:
                queues.remove(queue)
            if len(selected) == count:
                break
    return selected
