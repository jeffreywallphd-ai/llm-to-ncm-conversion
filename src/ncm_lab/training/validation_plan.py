"""Fixed held-out cutoff indices, separate from head-position and training RNG."""

from ncm_lab.runtime.io import canonical_hash

POLICY = "codec_stratified_coarse_indices_v1"


def codec_validation_plan(dataset, dev, resolutions, max_examples, seed):
    indices = dev[:max_examples]
    count = len(indices)
    coarse = len(resolutions) - 1
    assignments = []
    for i, index in enumerate(indices):
        cutoff = (
            (coarse - 1) // 2
            if count == 1
            else round(i * (coarse - 1) / (count - 1))
            if count < coarse
            else i % coarse
        )
        example = dataset.metadata(index)["example"]
        identity = example["original_id"]
        position_seed = int(canonical_hash([POLICY, seed, identity, index])[:15], 16)
        assignments.append(
            {
                "index": index,
                "example_id": identity,
                "cutoff_index": cutoff,
                "resolution": resolutions[cutoff],
                "head_seed": position_seed,
            }
        )
    coverage = {str(i): sum(row["cutoff_index"] == i for row in assignments) for i in range(coarse)}
    return {
        "policy": POLICY,
        "seed": seed,
        "assignments": assignments,
        "coverage_counts": coverage,
        "complete_coarse_coverage": all(coverage.values()),
        "aggregation": "equal mean of per-example actual 0.5 full normalized MSE + 0.5 planned cutoff normalized MSE + beta sampled KL",
        "head_sampling": "independent stable per-example RNG; one coarse reconstruction per example",
    }
