# ACL-003: hierarchy and skip ablations

Which effects are due to resolution, learned expansion, and the default skip?

## Protocol

Hypothesis: Predeclared paired contrasts isolate useful contributions; a flat/cost-matched win or null contrast limits the mechanism claim.

Treatments and controls: FLAT_ADAPTED, HIER_SKIP, HIER_NO_SKIP, FIXED_BROADCAST, EARLIER, LATER, LATE_NEAR_BUDGET. Match target, tokenizer, thinking
mode (off), precision, prompt/stop policies, resource envelope and optimizations.
Released comparisons and equal-additional-adaptation comparisons are separate.

Inputs: oasst1, general_dev, gsm8k, ifeval; pinned revisions and terms are in
[`data/manifests`](../../../data/manifests). Preserve official splits and group by
report/context or conversation tree. Freeze development selection before opening test.
Do not expose gold, future target features, or oracle routing information to deployment.

Primary outcomes: cycle_seconds_per_expected_committed_token, accepted_prefix_survival. Training seeds: [0, 1, 2]; five randomized
timing trials. The independent unit is a request group, with training seed uncertainty
reported separately. Retain failed and timed-out requests in denominators.

Dependencies: ACL-001. CPU checks are portable. Real-model work
requires cached locked weights, measured hardware readiness, a compatible GPU stack,
correctness gates, data/evaluator access, and finite run AND aggregate phase budgets.
Human owner and independent reviewer are unassigned; the actual team must assign them.

## Commands

Run from any directory after installation with `NCM_REPO_ROOT` pointing at the checkout:

```sh
ncm experiment plan --id ACL-003 --profile spark
ncm experiment run --id ACL-003 --config experiments/acl/ACL-003_hierarchy_and_skip_ablations/configs/run.yaml --budget configs/budgets/pilot.yaml
ncm results analyze --experiment ACL-003
```

The run command records precise unmet prerequisites instead of downloading weights.
Select checkpoints/architecture on development data only. Stop at the first configured
budget cap; no automatic escalation. See [protocol.yaml](protocol.yaml) for contrasts,
selection, exclusions and stopping rules, and [workflow](../../../docs/experiment_workflow.md).

## Results and interpretation

Not run; no empirical findings yet.

CPU software validation is separate from research results. Real-model/optimized engine
and GPU parity remain unverified. See [result index](results/README.md). An eager reference
cannot establish deployment speed. A null result or unsupported hypothesis remains in
the run history. No native NCM, semantic-concept, or universal hardware claim is implied.
Claim reference: `ACL-003-C1`; eligibility is selected in the study manifest, never inferred
from file existence or a completed fixture run.
