# ACL-002: main quality speed

Does progressive drafting improve measured efficiency with preserved output and task quality?

## Protocol

Hypothesis: Matched trained comparisons improve complete-request latency with parity; slower or inconclusive results are retained.

Treatments and controls: AR, FLAT_RELEASED, FLAT_ADAPTED, HIER_SKIP, FLAT_COST, STRONG_EXTERNAL. Match target, tokenizer, thinking
mode (off), precision, prompt/stop policies, resource envelope and optimizations.
Released comparisons and equal-additional-adaptation comparisons are separate.

Inputs: gsm8k, ifeval, humaneval_plus, longbench; pinned revisions and terms are in
[`data/manifests`](../../../data/manifests). Preserve official splits and group by
report/context or conversation tree. Freeze development selection before opening test.
Do not expose gold, future target features, or oracle routing information to deployment.

Primary outcomes: request_seconds, committed_tokens_per_second, exact_token_agreement, official_task_quality, memory_bytes. Training seeds: [0, 1, 2]; five randomized
timing trials. The independent unit is a request group, with training seed uncertainty
reported separately. Retain failed and timed-out requests in denominators.

Dependencies: ACL-001. CPU checks are portable. Real-model work
requires cached locked weights, measured hardware readiness, a compatible GPU stack,
correctness gates, data/evaluator access, and finite run AND aggregate phase budgets.
Human owner and independent reviewer are unassigned; the actual team must assign them.

## Commands

Run from any directory after installation with `NCM_REPO_ROOT` pointing at the checkout:

```sh
ncm experiment plan --id ACL-002 --profile spark
ncm experiment run --id ACL-002 --config experiments/acl/ACL-002_main_quality_speed/configs/run.yaml --budget configs/budgets/pilot.yaml
ncm results analyze --experiment ACL-002
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
Claim reference: `ACL-002-C1`; eligibility is selected in the study manifest, never inferred
from file existence or a completed fixture run.
