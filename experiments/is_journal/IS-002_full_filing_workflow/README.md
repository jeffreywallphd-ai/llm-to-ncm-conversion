# IS-002: full filing workflow

How do evidence preparation and retrieval affect the full DocFinQA workflow?

## Protocol

Hypothesis: A table-preserving condition improves evidence/answer outcomes within cost constraints; citation validity alone cannot establish support.

Treatments and controls: AR, FLAT_RELEASED, HIER_SKIP. Match target, tokenizer, thinking
mode (off), precision, prompt/stop policies, resource envelope and optimizations.
Released comparisons and equal-additional-adaptation comparisons are separate.

Inputs: docfinqa; pinned revisions and terms are in
[`data/manifests`](../../../data/manifests). Preserve official splits and group by
report/context or conversation tree. Freeze development selection before opening test.
Do not expose gold, future target features, or oracle routing information to deployment.

Primary outcomes: numerical_accuracy, retrieval_recall, complete_request_seconds. Training seeds: [0]; five randomized
timing trials. The independent unit is a request group, with training seed uncertainty
reported separately. Retain failed and timed-out requests in denominators.

Dependencies: IS-001. CPU checks are portable. Real-model work
requires cached locked weights, measured hardware readiness, a compatible GPU stack,
correctness gates, data/evaluator access, and finite run AND aggregate phase budgets.
Human owner and independent reviewer are unassigned; the actual team must assign them.

## Commands

Run from any directory after installation with `NCM_REPO_ROOT` pointing at the checkout:

```sh
ncm experiment plan --id IS-002 --profile spark
ncm experiment run --id IS-002 --config experiments/is_journal/IS-002_full_filing_workflow/configs/run.yaml --budget configs/budgets/pilot.yaml
ncm results analyze --experiment IS-002
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
Claim reference: `IS-002-C1`; eligibility is selected in the study manifest, never inferred
from file existence or a completed fixture run.

Revision-2 status: later IS grouping, evidence-bundle and workflow repairs are pending. See docs/remediation_report.md and the ACL-005 stage specifications.
