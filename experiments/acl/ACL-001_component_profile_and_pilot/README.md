# ACL-001: component profile and pilot

Do physical query reductions leave a credible learning/runtime opportunity?

## Protocol

Hypothesis: Cycle time / expected committed tokens improves on the best matched flat development control; no runtime opportunity and no quality benefit stops escalation.

Treatments and controls: FLAT_ADAPTED, HIER_SKIP, HIER_NO_SKIP. Match target, tokenizer, thinking
mode (off), precision, prompt/stop policies, resource envelope and optimizations.
Released comparisons and equal-additional-adaptation comparisons are separate.

Inputs: oasst1, general_dev; pinned revisions and terms are in
[`data/manifests`](../../../data/manifests). Preserve official splits and group by
report/context or conversation tree. Freeze development selection before opening test.
Do not expose gold, future target features, or oracle routing information to deployment.

Primary outcomes: cycle_seconds_per_expected_committed_token. Training seeds: [0]; five randomized
timing trials. The independent unit is a request group, with training seed uncertainty
reported separately. Retain failed and timed-out requests in denominators.

Dependencies: SHR-003. CPU checks are portable. Real-model work
requires cached locked weights, measured hardware readiness, a compatible GPU stack,
correctness gates, data/evaluator access, and finite run AND aggregate phase budgets.
Human owner and independent reviewer are unassigned; the actual team must assign them.

## Commands

Run from any directory after installation with `NCM_REPO_ROOT` pointing at the checkout:

```sh
ncm experiment plan --id ACL-001 --profile spark
ncm experiment run --id ACL-001 --config experiments/acl/ACL-001_component_profile_and_pilot/configs/run.yaml --budget configs/budgets/pilot.yaml
ncm results analyze --experiment ACL-001
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
Claim reference: `ACL-001-C1`; eligibility is selected in the study manifest, never inferred
from file existence or a completed fixture run.

Revision-2 status: shared factory, bounded stages, incremental policy/verifier and codec/transfer controls are implemented; released research execution is pending. See docs/remediation_report.md and the ACL-005 stage specifications.

2026-10-28 wiring: `configs/matrix.yaml` selects this experiment's actual bounded study jobs and reused dependency closure. The full study is `experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml`; it requires measured calibration. Raw runs retain configurations, input/dependency receipts and failures. `ncm results analyze --experiment ACL-001` produces per-seed observation analysis and codec/cost tables; claim eligibility and explicit selection are separate. CPU fixtures remain ineligible. See `docs/remediation_report.md` for executed validation and external prerequisites.
