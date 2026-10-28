# Finite compute and storage plan

CPU, single-Spark and single-CUDA profiles are separate. Doctor reports actual
configuration, installed backends, disk and cached model availability, with unknown
values recorded as null and a reason. Unified memory is never summed twice.

Planning is offline. `ncm experiment plan` shows the selected methods, seeds,
provisional source-token matrix cap, timing repeats, dependencies and unknown costs.
It does not infer runtime from peak throughput. A final trained run's provisional
30M-source-token cap is not a study budget: three trained variants times three seeds
would already be 270M before pilots, ablations, teacher generation and evaluations.
The actual final cap must follow measured learning curves and then be locked.

The initial pilot phase reserves three methods x <=1M source-token passes,
<=100 optimizer steps/method and <=1800 seconds/method. Phase reservations are
exclusive/append-only and conservative: failed/blocked reservations are not silently
refunded. New phase IDs/caps are deliberate configuration changes. Step/time/token/
storage checks happen at operation boundaries; an individual GPU kernel or target
forward cannot be interrupted mid-kernel. Checkpoints support clean recovery.

Account separately for acquisition/preparation, teacher generation, frozen-target
feature extraction, drafter training, repeated evaluation, and checkpoint/cache
storage. Online extraction is default, and its target pass is inside training timing.
Record unique source tokens, repeated source-token passes, generated tokens,
anchors, supervised future positions, optimizer steps and elapsed time separately.
One anchor supervises 15 positions; many anchors multiply work, not unique data.

Five BF16 feature taps at width 4096 cost 40,960 bytes/token before metadata.
Do not build an unbounded full-feature cache. The shipped smoke uses no downloads;
data fetch requires an explicit positive byte cap. Setup fetched only 48 KiB of
training JSON prefixes, not multi-GB filings or model weights. Every fetched full
release writes a revision/size/SHA256 receipt; expected counts still need auditing.

No remote resources, paid compute or large sweeps are provisioned by install/plan.
The present host's RTX 4060 is not a substitute for a verified Spark Qwen3-8B run.

## Revision-2 implementation update — 2026-10-27

Current candidate study caps are enumerated in ACL-005/configs/study.yaml and stage budgets. They are conservative finite reservations, not estimates of required research compute. Calibration is required before study launch. Distinct corpus positions, repeated passes, physical target positions, supervised positions, codec search and teacher costs are distinct. Fixed-codec and complete-procedure curves require separate identities. See remediation_report.md for the meta-device storage calculation; no released-model runtime was measured.
