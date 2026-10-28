# First-paper corrections V3 â€” 2026-10-28

This pass starts at `8452835bf87f5caa8578c3fc9ba1b63e6f63ebbe` and implements
CORR-01 through CORR-14 on `codex/first-paper-corrections-v3`. The user requested
implementation, testing, a commit, and a local fast-forward of main. No research
downloads, released-model training, sweeps, remote provisioning, or submissions
are part of this pass. CPU fixture fitting is bounded software validation.

The unchanged target, physical future-query schedules, original low-rank skip,
response-only supervision, frozen codec transfer, incremental verification, and
rollback architecture are preserved. GPL-3.0 and vendored notices are preserved.

## Corrections and evidence

The [machine-readable checklist](corrections_v3_checklist.json) maps each correction
to source and acceptance checks. The main changes are:

- Planning and runtime share executor validation. Evaluation requires its actual
  output instruction and official scorer identity; warmup and requests share the
  prompt constructor. Static validation never loads a model or tokenizer.
- Observation, treatment, and comparison identities are separate. Analysis takes
  explicit run IDs or immutable matrix role bindings. It rejects duplicate
  observations, keeps tiers/seeds/scopes/checkpoints distinct, reports unmatched
  requests and failures, and sorts groups before paired bootstrap resampling.
  Different checkpoint hashes are expected across contrast roles. There is no
  automatic historical-run inclusion or fastest-retry selection.
- A compact declaration generates complete fit, development selection, parity,
  and final paths for the decisive contrasts. Claim coverage follows named
  contrasts rather than a job count. Main four-cell comparisons have no original
  skip; the practical original-skip comparison is separate. Runtime controls use
  matching skip settings and include capacity-only, fitted codec/CE-only,
  auxiliary-only, reconstruction-only, linear, independent, and single-scale.
- Reconstruction-only selects exclusively coefficient zero; prediction-aware
  selects exclusively strictly positive coefficients. Mixed searches are labeled
  adaptive selection. Objective and selected coefficient follow weights into the
  student and analysis provenance.
- The data ladder selects whole source groups at 0.25M/1M by default; 4M/16M are
  opt-in. Counts are distinct valid response labels after the known response
  anchor, including eligible EOS. Statistics and every fitted component use the
  admitted subset. Larger-codec reuse requires a separately named fixed-codec
  transfer experiment with disclosed preparation costs.
- Equal-total-adaptation-wall paths charge measured source preprocessing, teacher
  generation, codec search, cache preparation, and prior student attempts once per
  immutable dependency. Missing preparation costs cannot become zero. Released
  pretraining, artifact retrieval, and benchmark scoring are disclosed exclusions.
  An infeasible cap is recorded before model construction. Budget exhaustion may
  select a previously completed development endpoint only under the declared
  policy; cancellation status remains intact. User interruption is ineligible.
- Dedicated codec evaluation contributes versioned rate/distortion rows, separate
  from development selection. Oracle and predicted reconstruction errors use the
  same normalized target-state units and positions. Learning-curve points use
  contemporaneous cumulative costs; selected-procedure cost includes discarded
  later work and remains a separate quantity. Unknown coordinates stay unknown.
- Independent codec architecture version 3 applies each branch's nonlinear
  encoder to all fifteen positions before validity-weighted pooling. Its branches
  do not consume other codes. Legacy versions retain their original semantics.
  The declared comparison tests codec designs; depth/sharing differences preclude
  a pure parameter-sharing or pure-nesting causal claim.
- Profiling resolves each treatment's actual width and selected codec. Conflicting
  scientific options are rejected rather than stripped. Artificial component
  timings remain diagnostics.
- Checkpoints publish one atomic bundle containing weights/state and its internal
  receipt. Fault injection covers each publication boundary, with and without a
  previous version. Recovery resolves the prior complete bundle or the new one.
  Temporary files are cleaned; tampering does not trigger arbitrary fallback.
- Teacher records and codecs carry source contracts for target/tokenizer pins,
  canonical policy/template, thinking mode, state/horizon, and response boundary.
  Admission occurs before large model construction or optimizer steps. Locations
  and supported loading precision are separate from semantic identity.
- Endpoints distinguish GSM8K exact match, IFEval prompt/instruction strict
  accuracy, and continuous LongBench ROUGE-L. Instruction counts are weighted;
  continuous scores have no binary correct-request throughput. Missing metrics
  retain an explicit scoring failure, distinct from measured zero.
- ACL-004 partitions shared ACL-002 LongBench requests into predeclared input-token
  strata, with denominators and uncertainty. It introduces no independently
  trained or repeated model condition.

## Study and claim coverage

Use the single [study source](../experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml)
and [contrast declaration](../publications/acl/contrasts.yaml). Experiment matrices
are selectors over that source, including all required dependency/consumer paths.
The default tiers are 250,000 and 1,000,000; optional tiers are 4M and 16M.
Independent codec/student pairs are 17/17, 29/29, and 43/43; five timing blocks are
nested within each pair. AR/released references are explicitly shared.

The default materialization contains 2,559 jobs and 174 locked contrast instances.
All four tiers materialize 5,049 jobs. These counts describe finite software plans,
not successful experiments. All eight experiment selectors validate. Coverage also
includes 42 held-out codec diagnostics, 72 held-out causal-diagnostic consumers,
and one shared-observation robustness stage. The calibration guard blocks the full
study before launching any research job in the available environment.

Each declared final contrast has selection and parity prerequisites and covers
the complete designated official split, subject to explicit finite run limits.
Main released-reference comparisons designate GSM8K, IFEval, original LongBench
GovReport, and HumanEval+. Focused mechanistic controls designate GSM8K. A cap that
prevents full coverage cannot satisfy a full-test gate. Representation and causal
diagnostics have held-out consumers distinct from selection. ACL-004 reuses main
LongBench evidence. No claim has selected research evidence.

```sh
ncm matrix plan --spec experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml --output results/generated/study-v3-plan.json
ncm matrix plan --spec experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml --tiers 250000 1000000 4000000 16000000 --output results/generated/study-v3-all-tiers.json
```

Plan files report structural coverage separately from missing assets, calibration,
and research evidence. They are declarations, not permission to run the study.
Per-run caps and their conservative aggregate phase reservations are finite.
The stage envelope is 100 attempted steps, 40M processed positions, 1,800 seconds,
16 GB; evaluation is 10,000 steps, 50M positions, 1,800 seconds, 256 MB. These are
candidate caps, not a calibrated claim that the tiers can be learned within them.

After an authorized matrix execution, retain its exact output path:

```sh
ncm results analyze --experiment ACL-003 --bindings MATRIX_OUTPUT/analysis-bindings.json
ncm results analyze --experiment ACL-005 --runs EXACT_RUN_ID_1 EXACT_RUN_ID_2
ncm results attest --run EXACT_RUN_ID --claim-class full_test --analysis EXACT_ANALYSIS_MANIFEST
```

The first form is locked contrast analysis; the second is explicit exploratory
analysis, which cannot satisfy a final performance gate. Historical discovery uses
`ncm results index`; it never chooses scientific inputs. Evidence selection is a
separate explicit operation, and fixtures cannot be selected. Performance and
full-test attestations are both needed when a claim requires both classes.

## Compatibility and recovery

New codec selections are schema 3; corrected independent codec weights use
architecture version 3. Existing nested designs retain version 2 semantics. Old
independent weights are never relabeled as the corrected control. Atomic bundle
checkpoints retain `.pt` artifact paths. Complete legacy checkpoint/receipt pairs
remain readable, with their original digest checks.

Uncontracted legacy teacher/codec artifacts fail closed in research. Migration
requires the original immutable completed run, its recorded target/policy and
content hashes; current settings cannot manufacture missing provenance. Derived
migration receipts leave original bytes unchanged. Explicit diagnostic fixture
migration remains ineligible for research:

```sh
ncm data migrate-teacher --path ORIGINAL_TEACHER_RUN/teacher.jsonl --output DERIVED_CONTRACT.json
ncm checkpoint migrate-codec --selection ORIGINAL_CODEC_RUN/codec-selection.json --teacher-contract DERIVED_CONTRACT.json --output DERIVED_CODEC_SELECTION.json
```

## Verification and retained limitations

See [validation receipt](corrections_v3_validation.json) for the exact executed
commands, versions, final counts, installed-package checks, plan coverage, and
artifact hashes. Run artifacts remain unchanged; available temporary fixture failures are also
archived under `.cache/corrections-v3-validation/failures`. The receipt explains
the defects they exposed and the corrective checks.

Implemented and CPU validated are software states. Released-model/GPU validation
and research measurement remain pending. The available environment has CPU-only
PyTorch; it has no DGX Spark, cached Qwen3-8B weights, provisioned official scorer
stack, or verified HumanEval+ isolation. No speedup, quality gain, novelty,
generalization, financial benefit, or publication readiness is established.
IS-only FIX-13/14 remain explicitly deferred; this pass does not repair or claim
their financial grouping, workpaper, and serving workflows.

## Next authorized Spark pilot (pending)

First provision the pinned dependency environment and compatible CUDA PyTorch
under [Spark setup](spark_setup.md); preserve its working torch build. Check the
finite acquisition preview before explicitly downloading assets:

```sh
ncm doctor --profile spark
ncm prepare --profile spark --assets-root /ABS/ASSETS --dry-run
ncm prepare --profile spark --assets-root /ABS/ASSETS
```

Preparation prints an immutable launch directory, `LAUNCH`. It emits finite
teacher, codec, student, and verification budgets and a connected pilot matrix:

```sh
ncm matrix plan --spec LAUNCH/codec_pilot_matrix.yaml --output results/generated/spark-pilot-plan.json
ncm matrix run --plan results/generated/spark-pilot-plan.json
```

That matrix checks released-flat parity, train/dev teacher generation, positive
prediction-aware codec fitting/selection, causal transfer with resume-capable
checkpoints, and adapted parity. Verification runs real speculative and AR
generation, recording complete proposal/verification overhead. Resume explicitly
from the returned checkpoint with the same immutable source/model/policy contract;
do not edit an original run. Use development evidence to set affordable full-study
caps and freeze the protocol before final outcomes. Capture placement/dtype,
peak memory, storage, actual processed/distinct labels, teacher/codec/student
wall costs, accepted prefixes, and complete generation/request timing. These
quantities remain unmeasured on released models. Do not run the final study to
complete software installation or to fill a readiness checklist.
