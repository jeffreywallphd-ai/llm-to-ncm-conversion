# Native implementation status

## Block8 preparation vocabulary lookup — 2026-11-11

Preparation caches the pinned tokenizer's full vocabulary size once and uses it
to validate each token ID. The reported attempt reached its wall limit at 6,223
parents with a vocabulary size lookup in the per-token loop. A local check of
the cached tokenizer averaged 16.9 ms per lookup. The parent/offset fixture now
checks the number of vocabulary size calls and continues to verify exact block
offsets and EOS.

The completed source-stage candidate and locked evaluation manifests can be
passed to a fresh experiment after a preparation failure. A preparation-only
experiment using those saved artifacts can be resumed at teacher features after
its dataset is sealed.

Full Block8 CLI launches now use a four-hour inactivity interval by default and
record elapsed time for each attempt. Work, storage, write and retry allowances
remain finite. The watchdog uses durable work and artifact counters and stage or
checkpoint transitions. Cooperative stops close the attempt. Forced stops retain
an inspection record associated with the last manifest snapshot.

Validation: **574 passed, three optional CUDA skips** in the full offline CPU
suite. Ruff and offline planning passed. The preparation-only run inspected all
252,000 saved records and sealed its dataset. The queued resume waited for the
writer to exit and reused the dataset from the original preparation attempt.
The saved failed-run metadata and ledger hashes, and its index size and mtime,
match the pre-fix observations.

## Block8 raw preprocessing budget correction — 2026-11-10

The reported source attempt stopped at 199,999,655 answer positions after
369,049/939,343 rows. Raw source and candidate inspection were sharing a budget
intended for bounded training work. Stage-specific finite source and preparation
allowances now derive from declared raw row/character and parent/byte bounds.
Plans and sealed attempt configurations expose their actual budgets. All raw work,
including later exclusions, stays charged; model-training horizons, clock
enforcement, storage and write bounds remain intact.

The focused fixture reproduces the old source failure and the subsequent
preparation failure, then completes the full same-source scan and block dataset
with overlength references still excluded. Counter replay, hard per-run/aggregate
stops, stage isolation and offline bound validation passed. A fresh experiment is
required; no failed run or ledger is migrated or reset. Full production source
completion, corpus sufficiency and full-8B feasibility remain unmeasured.

Validation: **569 passed, three optional CUDA skips** in the complete offline CPU
suite; **10 focused regressions passed**. Ruff and offline planning passed.
Saved failed metadata/ledger hashes and index size/mtime remained unchanged. See
the [source-budget receipt](block8_source_budget_validation_2026-11-10.json).

## Block8 major preparation messages — 2026-11-09

The read-only viewer reports major source-index/publication, block-dataset,
teacher-feature and target-code cache phases from existing artifact milestones.
It emits only when an observed phase or active attempt changes. No row counters,
preparation percentages/ETAs, live SQLite queries, source scans or training changes
were added. Scanning and grouping remain a combined phase for older producers
without finer markers. Restarting only the viewer enables these messages for an
already-running experiment on its original sealed implementation.

Validation: **19 focused monitoring tests passed; Ruff passed**. These cover
source/dataset/cache transitions, silence despite time/index growth, retry
boundaries, read-only behavior and the existing ten-step estimate behavior. This
monitor-only change did not repeat model-training regressions. See the
[receipt](block8_prep_monitor_validation_2026-11-09.json).

## Block8 step progress — 2026-11-09

Codec/generator console progress follows every completed optimizer step. An
additive JSONL estimate refreshes every ten updates and remains unchanged on
intermediate steps, with days/hours/minutes remaining for that stage. Its average
uses observed completed-update wall cost in the current attempt; it is an estimate
of the configured horizon and cannot establish convergence or runtime feasibility.

A model-free read-only viewer can follow an older running experiment from another
checkout, without modifying its sealed code/configuration. Older records use
explicitly labelled viewer-observed timing; new records carry trainer estimates.
No timer-only reports or scientific/recovery-state changes are introduced.
Real-model ETA accuracy and monitoring overhead remain unmeasured.

Validation accounted for all 560 collected cases across bounded runs: **557
passed and three optional CUDA tests skipped**. Chat/TensorBoard tests passed
with loopback access after restricted-environment failures. Long regression batches
reached their execution caps and continued from the unfinished cases. Fourteen
focused progress unit checks and the actual tiny block8 pipeline passed; Ruff
passed. See the [receipt](block8_progress_validation_2026-11-09.json). The active
checkout still matches the running experiment's sealed implementation hash.

## Stage 2 unattended workflow — 2026-11-05

The Stage 2 launcher composes the six native stages at fixed 64-position
geometry, with 16,000 train / 200 dev / 200 locked test pairs, a 1,000-update
FP32 codec and a 4,000-update BF16 generator. It enforces the existing codec
gate, runs final evaluation, and starts the bounded loopback chatbot on the
verified selected export. It can reuse completed stages via its exact
manifest. The README covers pinned OASST and the explicit bounded Tülu option.

Verification: **466 passed, 1 CUDA-only skip** in the full suite; Ruff passed.
A separate tiny CPU run completed the single launcher through real loopback
chat startup on the selected export and bounded shutdown.

Production data capacity, Spark memory and recovery, runtime, and quality
remain unverified. No research run was started for this implementation.

## Stage 1 console correction — 2026-11-04

Removed the shared child runner's 15-second timer-only status output. Initial stage
settings/log locations, actual worker messages and completed-step reports, outcomes,
final output drain, polling/backoff, cancellation, resource checks and checkpoint
capture are unchanged. See [quiet waiting and recovery limits](stage1.md).

Verification: **459 passed, 1 CUDA-only skip** in the full suite; **44 passed** in
targeted checks; Ruff passed. This includes the previously unexecuted SQLite
regressions and existing monitoring/recovery coverage. The
[dated receipt](timer_status_validation_2026-11-04.json) records exact commands,
source hashes and limits. Only the timer bookkeeping/branch changed in package code;
README bytes and scientific/numerical code are unchanged.

The package source hash changes, so existing experiments/recovery checkpoints still
require their exact matching code/environment. No compatibility bypass, active-run
interruption, research download, real-model training or GPU/Spark validation occurred.

## Stage 1 workflow — 2026-11-04

The separate manifest-bound workflow implements complete-answer qualification,
locked test data, independent BF16 teacher / FP32 codec and targets / BF16 generator
precision, fixed component codec qualification, initialization diagnostics and
versioned dev text-CE generator selection. Stage 1 defaults to medium/64, 1,000
codec updates and 500 generator updates; the existing pilots are unchanged.

Public scripts cover planning, foreground separate-stage/all execution, exact
manifest-owned checkpoint recovery, explicit pre-test total-step extension and
selected-export evaluation. Reports include partial failures/review stops, raw
triplets, exposures, selection versus terminal step, and finite work/storage/I/O
accounting with execution clocks explicitly disabled. Optional AR reference and
32-pair overfit diagnostics remain separate operator actions. See the
[Stage 1 guide](stage1.md) and the explained commands in the root README.

Software verification: **414 passed, 1 CUDA-only skip** in the full suite;
the additional interrupted-target restart check passed separately on the same
implementation. Ruff, the literal README Bash sequence against a tiny backend,
eight shell tests, offline wheel/source parity and model-free entry points passed.
See the [dated receipt](stage1_validation_2026-11-04.json) for exact commands,
source identity, retained failures and limits. These are CPU software fixtures.

**Unexecuted research work:** approved real-data capacity qualification, Qwen3-8B
training, full optimizer/checkpoint recovery on Spark/CUDA, answer-quality review,
convergence and speed measurement. No research model/data download, full-model
training, sweep or persistent server was launched during implementation. Tiny CPU
fixtures cannot resolve these blockers. Real runs also require sufficient cache,
simultaneous checkpoints/export and filesystem headroom; the derived baseline
memory/storage estimates are not a measured training fit.

## Earlier implementation evidence

| Milestone | Implementation | Actual evidence |
|---|---|---|
| 36 original blocks, 32 updates, 8/32/64/256/1024 canvases, width 384 | implemented | Tiny-width all-size geometry/gradient tests and meta shape inspection; no real-8B claim |
| Frozen teacher alignment and shared progressive codec | implemented | Native CPU gradients/alignment/cache checks |
| Sixteen cumulative losses and full-backbone tuning | implemented | Tiny production-class optimizer updates |
| Complete sharded recovery and independent export | implemented | CPU deterministic recovery, corruption and interruption checks |
| Bounded fixed-batch codec/generator learning | implemented | See native_validation.json; fixture only |
| Pinned real-data codec fit | implemented path | Not executed: full Qwen assets/CUDA unavailable |
| Real 8B optimizer/recovery feasibility | implemented probe | Not executed: CUDA/Spark/assets/storage prerequisites |
| Downstream quality, throughput, speed improvement | unestablished | Not measured |

The current receipt and six workflow READMEs identify actual native run IDs and
limitations. This is an executable initial research implementation, not proof that
parallel generation retains the pretrained model's quality. Source/target quotas
can be unavailable. No financial, business, generalization, novelty, or publication
claim is made. Formal studies remain future extensions.

Read-only canonical/TensorBoard monitoring is implemented and CPU validated:
Historical revision `3072d45` passed 78 tests with 1 CUDA skip; actual event/HTTP/browser checks and a
connected installed-wheel smoke completed. See [monitoring validation](monitoring_validation.json).
Histogram collection is disabled/unsupported; no real-8B or matched overhead claim is made.

Current C1–C8 status: source binding, stochastic BF16 numerical recovery, fixed codec
validation coverage, executed-phase labels, replay storage/outcomes, aggregate
device checks and independent inference head policy are implemented. The correction
receipt records current full-suite outcomes and actual BF16 production-class CPU
codec/generator runs. Real-data fitting, full-8B recovery, Spark/CUDA allocation,
quality and speed remain unexecuted/unmeasured. See
[native corrections](native_corrections.md) and its [receipt](native_corrections_validation.json).

Final correction suite: 114 passed, 1 CUDA skip in 80.40 seconds;
installed source bytes match the wheel, smoke completed, and preparation plan started
no downloads. BF16 codec/generator recovery was executed on CPU fixtures.

## Tulu extension implementation

The dedicated pinned multi-shard route, complete-context rendering, disk-backed
source/language/overlap/group/selection policy, capacity gate, sequential wave CLI
and explicit objective completion are implemented. See
[operator guide](extension_training.md) and [dated software validation](tulu3_extension_validation_2026-10-29.json).

Real extension capacity remains **unknown**. No real Tulu scan, 10M-position codec
fit, 100M-token generator wave or Qwen weight download ran during implementation.
Genuine operational blockers remain pinned full assets/qualified data, suitable
validated CUDA/Spark hardware, and sufficient simultaneous cache/checkpoint/free
storage. Full-model fit, retained quality, throughput and speed remain unmeasured.
The historical OASST-only larger envelopes do not establish enough distinct data.

Final extension software verification: 143 passing tests, one CUDA skip, Ruff and
connected monitoring smoke passed. Local two-shard Tülu-shaped waves exercised
qualification, separate stage processes, terminal export, exact same-wave recovery,
retained preprocessing charges, corrupted-output rejection and Windows child-tree
cancellation. These are software fixtures, not measurements of released Tülu capacity.

## Spark memory-accounting correction — 2026-11-02

Reclaim-aware host snapshots, bounded read-only cgroup-v2 headroom, selected-device
capacity operands, explicit unknown/zero diagnostics and compatible metric
projection are implemented. See [implementation/validation](memory_accounting_fix.md).
Synthetic reproduction and safety tests do not establish Spark fit. Actual GB10
training/recovery, activation/allocator peaks and namespace-hidden constraints
remain unverified; this local CPU environment still lacks CUDA and adequate full
checkpoint disk space. Historical blocked runs/handoffs/checkpoints remain intact.

Final software verification: 214 passed, one CUDA skip, Ruff and whitespace checks
passed; [dated receipt](memory_accounting_validation_2026-11-02.json) includes the
preserved initial integration failure and successful corrected rerun.

## Pilot clock policy — 2026-11-03

The shared real pilot disables run and phase clock enforcement while preserving
the two-step generator target, observed elapsed charges, non-time safeguards and
phase spending. Other jobs remain bounded by default. Strict reviewed-source
recovery and at-target finalization are implemented; see
[operator procedure](pilot_clock_policy.md) and [validation receipt](pilot_clock_validation_2026-11-03.json).
The user's reported checkpoint is absent locally, with unknown source identity;
only the recorded predecessor is approved. Genuine tiny pre-edit BF16 checkpoint
continuation is software evidence, not Spark/8B feasibility. CUDA/full-model
validation and the actual user's resume remain unexecuted; current CUDA/disk
blockers and prior evidence remain intact.

Final local software validation: 255 passing tests, one unavailable-CUDA skip;
Ruff and whitespace checks passed. The same source passed normal local Temp,
loopback and owned-child cleanup checks after sandbox-restricted failures.

## Named pilots and local chatbot — 2026-11-03

The 2 training step pilot and 100 step pilot have named configurations/scripts,
separate exports and success-gated chat startup after the training child exits.
Werkzeug provides the bounded local UI, preview without ML dependencies and one
verified export reused across conversations. Reviewed checkpoint continuation from
0c9fdd5 and the immutable historical clock-policy chain are supported explicitly.

Full local verification: 282 passed, one unavailable-CUDA skip; Ruff passed. Tiny
production 2→100 continuation and actual export inference, fixed-prompt samples,
preview/live HTTP, launch failure gates and genuine pre-edit BF16 state continuation
passed. See [operator guide](pilot_chat.md) and [dated receipt](pilot_chat_validation_2026-11-03.json).
The actual user's Spark checkpoint and real-8B 100-step run remain untested here.

## Configurable canvas pilots (2026-11-04)

The five production presets, medium default, shared script/CLI flags, isolated
local per-size preparation, complete-reference qualification, exact receipt reuse,
strict same-size recovery, no-chat mode and canvas-aware export/monitoring are
implemented. See [canvas contracts](configurable_canvas.md) and the root README.
The historical 1024 procedure and Tulu wave retain their explicit geometry.

Real short-answer availability for every preset, full-Qwen all-size optimization,
Spark kernel/memory/recovery peaks and trained output quality remain unverified.
The codec pilot remains two updates and its adequacy is not established. Older
source training checkpoints cannot resume through this scientific implementation
change; historical compatibility receipts remain unchanged. Fresh matching
experiments and existing inference exports are the supported routes.

Final canvas verification: **367 passed, 1 CUDA skip** in 371.68 seconds;
`python -m pytest -q`, `ruff check .`, whitespace checks, six actual model-free
README plans and offline wheel/package-content checks passed. Focused final guards
passed 101 tests. These are CPU software fixtures, including actual C-position
forward/backward/export checks at all five sizes and an 8-position local preparation
chain followed by 98 additional updates to 100 total. See the
[dated evidence receipt](configurable_canvas_validation_2026-11-04.json), which also
retains the initial two development failures and their corrected reruns.

## Stage 1 SQLite preparation correction (2026-11-04)

The shared OASST index now initializes its supplied parent hierarchy before
connecting, reports directory/open errors with path and chained cause, and closes
the connection throughout setup, ingestion and iteration. Stage 1 and legacy
preparation close their owned readers on early termination. SQL, source admission,
grouped partitions, production quotas, scientific settings, budgets and the README
combined block are unchanged. Prior runs were not edited or deleted.

Added offline real-Parquet/SQLite regressions for fresh/existing parents, spaces,
relative/string paths, row bounds and filters, unchanged sources, stale-index
rejection, JSONL independence, connection cleanup, legacy preparation, prepare-only
Stage 1 artifact verification and nonzero filesystem failure isolation.

**Validation outstanding:** the user explicitly requested skipping tests on this
system, which has no configured development Python environment. No dependencies
were installed. The following checks were not executed and no passing test result
is claimed for this correction:

```sh
python -m pytest -q tests/unit/test_oasst_index.py tests/integration/test_stage1_preparation_sqlite.py
python -m pytest -q tests/unit/test_real_data_contracts.py tests/unit/test_stage1_data.py tests/unit/test_stage1_policy.py tests/integration/test_stage1_workflow.py tests/integration/test_stage1_bash.py
python -m pytest -q
ruff check .
```

Full-model/GPU validation remains unexecuted. No research training or downloads
were launched. See [preparation troubleshooting](stage1.md#preparation-sqlite-troubleshooting)
for the fresh-experiment procedure and saved failure locations. This fix alone
does not validate data quotas, downstream stages, feasibility or output quality.

## Stage 3 single-pass workflow — 2026-11-06

The separate Stage 3 launcher/configuration/source-distillation-qualification path
composes the production teacher, codec, target and generator engine. It adds
versioned warmup/cosine scheduling, selected-weight snapshots separate from full
recovery, advisory finite codec quality policy, serial final AR comparison,
manifest resume and selected-export local chat. See [operator guide](stage3.md).

Validation uses tiny production-class software fixtures, local JSONL/Parquet and
synthetic teacher outputs. This does not demonstrate accepted production Tülu
capacity, real teacher answer quality/acceptance, large/256 full-8B backward or
recovery memory fit, useful generation, convergence or speedup. Those are open
hardware/data/research milestones. Prior Stage 2 experiments must continue using
their own existing checkout/artifacts; this implementation does not migrate them.

Logical payload/SQLite write accounting is conservative; it is not a physical
storage-device wear measurement. Broader cross-experiment contamination remains
unverified unless exact prior locked manifests are supplied. All quality warnings
and failed attempts remain in experiment reports rather than being edited away.

Final validation: **498 passed, 1 skipped** in the complete CPU-fixture suite;
the skipped tiny CUDA forward/backward fixture separately **passed**. Final
focused integration/policy checks passed (56 tests), Ruff passed, all 29 README
shell blocks parsed, and the offline wheel build passed. The corrected portable
memory test uses an explicit 8 GiB fixture rather than asserting this host must
fail admission. These are software checks, not full-8B Spark evidence. Exact
commands, scope and resolved failures are recorded in
[the validation receipt](stage3_validation_2026-11-06.json).


## Stage 3 source-index allowance correction — 2026-11-09

Diagnosed the real 4 GB source-index failure using bounded read-only SQLite
queries. The temporary complete-scan index and selected candidate cache now have
separate 32 GB/4 GB allowances, bounded lossless payload compression, SQLite page
limits, early disk admission and structured storage reports. Full-scan group
integrity, source/target policy and training settings are preserved. Earlier
failed runs and immutable configurations remain intact; operators must start a
new experiment under the new implementation/storage policy.

Local Parquet fixtures cover a complete source index larger than the selected
cache, preservation of long references, and hard resource failure before any
model stage. Unit fixtures cover payload integrity/corruption limits, actual
SQLite page enforcement, independent candidate limits and disk admission. These
software checks do not prove the entire pinned release fits the new allowance.
Production full-scan completion, target acceptance, full-8B feasibility, quality
and speed remain unmeasured by this correction.

Validation: the complete offline CPU suite passed (514 passed, one optional CUDA
skip), `ruff check .` passed, and the wheel includes the new module. See the
[source-storage validation receipt](stage3_source_storage_validation_2026-11-09.json).

## Separate block8 workflow — 2026-11-09

Historical entry: this custom baseline was subsequently retired and replaced by
the native hierarchical workflow described below. Its dated receipt applies only
to commit a80b6f5.

Implemented a separate original-Qwen context-causal masked-block decoder, eight-token
rollout/cache commit loop, parent/offset dataset preparation, approved Tulu adapter,
full-backbone denoising plus AR training, dev selection, sharded selected export and
recovery, finite locked-test rollouts, offline planning and CLI. Native single-pass
hierarchy/codec workflows remain separate. See [block8 guide](block8.md).

Local software fixtures exercise full layer gradients, geometry/alignment, real
cache behavior, EOS, split integrity and source/storage failures, plus exact resumed
FP32/BF16 updates. Production 8B/Spark activation and recovery fit, adaptation corpus
sufficiency, output quality, convergence and speed remain open. The operational
prototype horizon is not a validated scientific recipe; no real training/download
was started as part of this implementation.

Final block8 verification: **540 passed, two optional CUDA skips** in the complete
offline suite; the separate tiny BF16 CUDA test passed. Lint, wheel packaging and
35 shell-block syntax checks passed. See the
[block8 validation receipt](block8_validation_2026-11-09.json).

## Native hierarchical block8 replacement — 2026-11-09

The custom masked-block baseline is retired from active code. The same block8
entry point now composes native hierarchy generation and shared codec supervision:
source, parent/offset data, bounded frozen parent teacher states, codec fitting,
frozen fine targets, full-backbone generator fitting and locked full-answer rollouts.
New schemas reject old custom artifacts. Historical evidence remains identifiable.

The implemented model preserves all 36 layers, four initial queries, 32 updates,
width 384 and sixteen cumulative losses per eight-token block. Shared causal-cache
support is checked against uncached traversal at every scale. Teacher/codec stages
are explicit; no target or online model enters generator forward. Recovery, schedule,
phase/write bounds and failed-run preservation are independent for codec/generator.

Real 8B/Spark memory, recovery, data sufficiency, convergence, text quality and speed
remain open measured milestones. See the [plan](hierarchical_block8_plan.md) and
[current operator guide](block8.md). Prior custom-baseline receipts describe the old
commit and do not validate this replacement. No production experiment was run.

Final replacement validation: **542 passed, three optional CUDA skips** in the
complete offline suite; **two tiny CUDA tests passed** separately. Focused CPU
fixtures passed (28), as did lint, wheel, offline plan and 36 shell syntax checks.
The [new receipt](hierarchical_block8_validation_2026-11-09.json) applies to this
hierarchical replacement; the older custom-baseline receipt remains historical.
