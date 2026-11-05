# Native implementation status

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
