# Stage 1 single-pass experiment

Use the root README section **Stage 1 single-pass experiment — run everything**.
Its single Bash block creates one unique experiment and executes prepare, teacher,
codec, targets, generator and selected evaluation in foreground sequence. The
subshell stops after a nonzero result without closing/changing the interactive
parent shell. No environment activation, tmux/pager, background training, asset
download, browser, chatbot or TensorBoard server is started by these scripts.

The baseline is proposed, not a validated optimum: medium/64 including EOS;
2,000 train / 200 dev / 200 locked test pairs; codec 1,000 updates (FP32, physical
batch 4, accumulation 4, constant 1e-4); generator 500 updates (BF16, batch 1,
accumulation 8, pretrained/new rates 1e-5/1e-4). The pinned Qwen3-8B, four queries,
width 384, 36 original blocks and all 32 refinements remain. No module-only warmup,
LoRA, layer-freezing workaround, autoregressive continuation or output repair is
introduced. Smaller canvases retain the backbone optimizer memory requirement.

Independent stage files live under `configs/stage1/`. Teacher extraction remains
BF16 and exits before FP32 codec fitting, which loads only the frozen vocabulary
head using the bounded loader. Targets are encoded under the selected codec's
recorded FP32 policy and stored in FP32; their cast to generator dtype at the loss
boundary is explicit in recovery metadata. The optimizer retains tiled FP32 AdamW
arithmetic/moments and private stochastic BF16 write-back, with no FP32 master copy.

## Plan and configure

Run in the existing installed CUDA environment; setup stays separate and must not
replace CUDA Torch with the CPU extra. Pinned local assets are verified during
execution. Missing approved assets print the separate sanctioned acquisition command:
`ncm prepare --config configs/validation/real_pilot.yaml`. These scripts do not call it.

Planning defaults to no execution. Without a directory it prints an offline proposal:

```bash
python scripts/run_stage1.py --plan-only
```

Planning with a new, empty directory saves `manifest.json`, `plan.json` and the
immutable resolved proposal. `--execute` and `--plan-only` are exclusive. Planning
does not hash full weights, scan the corpus, allocate research tensors or charge
training work. Asset presence and unverified quantities are labeled separately.

For the examples below, replace `/ABS/NEW/EXPERIMENT` with a new directory and
`/ABS/EXISTING/EXPERIMENT/manifest.json` with the exact printed manifest. Replace
`/ABS/APPROVED-DATA-CONFIG.json` with a native approved data-stage configuration.

```bash
python scripts/run_stage1.py --canvas-size medium --seed 17 --experiment-dir /ABS/NEW/EXPERIMENT --plan-only
```

```bash
python scripts/run_stage1.py --config configs/stage1/experiment.yaml --data-config /ABS/APPROVED-DATA-CONFIG.json --codec-steps 1000 --generator-steps 500 --experiment-dir /ABS/NEW/EXPERIMENT --plan-only
```

Canvas names/numeric aliases are xsmall/8, small/32, medium/64, large/256,
xlarge/1024. Batch/accumulation, learning settings and precision are typed per
stage. New-run overrides are versioned before execution; existing manifests reject
configuration changes and conflicting geometry/seed. No default replaces stored
settings during recovery. A fresh generator always starts from the pinned original
backbone and recorded hierarchy seed, never a pilot checkpoint.

## Data qualification and locked evaluation

The approved OASST source scan is explicitly enlarged to a bounded 100,000 rows.
The actual file size, inspected rows and cap are reported. A source may still lack
the requested complete short answers, especially at eight positions. Supply an
explicit approved data configuration or use a clearly labeled diagnostic; no
synthetic references, hidden source switch, quota reduction or larger canvas fills
a shortfall. Raw references include genuine terminal EOS; arbitrary truncated
prefixes are excluded. Prompts retain their independent 512-token allowance.

Exact normalized duplicates connect their parent groups before splitting/filtering.
Overlapping group IDs are unioned. Entire connected groups get a deterministic
train/dev/test assignment from split seed 17, independently of the training seed
and canvas. Validation seed 20261029 fixes length-stratified selected IDs and
cutoff/head sampling. Quotas never use repetition. Natural eligible pools and a
common <=8-position subset are reported by IDs; comparisons across differing
eligibility pools do not isolate architecture.

Training/development records retain the existing native schema. Test records use
`native_locked_evaluation_example_v1` in a separate sealed
`native_stage1_evaluation_v1` manifest. Training/teacher/target artifact writers and
loaders reject that locked partition. No relabeling or test-derived normalization,
scales, codec fitting, targets or checkpoint selection occurs. Final test CE is
computed directly against the selected export, without a test codec/teacher.

### Preparation SQLite troubleshooting

The shared OASST reader creates the index parent hierarchy before opening SQLite;
fresh preparation needs no manual directory creation. The intended index is inside
the native preparation run, at
`native_runs/stage1_prepare/<run-id>/artifacts/source-index.sqlite`, not directly
under the experiment manifest or controller attempt. Existing parent directories
are accepted; unusable paths fail without fallback, replacement or index reuse.
Connections close on completion, setup/ingestion errors and reader cancellation.

Directory-creation and connection failures report the operation and intended index
path. The original exception and chained traceback are retained in the native
run's `failure.json`; `stages/prepare/<attempt>/launch.json` identifies that run.
The experiment manifest and partial `reports/stage1_summary` retain failure, the
worker exits nonzero, and dependent stages remain pending without a successful
preparation handoff. Check the actual filesystem error rather than assuming every
SQLite open error means a missing directory.

After installing this fix, retain the reported failed experiment
`runs/stage1/medium/20261104T222631Z-54a2db31` as evidence. Use the existing root
README **Stage 1 single-pass experiment — run everything** combined Bash block to
create a new experiment in the installed CUDA environment with verified local
Qwen/OASST/tokenizer assets. Implementation identity changes can reject the old
manifest; do not bypass those checks. This correction does not establish corpus
capacity, downstream training success, hardware fit or answer quality.

## Codec selection and qualification

Initialization is measured before any update under preserved Torch/Python/training
sampler state. It is not eligible for trained selection. Fixed codec validation
uses 128 dev examples and all 31 coarse level indices (not deduplicated slot counts).
Reduced explicit diagnostics can repeat validation assignments to cover all levels;
the report counts unique examples separately.

The objective remains `0.5 full normalized MSE + 0.5 cutoff normalized MSE +
0.1 KL(teacher || reconstruction)`, temperature one. Head positions are sampled at
most 32 across each microbatch, in chunks of eight; actual counts are logged.
Teacher states and head are frozen. Training cutoffs use the training sampler;
validation cutoffs/head positions use independent fixed seeds. Full-dev diagnostics
also report bounded teacher-head agreement, not answer correctness.

The selected **trained** codec must have finite valid artifacts/normalization/scales,
strictly lower fixed-plan combined dev score than initialization, and no component
regression beyond `atol=1e-7 + rtol=1e-5 * abs(initial_component)`. The default
combined minimum delta is zero; tolerances/policy version and every component delta
are recorded. This is an operational review policy, not significance/convergence or
a universal acceptable MSE/KL threshold. Variance and scale-floor occupancy remain
raw diagnostics without an invented universal cutoff.

Non-improvement seals the completed trained codec and writes `review_required`.
The wrapper returns nonzero and stops the normal Bash block before targets or
generator training. It never trains longer to manufacture a passing result.

For a **finite but non-improving diagnostic only**, explicitly record a reason:

```bash
python scripts/train_stage1_codec.py --experiment /ABS/EXISTING/EXPERIMENT/manifest.json --allow-codec-review --reason "Explicit diagnostic proceeding despite non-improvement; inspect saved component deltas" --execute
```

Then use the ordinary resume command to continue pending work. This never overrides
invalid/nonfinite/corrupt artifacts. The original failed gate remains immutable;
the separate override record makes the resulting experiment a diagnostic.

## Generator selection, reductions and checkpoints

Stage 1 selects trained generator checkpoints by mean per-example text CE on the
fixed 64-example dev subset. Auxiliary and total losses remain reported. Historical
pilots retain combined-loss selection. Initialization CE/samples are separate from
trained selection, so a non-improving trained model still finishes with an honestly
reported selected trained export. Eight fixed dev prompts are retained at step zero
and each 50-update validation; sampling has no clock truncation in Stage 1.

The baseline objective stays `text CE + 0.1 auxiliary loss`. Each physical
microbatch's normalized objective is averaged over the **actual** accumulation
count. Generator batch one gives equal-example normalized losses; codec batch four
pools valid elements inside each microbatch before averaging four microbatch losses.
This is not a global-token average or proof that batch/accumulation rearrangements
are numerically equivalent. Clipping follows accumulation; one optimizer call is
one completed update. Labels remain slot-aligned with no extra AR shift.

Stage 1 validates independently of checkpoint writing. Save at periodic boundaries,
terminal completion and improved trained selections, once if reasons coincide.
`current` recovery and selected export/checkpoint pointers stay distinct. The
existing bounded retention preserves selected/recent checkpoints and interrupted
writes. Selected-state loading reuses the model allocation; it does not allocate a
second full backbone/optimizer.

## Resume, repeat and explicit extension

```bash
python -u scripts/run_stage1.py --resume /ABS/EXISTING/EXPERIMENT/manifest.json --execute
```

The manifest verifies completed receipts and continues only pending work. Training
resumes its own latest complete recorded recovery point; stages before a first
checkpoint restart from recorded initialization and retain prior charges. A target-
boundary checkpoint needs finalization, not repeated updates. Completed explicit
stages return `already_completed` after verification without rewriting artifacts.
No global handoff or arbitrary newest-run scan is used.

```bash
python scripts/train_stage1_codec.py --experiment /ABS/EXISTING/EXPERIMENT/manifest.json --execute
```

```bash
python scripts/train_stage1_generator.py --experiment /ABS/EXISTING/EXPERIMENT/manifest.json --execute
```

These wrappers never run prerequisites. Their optional `--resume-checkpoint PATH`
must identify verified recovery inside that exact experiment. Source, geometry,
codec, precision, reductions, selection, batching and implementation remain strict.
No new legacy checkpoint compatibility bridge is introduced.

The initial generator target remains 500. Before locked evaluation has started, an
explicit operator may extend the **total** target within the predeclared 1,000-step
resource ceiling, preserving input identities, optimizer state and phase charges:

```bash
python scripts/run_stage1.py --experiment /ABS/EXISTING/EXPERIMENT/manifest.json --stage generator --extend-generator-steps 1000 --reason "Explicit total-step continuation before locked evaluation" --execute
```

This adds the remaining work to reach 1,000 total, not 1,000 extra updates. The
original proposal and extension history remain visible. Extension after observing
locked test results is rejected, including after attempting the optional AR reference
before the main evaluation. Additional seed runs (for example 23/41) are
explicit new experiments under the same split/validation seeds, never automatic
replications. Reusing one codec in a separately declared transfer would test only
generator-seed variation, not codec-seed robustness; no such reuse is inferred here.

## Resources, progress and outcomes

Every stage/controller disables execution-clock enforcement while measuring
elapsed time. Cleanup timeouts apply only after cancellation. Non-time work,
context, persistent aggregate charges, storage, memory and integrity checks stay
active. Qualification derives finite work from actual counts, required epochs,
token lengths, batching, all validations, initialization, fixed samples,
normalization/scales and evaluation. A visible 25% work margin and finite retry
allowance do not authorize more training updates. The generator's separately
declared 1,000-step ceiling permits explicit later extension; it is not executed
automatically. Supporting epoch/token caps cover the actual qualified sampler work.

Resource plans use meta-device parameter inventories, distinguish FP32 codec/head
from BF16 teacher/generator plus FP32 moments, and include model cache, feature/code
caches, selected/recent/in-progress checkpoints, export and filesystem headroom.
Baseline state estimates exclude unmeasured activation/kernel peaks. No host/device
capacity summation, cache dropping, swap tuning or privilege changes are used.

Persistent write ledgers count completed checkpoint/cache tensor and manifest bytes
separately from retained storage. Pruning never erases cumulative I/O charges.
Interrupted writes retain conservative reservations with unknown completion instead
of a guessed zero. Worst-case write planning permits a new selection at every
validation boundary. Non-time resource failures stop truthfully.

The active-attempt disk estimate includes shared artifacts and headroom. Preserved
prior attempts consume actual free space and the separate aggregate retained-storage
allowance; each new admission checks current free disk again. It does not promise
that every retry fits simultaneously within the first-attempt estimate.

Console lines identify phase, completed updates, loss components, gradient norm,
exposures, selected step and available host memory. Long phases print an elapsed
heartbeat with log location, not invented progress. Full stdout/stderr, native runs,
failed attempts, snapshots and lineage remain under the experiment. Ctrl+C records
cancellation and stops only owned child groups. An execution lock rejects concurrent
writers; never remove another process's lock or reset a ledger to evade caps.

Final `reports/stage1_summary.md` and `.json` distinguish training completion,
codec qualification, dev text-CE improvement and answer-quality review. Partial
reports are not final success. All fixed-prompt samples and final raw prompt /
reference / prediction triplets remain local with source IDs and lengths. Reports
include all-C computation, EOS/empty/absent/premature-length diagnostics, repetition,
exact-match diagnostics, declared objective rules and end-to-end generation timing.
Semantic relevance/coherence needs human review; exact match is not universal
open-ended correctness and one traversal is not a speedup claim.

Optional original-Qwen AR reference is a separate explicit inference action, excluded
from the main block and default workload:

```bash
python scripts/evaluate_stage1.py --experiment /ABS/EXISTING/EXPERIMENT/manifest.json --ar-reference --ar-max-new-tokens 64 --execute
```

Choose a cap at most the selected canvas (64 here). It uses the same locked prompts,
has separately declared finite AR work charges, never trains/changes selection or
repairs NCM answers, and is labeled AR quality/latency reference, not comparable
parallel likelihood or an automatic speedup measurement.

The isolated small-set diagnostic uses 32 deterministic training pairs, separate
dev/test pools and 200 generator updates at batch one/accumulation four. It retains
the full 8B architecture and optimizer memory footprint:

```bash
python scripts/run_stage1.py --config configs/stage1/overfit.yaml --experiment-dir /ABS/NEW/DIAGNOSTIC --plan-only
```

Execute its stored manifest explicitly. It is not inserted into the baseline, does
not establish generalization, and never initializes the main experiment implicitly.

## Software verification

```bash
python -m pytest -q tests/unit/test_stage1_policy.py tests/unit/test_stage1_data.py tests/integration/test_stage1_workflow.py tests/integration/test_stage1_bash.py
python -m pytest -q
ruff check .
```

The literal README block is extracted for Bash syntax, parser, shell failure/order,
space-containing path and parent-shell isolation tests. A separate test executes
that literal block against the actual tiny production-class backend; its test-only
Python shim adds an explicit fixture config to planning, never a production hidden
environment override. Standard defaults remain 1,000/500. Fixtures cannot establish
real-Qwen fit, convergence, useful answers, hardware recovery feasibility or speed.

The baseline reduction distinction was checked against the primary
[Accelerate accumulation guidance](https://huggingface.co/docs/accelerate/main/usage_guides/gradient_accumulation).
No Accelerate dependency or token-weighting change was added.
