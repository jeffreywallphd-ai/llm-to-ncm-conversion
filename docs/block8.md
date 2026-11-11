# Native hierarchical eight-token block-autoregressive workflow

This workflow trains the same embedded hierarchy used by the native stages as a
next-block generator. The custom mask-token baseline is removed. The command name
`run_block8.py` is retained, but its config, dataset, recovery and export families
are new and reject the former custom model. Historical commits and the old dated
software receipt remain evidence of the superseded implementation.

The architecture and research decisions are recorded in the
[design plan](hierarchical_block8_plan.md). No released SDAR or other external
baseline is integrated or claimed by this workflow.

## Model and generation

Every block uses the canonical eight-position native geometry: all 36 original
Qwen3-8B layers in the 8/16/12 allocation, four learned initial queries, latent
width 384, 32 sequential hierarchy updates and 16 cumulative-code outputs. Repeated
resolutions still execute. Each block starts with fresh queries and zero compact
state, and predicts eight tokens with one logical backbone traversal. This is a
hierarchical parallel next-block predictor, not masked diffusion.

The prompt and previously committed text are causal context. Answer queries see
all valid preceding context and the current hierarchy states. Context cannot see
answer states. Interval-overlap resizing, integer-center RoPE, nonzero gates and
pair-shared projections use the existing shared native implementation. True answer
length, padding validity, current/future gold tokens, teacher states and target
codes never enter the generator's prompt-only forward API.

Deployment caches only causal KVs of actual text. Hierarchy queries read that
cache without modifying it at every layer/scale. After each emitted block, a
separate causal traversal appends its actual text KVs. A predicted EOS trims the
block and ends the answer. Intermediate boundaries add no EOS. Output-token,
block and total-context limits stop with an explicitly censored result and no
fabricated terminator. Initial prefill, hierarchy traversals, physical per-layer
query positions and causal commits are counted separately. Short EOS output does
not skip hierarchy work or imply a measured speedup.

No mask embedding, unmasking schedule, 1/2-token curriculum, drafter, verifier or
AR correction remains. In particular, four nonempty initial hierarchy queries are
incompatible with a 1/2-token block curriculum. Block size stays eight throughout.

## Complete workflow

`--execute` composes these independently budgeted stages:

1. Verify/import approved pinned Tulu sources, unless explicit sealed inputs are supplied.
2. Prepare complete parent answers once, with disk-backed eight-token offset views.
3. Extract a bounded train/dev subset of frozen original-Qwen prediction states.
4. Train and select the shared native continuous multiscale codec.
5. Encode frozen fine-code targets for the complete admitted train/dev pool offline.
6. Train and select the full-backbone embedded hierarchical generator.
7. Evaluate bounded full-answer rollouts on locked original-reference test parents.

Teacher prediction state `P-1+t` supervises answer token `t`. Extracting a complete
parent once and slicing states at block offsets is equivalent to extracting each
block with prompt+preceding answer tokens. Parent states and fine codes are stored
once, without copied context prefixes. Pooling and residuals always use the fixed
eight-position block geometry, including the final partial block's loss-only mask.
Full-answer codes from other canvases are not relabeled or cropped into this family.

The default teacher subset targets 10M train answer positions, requires at least
1M, and admits whole parents with at most `max_answer_tokens-1` overshoot. It retains
at most 256 dev parents. Actual admitted counts and the target/minimum are distinct.
Teacher dtype defaults to BF16. The codec and its frozen vocabulary head use FP32;
the selected encoder builds FP32 raw fine-code targets. Generator auxiliary losses
explicitly cast target codes to generator parameter dtype at the declared boundary.

Codec fitting reuses native reconstruction MSE and frozen-head KL consistency at
full and coarse resolutions. Train-only normalization and all sixteen channel-scale
vectors use fixed parent-balanced subsets, default at most 4,096 block views.
Dev never fits these statistics. Scale floor is 0.001, coordinates remain raw encoder
codes, and progressive residual arithmetic is unchanged. The teacher/head remain
frozen. Selected codec weights are frozen before complete train/dev target building.

Generator training loads only tokenized parents and frozen fine-code files, with
no online teacher or codec. The objective is native same-position token CE plus
0.1 times the mean sixteen-level cumulative-code loss plus 0.1 times shifted AR CE.
The AR term uses a separate causal training traversal and never repairs inference.
Both native and AR traversals use the configured per-layer activation recomputation.
All original backbone and hierarchy parameters train; there is no module-only freeze.

Defaults are 10,000 codec updates (microbatch 4, accumulation 8) and 20,000 generator
updates (microbatch 1, accumulation 16), with separate rate groups and completed-update
linear-warmup/cosine schedules (500/1,000 warmup updates respectively, floor ratio 0.1).
These are declared recipes, not measured optima or a guarantee of convergence.
The tiled FP32 AdamW/moments and private stochastic BF16 write-back policy are reused.

## Data, selection and integrity

Reuse approved source/component/license/language/overlap policies or exact sealed
native tokenized/distilled artifacts. Every view inherits its parent's split.
Whole conversation groups and exact rendered-prompt aliases are checked across all
splits before length filtering; conflicts fail without moving splits. Original
references and human/synthetic/mixed/unknown origins are preserved. Existing sealed
teacher-generated answers can be supplied explicitly; this command does not generate
new sequence-distillation targets or silently change target provenance.

Defaults allow 512 prompt tokens, 1,024 answer positions including genuine EOS and
1,536 total context positions. Answers longer than eight tokens are retained as
multiple blocks. Overlength/incomplete answers are excluded and counted without
truncation. Render once and append token IDs; do not introduce extra assistant turns
or decode/retokenize at boundaries. Count unique parents/positions separately from
views, repeated context processing and training exposures. Broader contamination
beyond explicitly sealed source/overlap inputs remains unknown.

Codec dev selection uses its fixed reconstruction/consistency objective. Generator
selection uses mean fixed-dev-view text CE; it is not AR perplexity. Step-zero weights
are eligible and selected/terminal steps are separate. Locked test parents never enter
teacher features, codec/statistics, target fitting or selection. Final rollouts record
exact token match, EOS/censoring, lengths, work and timing; free-form semantic quality
and speedup remain unmeasured.

## Commands

Run from the repository root in the working CUDA environment. Plan without research
asset acquisition, model allocation, source scanning or training:

```bash
python -u scripts/run_block8.py --plan-only
python -m ncm_lab block8 --plan-only
```

Start the complete hierarchical workflow with verified local pinned assets:

```bash
python -u scripts/run_block8.py --execute
```

For source/data qualification only, stopping before teacher allocation or codec training:

```bash
python -u scripts/run_block8.py --execute --prepare-only
```

Resume that exact printed manifest to continue all remaining stages:

```bash
python -u scripts/run_block8.py --resume /ABS/HIERARCHICAL_BLOCK8/manifest.json --execute
```

If another process holds the writer lock, `run_block8.py --resume ... --execute`
waits for that process to release it. The waiting command continues once
preparation is complete. A watchdog stop record or incomplete preparation after
release ends the wait with an error. Press `Ctrl+C` to cancel a queued resume.

To reuse exact existing approved train/dev and locked evaluation artifacts:

```bash
python -u scripts/run_block8.py --execute \
  --data-manifest /ABS/TRAIN_DEV/manifest.json \
  --evaluation-manifest /ABS/LOCKED_TEST/manifest.json
```

A completed source-stage receipt records the candidate and locked evaluation
manifests as `data` and `evaluation`. These artifacts can supply a fresh
experiment after a preparation failure and implementation update. Use
`--prepare-only` to validate preparation before starting teacher features, then
resume the resulting experiment with its sealed configuration.

Generate with the selected hierarchical export, without teacher or codec allocation:

```bash
python -u scripts/run_block8.py --execute \
  --checkpoint /ABS/SELECTED_HIERARCHICAL_EXPORT/manifest.json \
  --prompt-file /ABS/prompt.txt
```

`--messages-file` accepts the shared complete-conversation renderer. `--config PATH`
creates a new explicit recipe. New runs live under `runs/hierarchical_block8/`.
Old custom `block8_workflow_v1` configs/experiments/exports are incompatible and must
remain on their old commit if needed. No migration or latest-run search is performed.
Keep ongoing native experiments on their original checkout: implementation hashes
change when this package/shared cache support changes. The native single-pass API,
architecture and objectives otherwise remain separate.

## Step progress and remaining-time estimates

Codec and generator print one line after every completed optimizer update. Lines
include completed/target updates, percentage, objective components, gradient norm,
learning rates, and dev selection loss when validation runs. The same information
is appended to the attempt's `metrics.jsonl`; its `progress` field stores the
estimate used in the console.

Every tenth completed update refreshes the average wall time per update and the
remaining duration, shown as days, hours and minutes. The average covers completed
updates observed since training resumed in this attempt, excluding initial model
loading, initial validation and initial checkpointing. It includes validation and
earlier periodic checkpoint overhead during training. Remaining time is the average
multiplied by remaining updates in this training stage. It estimates the requested
update horizon, not convergence or the budget-limited stopping point. Generator
duration is not inferred from codec duration. Before the first calculation the
estimate is pending; intermediate updates repeat the previous average and estimate
unchanged. The terminal update reports zero remaining optimizer updates. A new
attempt starts fresh observations; numerical recovery/RNG state is unaffected.

Illustrative formatting only (these numbers are not measured performance):

```text
block8_generator step 10/20000 (0.05%) text_ce=2.5 latent_loss=0.1 ar_ce=1.8 grad=0.9 lr=1e-07,1e-06 avg_step=60.00s eta=13d 21h 10m estimate_step=10
block8_generator step 11/20000 (0.06%) text_ce=2.4 latent_loss=0.1 ar_ce=1.7 grad=0.8 lr=1.1e-07,1.1e-06 avg_step=60.00s eta=13d 21h 10m estimate_step=10
```

There are no timer-triggered reports, heartbeats, countdown refreshes, extra forward
passes, or background training threads. Long updates, initial validation/export,
and checkpoint writes can be quiet until work completes.

To follow an exact experiment from another terminal:

```bash
python -u scripts/monitor_block8.py --experiment /ABS/EXPERIMENT/manifest.json
```

Use `--once` for a single snapshot. The read-only viewer follows stage transitions,
major preparation milestones, new optimizer-step records and checkpoint counters.
Preparation messages identify source input setup, source-index construction,
publication of qualified train/evaluation data, block-dataset construction,
frozen teacher-feature caching, and frozen target-code caching. They report each
observed phase once per attempt, without row counts, percentages or preparation
ETAs. Artifact creation/sealing supplies these milestones; the viewer never opens
the active SQLite database or scans source records. Older producers do not expose
separate scan/grouping boundaries, so overlap checks, filtering, tokenization,
deduplication and grouping share one source-index phase. Fast phases missed between
polls are not replayed as invented progress. Input checking and finalization are
shown when their artifact boundaries are observed.

The viewer attaches at the latest
complete record in each existing attempt, retains actual restored step numbers,
and keeps attempts separate. It waits for new files and tolerates a final partial
JSONL record. Polling discovers writes but never emits timer-only status. Ctrl-C
stops only this viewer. Its finite read-work cap is 2,592,000 polls and its aggregate
metrics read cap is 256 MiB; no derived files or TensorBoard server are created.

An older running trainer can be followed without editing its checkout or sealed
configuration: invoke this viewer script by its absolute path from the new checkout.
For old records without a stored estimate, the viewer measures only newly observed
steps after attachment and marks `estimate_source=viewer_observations`. Its initial
historical record has a pending estimate. Those estimates can include file-polling,
observer suspension and observation delays; recorded trainer estimates take
precedence whenever present. No implementation-hash exemption or cross-version
training resume is introduced. Do not replace code under an active experiment to
add console output. After updating this viewer, restart only the monitoring
terminal with the same command; the running training process stays on its sealed
implementation.

## Recovery and resources

Codec and generator each have independent sharded model/optimizer/RNG/scheduler
recovery, selected weights, completed-step counters and persistent phase ledgers.
All shards verify before restoration; at-target recovery performs no extra updates.
Retain two own recoveries and their referenced selected exports; publish replacement
manifests before retirement. Failed attempts, pending writes, immutable settings and
aggregate charges remain. Teacher/target cache failures restart their owned stage
under the same retry/work/storage accounting.

The full Block8 launchers use progress supervision by default. They compare durable
work counters, write counters, stage state and recovery pointers every four hours.
An interval without an advance requests a budget stop at the next work check. The
launcher terminates its worker if it remains unresponsive after a 30-second grace
period. A forced stop records the observed manifest snapshot, and the monitor
displays that record. The attempt's open ledger and locks support inspection before
recovery. `resources.max_stall_seconds` configures the observation interval in a new
experiment. The launchers retain elapsed observations, and a run that advances can
continue beyond the former lifetime limits. An explicit `enforce_wall_time_limits`
setting of `true` selects the configured finite wall-clock policy.

Each stage allows three attempts, 500 GB retained per attempt and 1.5 TB per phase.
Cumulative write allowances are 6 TB per attempt and 18 TB per phase, alongside
finite forward, token and position limits. Logical writes include conservative
page accounting.
Source index (32 GB) and selected-cache (4 GB) allowances remain independent.

Admission inventories the actual hierarchy for generator training, the native codec
plus frozen head for codec fitting, and the frozen original teacher/selected codec
for cache building. A single effective Spark memory pool is used, never host+device
summation. Declared activation reserve is 8 GB. Disk checks account for bounded feature
or fine-code caches, or three recovery/three inference-weight versions during commits,
and a 20 GB free reserve. Stage-local models are released and allocator caches cleared
between stages. Baseline admission arithmetic does not establish full-8B runtime fit.

### Raw preprocessing work budgets

The top-level `budget` remains the default for teacher, codec, target, generator
and evaluation work. Source scanning and block-dataset preparation have separate
finite work allowances in resolved `stage_budgets`, visible in `--plan-only`.
These are sealed before execution. Each attempt's `resolved_config.json` records
the actual selected budget and ledger phase ID.

The full source scan charges tokenized prompts and original responses, including
responses later excluded from the block dataset. Source candidates intentionally
retain long original responses; block preparation charges their inspection before
length/EOS exclusion. Neither stage can use the admitted 1,024-position answer
limit as its raw-work bound. The old shared 200-million answer-position cap stopped
the reported scan after 369,049 of 939,343 rows, with 199,999,655 positions charged.

Source work uses declared source rows plus bounded overlap-reference rows. Its
per-row planning allowance is `16 * max_row_chars + 128 * max_messages + 1024`,
covering character expansion/byte tokenization and chat framing/EOS reserves.
Block preparation uses `max_parents * max_row_bytes`: serialized integer token
arrays cannot contain more positions than bytes. Source steps cover the scan and
overlap passes; preparation retains its independent step ceiling. The resulting
token/answer-position caps are at least the base caps, and their aggregate phase
caps cover the declared retry allowance. Larger explicit base caps are preserved.
These conservative work allowances are not corpus-size or runtime predictions.

Actual work stays fully charged under the row, payload, index, write and storage
caps and the selected clock policy. The versioned `source_work_budget` and
`preparation_work_budget` records explain the resolved allowances. A larger
preprocessing allowance does not request
additional training updates or admit overlength answers into the generator.

After this correction, start a **fresh** experiment with `--execute`. Do not edit
the stopped run's configuration or ledgers, or resume it under the changed code.
Its incomplete SQLite index is retained for diagnosis and is not a qualified
dataset. The [validation receipt](block8_source_budget_validation_2026-11-10.json)
records reproduction with small pinned Parquet fixtures and preservation checks.

The design is implemented; real full-source completion, corpus sufficiency, full-8B
Spark gradients/optimizer/recovery peaks, convergence, output quality and latency
require separate measured evidence. Fixtures cannot support those claims.

## Validation receipt

The complete offline suite passed with **542 passed, three optional CUDA skips**
in 738.06 seconds. Both separate tiny CUDA tests passed, including the full frozen
teacher/codec/targets/hierarchical-generator pipeline. Focused CPU tests passed
(28), including exact codec optimizer/RNG and FP32/BF16 generator recovery.
Lint, offline plan, wheel packaging and 36 shell-block syntax checks passed.
See the [dated receipt](hierarchical_block8_validation_2026-11-09.json) for hashes,
resolved development failures, declared settings and limits of the evidence.
