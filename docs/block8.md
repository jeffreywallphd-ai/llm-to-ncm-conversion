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

To reuse exact existing approved train/dev and locked evaluation artifacts:

```bash
python -u scripts/run_block8.py --execute \
  --data-manifest /ABS/TRAIN_DEV/manifest.json \
  --evaluation-manifest /ABS/LOCKED_TEST/manifest.json
```

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

## Recovery and resources

Codec and generator each have independent sharded model/optimizer/RNG/scheduler
recovery, selected weights, completed-step counters and persistent phase ledgers.
All shards verify before restoration; at-target recovery performs no extra updates.
Retain two own recoveries and their referenced selected exports; publish replacement
manifests before retirement. Failed attempts, pending writes, immutable settings and
aggregate charges remain. Teacher/target cache failures restart their owned stage
under the same retry/work/storage accounting.

Wall clocks are enabled by default for this new workflow: 12 hours per stage attempt,
36 hours per stage phase, three attempts maximum, 500 GB retained per attempt and
1.5 TB per stage phase. Cumulative write allowances are 6 TB per attempt / 18 TB per
stage phase, alongside finite forward/token/position limits. These logical-write
allowances include conservative page accounting, not physical device-wear measurements.
Source index (32 GB) and selected-cache (4 GB) allowances remain independent.

Admission inventories the actual hierarchy for generator training, the native codec
plus frozen head for codec fitting, and the frozen original teacher/selected codec
for cache building. A single effective Spark memory pool is used, never host+device
summation. Declared activation reserve is 8 GB. Disk checks account for bounded feature
or fine-code caches, or three recovery/three inference-weight versions during commits,
and a 20 GB free reserve. Stage-local models are released and allocator caches cleared
between stages. Baseline admission arithmetic does not establish full-8B runtime fit.

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
