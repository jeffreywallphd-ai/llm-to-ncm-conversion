# Eight-token block-autoregressive workflow

This is a separate experimental model family, `qwen_context_causal_block8_v1`.
It initializes the original pinned Qwen3-8B text backbone, embeddings, final norm
and vocabulary head, and adds one learned mask embedding. It does not instantiate
the native embedded hierarchy, a codec, an online teacher, or an AR verifier.
The existing single-pass workflows keep their architecture and objectives.

This implementation is a software milestone. Full-8B Spark training/recovery,
adaptation-data sufficiency, useful output quality and latency remain unmeasured.
The 1,000-update default is an operational prototype, not a validated research
recipe. Small blocks do not shrink the backbone, gradients or FP32 Adam moments.

## Architecture and objectives

The prompt and all committed answer tokens form a strictly causal context.
Only the active block attends bidirectionally. At deployment the active block
always has eight token positions, independent of the unknown remaining length.
Every layer retains its original Qwen attention/MLP/norm parameters. RoPE uses
contiguous token positions offset by the complete causal context length.

Generation begins with eight learned mask embeddings. Four bounded refinement
passes greedily fill the most confident remaining positions, breaking confidence
ties by position. Already revealed tokens remain fixed. This is confidence-ordered
masked decoding, not a claim to an exact diffusion likelihood or joint AR sampling
distribution. A block is committed only after all eight positions are filled.
The first genuine EOS trims the block; no further blocks are generated.
Output-token and block-count stops are explicitly censored, without fabricated EOS.

Persistent KV state contains only actual committed text. Active passes read from
the prefix cache without changing it. After each block, its final token embeddings
are processed once under ordinary causal attention to append their KVs. Both this
commit traversal and the initial prompt prefill are included in work/timing counts.
Four refinements plus one commit are five backbone calls per full eight-token
block; this is not an eightfold speedup. Actual attention kernels, batch size,
context length and device behavior determine wall time.

The separate `denoise(context_ids, block_input_ids, prefix_cache=...)` interface
accepts explicit denoising inputs, with `-1` meaning the learned mask. It accepts
no labels, teacher features, codec codes or true-length masks. Training-only
revealed current-block tokens are deliberate corruption inputs; preceding gold
blocks are teacher-forced text context. Unrevealed labels and all later blocks
remain outside model inputs. The old native prompt-only API is unchanged.

Training uses one unpadded example per microbatch and configurable accumulation.
Each microbatch samples a parent answer uniformly, then a block within that answer;
long answers do not receive proportionally more parent selections. Half of active
blocks are fully masked. Otherwise a uniformly sampled nonempty subset of valid
target positions is masked; padding remains masked and receives no loss. The
objective is mean masked-token CE plus 0.5 times ordinary shifted next-token CE.
Denoising logits align to the active token at the same position. AR states at
`context_length - 1 + t` predict target token `t`. There is no latent-state loss.

All backbone parameters train from the first update. The default active sizes
are 1/2/4/8 through completed-update endpoints 50/100/200/1000. Inference and
held-out selection always use eight positions. Learning rates are explicitly
constant, with a separate rate for the new mask vector. The existing tiled FP32
AdamW/moments and private stochastic BF16 write-back policy are reused unchanged.
Activation checkpointing and chunked vocabulary CE are enabled for the real model.

This first workflow uses sampled-block training with a denoising traversal and a
separate causal traversal for the AR term. It does not implement the optional
concatenated clean/noisy all-block training optimization or on-policy teacher
continuation generation. Repeated prefix computation is charged. These are future
training-efficiency/data extensions, not prerequisites for the implemented block
conditional. Explicit existing distilled targets can already be imported.

The context-causal adaptation direction is informed by
[From Next-Token to Next-Block](https://arxiv.org/abs/2512.06776) and
[Block Diffusion](https://arxiv.org/abs/2503.09573). This code is an independent,
bounded prototype, not a reproduction of their complete recipes or results.

## Dataset preparation

Default preparation composes the approved pinned Tülu source adapter and its
language, component/license, overlap and full-scan grouping policies. It imports
released original references, with their recorded human/synthetic/mixed/unknown
origins; it does not generate new teacher answers. Source SQLite and selected-cache
limits stay separate. Source policy/lock receipts and resolved adapter settings are
sealed in the new experiment. No asset downloads happen automatically.

Existing tokenized `native_data_v1`, `stage3_candidates_v1` or `stage3_distilled_v1`
artifacts can instead be supplied explicitly, together with a locked evaluation
artifact. Model/tokenizer/template identities must match. Explicit distilled
inputs retain their `teacher_generated` origin. Fixtures cannot become real data.

`block8_parent_offset_data_v1` stores each complete parent example once as bounded
lossless compressed JSON in SQLite. A separate table contains only parent ID,
answer-token start offset and split. Prefixes are formed by appending preceding
answer IDs to the original rendered prompt IDs; there is no retokenization or
extra assistant-turn delimiter at block boundaries. Views at 1/2/4-token curriculum
sizes are derived from the same parent without new source examples.

Whole conversation groups and exact rendered-prompt aliases are checked across
all supplied splits before length filtering. Conflicts fail; splits never move.
The adapter's full grouping happens before bounded source selection. Broader
contamination beyond the explicit sealed source/overlap inputs remains unknown.

Default bounds are 512 prompt tokens, 1,024 answer tokens including genuine EOS,
and 1,536 total context positions. Answers longer than one eight-token block are
retained. Overlength/incomplete answers are excluded and counted, without truncation
or artificial terminators. Intermediate blocks contain no EOS. Final padding is
loss-only; neither its mask nor the true answer length becomes a generator input.

Capacity reports distinguish inspected/selected parent answers, source origins,
unique answer/prompt token positions, eight-token views and exclusions. Repeated
prefix processing and epochs do not create additional unique answers. The default
minimum is 90,000 train / 2,000 dev / 2,000 locked test parents. Capacity failure
precedes model allocation. SQLite page caps and row/view/write limits preserve
failed indexes and diagnostics; no partial dataset is published as complete.

## Commands

Run from the repository root in the working CUDA environment. Planning requires
no model/source acquisition or full-model allocation:

```bash
python -u scripts/run_block8.py --plan-only
python -m ncm_lab block8 --plan-only
```

Explicitly prepare only, or execute source preparation, block preparation,
full-backbone training, held-out selection and bounded locked-test rollouts:

```bash
python -u scripts/run_block8.py --execute --prepare-only
python -u scripts/run_block8.py --execute
```

Keep the printed absolute experiment manifest. Resume that exact experiment:

```bash
python -u scripts/run_block8.py --resume /ABS/BLOCK8/manifest.json --execute
```

Alternatively supply exact existing approved tokenized artifacts. No latest-run
search, native checkpoint migration or fabricated source quota is performed:

```bash
python -u scripts/run_block8.py --execute --prepare-only \
  --data-manifest /ABS/TRAIN_DEV/manifest.json \
  --evaluation-manifest /ABS/LOCKED_TEST/manifest.json
```

Generate from the selected **block8 export** printed by the workflow:

```bash
python -u scripts/run_block8.py --execute \
  --checkpoint /ABS/BLOCK8_SELECTED_EXPORT/manifest.json \
  --prompt-file /ABS/prompt.txt
```

`--messages-file` instead accepts the shared complete text conversation renderer.
Use `--config PATH` for a new explicit custom recipe. Resume rejects scientific
overrides and changed implementations. Keep an existing running experiment on its
original checkout; adding this package changes the repository implementation hash.
The existing native chat/export loader deliberately rejects block8 exports, and the
block8 loader rejects native and recovery artifacts. No block8 web chat is supplied.

## Selection, recovery and budgets

Selection is the mean fixed-dev-view fully masked CE; it is not AR perplexity.
Step-zero weights are eligible and selected/terminal steps are reported separately.
The locked test is used only after selection for bounded free-running rollouts,
token exact match, EOS/length/work and timing observations. This does not measure
free-form semantic correctness or establish a speedup over original Qwen.

Sharded recovery includes model/mask parameters, FP32 optimizer moments, private
rounding RNG/counters, global and sampler RNG, completed update, selection and
curriculum identity. Every shard is verified before live state restoration.
Only completed updates advance the curriculum. At-target recovery adds no updates.
Within one successful attempt, retain two recoveries and all their referenced
selected exports; replacement commits precede retirement. Earlier failed attempts,
pending writes and their charges remain intact.

Each stage has independent persistent per-attempt and aggregate work/storage/wall
ledgers, a three-attempt cap and cumulative write accounting. Wall limits are
**enabled** by default for this new workflow, independently of the real native
pilot's disabled-clock policy. The default is 12 hours per stage attempt / 36 hours
per stage phase, 500 GB retained per attempt / 1.5 TB per phase, and finite token,
forward-call and cumulative-write caps. These are allowances, not runtime forecasts.

Before weight allocation, meta inventory admits weights, gradients, FP32 moments
and a declared 8 GB activation allowance against a single effective memory pool.
Spark host and device pools are never added. Disk admission covers three recovery
versions, three inference-weight versions and 20 GB free reserve. These checks do
not prove actual activation/kernel/recovery feasibility. Full source admission also
retains the existing 32 GB index / 4 GB cache policy and conservative disk check.

## Validation

Run `python -m pytest` and `ruff check .`. Focused tests live in `tests/block8`.
They cover real tiny Qwen modules, all 36 backbone gradients, active/causal masks,
RoPE and exact cache semantics, label alignment, all eight EOS positions, longer
rollouts, full Parquet preparation, split conflicts, bounded SQLite failure,
sharded export corruption and exact FP32/BF16 interrupted/resumed updates.
All such fixtures remain software evidence only.

Final verification: the complete offline CPU suite passed with **540 passed, two
optional CUDA skips** in 672.21 seconds. The separate tiny BF16 CUDA gradient,
optimizer and cache fixture passed. Lint, wheel packaging, offline planning and all
35 README/guide shell-block syntax checks passed. See the
[dated validation receipt](block8_validation_2026-11-09.json).
