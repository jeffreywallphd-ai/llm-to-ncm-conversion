# Tülu 3 extension training

This workflow follows operator inspection of the separate OASST pilot and the
bounded Spark feasibility profile. Installation, help, planning, CPU tests and the
pilot never start the extension automatically. No real Tülu capacity or Qwen
training result is established by the software fixtures.

## Install and operate

For CPU software checks, from the repository root:

```sh
uv sync --locked --python 3.12 --extra cpu --extra prepare --extra monitoring --group dev
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv run --no-sync ncm monitor smoke --config configs/monitoring/smoke.yaml
```

On Spark, retain the validated CUDA Torch installation and use
`pip install -e '.[upstream,prepare,monitoring]'`; see [Spark setup](spark_setup.md).
The preparation extra pins PyArrow and CPU `langid==1.1.6`, whose embedded classifier
is hash-bound in the data policy. No language model download is needed.

```sh
ncm wave plan --config configs/waves/tulu3_extension.yaml
# Explicit operator start, after inspecting initial testing and feasibility:
ncm wave run --config configs/waves/tulu3_extension.yaml
# Resume this exact extension wave, using the manifest printed by run:
ncm wave run --config configs/waves/tulu3_extension.yaml --resume /ABS/WAVE/manifest.json
# In another terminal; rerun when its finite server lifetime ends:
ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006
```

The offline plan reads metadata, file existence/sizes and existing spending ledgers.
It performs no network retrieval, model construction, training, reservations or
handoff publication. Missing prepared data means capacity **unknown**. An optional
top-level `qualified_data` in a copied wave config can explicitly name an existing
data handoff for read-only, content-verified capacity inspection; it does not cause
automatic pilot-to-extension transfer or discovery of a latest run.

## What runs

| Order | Separate child process | Input and output |
|---|---|---|
| 1 | `prepare` tokenizer/data/notices | Dedicated selected extension lock; immutable acquisition receipt |
| 2 | `data prepare` | Six verified Parquet shards and explicit OASST overlap source → grouped data, selection, capacity report |
| 3 | `prepare` weights, then `preflight` | Only after qualification; canonical cached Qwen files are reused and verified |
| 4 | `teacher extract` | Exact qualified codec train/dev IDs → frozen final-normalized prediction states |
| 5 | `codec train` | New shared codec, at least 10M distinct train answer positions, three complete passes; selected export/scales |
| 6 | `targets build` | Selected frozen codec and broader prepared pool → fine codes; full-width states stream without permanent broader cache |
| 7 | `generator train` | New pinned-Qwen generator → nominal 100M processed training prompt-plus-answer tokens, including 1M module-warmup tokens, then full fine tuning |

Stage configs are `configs/data/tulu3_extension.yaml`,
`configs/teacher/tulu3_codec_10m.yaml`, `configs/codec/tulu3_10m_answer_tokens.yaml`,
`configs/targets/tulu3_extension.yaml`, and
`configs/generator/tulu3_100m_processed_tokens.yaml`. They also work independently,
using separate `runs/handoffs/tulu3-standalone/` references. The controller replaces
those with exact per-wave handoffs and immutable resolved configs. It uses the
existing production trainers, objectives, monitoring, precision and recovery.

## Data and qualification

Dataset: `allenai/tulu-3-sft-mixture`, upstream `train`, revision
`b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`. The dedicated asset lock pins six
Parquet SHA256s totaling 1,412,954,868 bytes and the dataset card/notices. Qwen model
and tokenizer stay at `b968826d9c46dd6066d109eabc6255188de91218`. The pilot lock and
download selection are unchanged. Tokenizer/data-only acquisition precedes weights.

One original row becomes one example: optional leading system, every earlier
user/assistant turn, final user context, and the final assistant as the separate
complete target. The target never enters the rendered prompt. The shared renderer
uses the pinned non-thinking chat template with a generation prefix. Unsupported
roles, malformed sequences, empty content, rows above 300 messages/131,072 characters,
and complete contexts above 512 tokens are excluded. Answers above 1,024 tokens
including EOS are excluded, never truncated. Raw special delimiters or special
answer token IDs are rejected; exactly one tokenizer EOS is appended.

The versioned 19-label component policy fails on unmapped labels and retains the
observed inventory. Default exclusions are OASST, multilingual Aya and deliberately
repeated Tülu identity responses. English admission uses normalized `langid` scores
at least 0.9 over up to 4,096 natural-language characters. Code fences, inline code,
display/inline math and obvious code lines are omitted only from classifier evidence;
stored reference text is preserved. Short/uncertain natural-language answers are
excluded. Structured code/math answers can rely on adequate English user evidence.
This is a heuristic, not a proof of English content.

IDs are namespaced by dataset/component. Disk-backed union/find joins known row
groups, canonical prompt alternatives and exact canonical prompt/answer duplicates
across shards before the seed-17, 15%-dev connected-group split. Exact matching
OASST prompt contexts are excluded using the explicit pinned approved-English
immediate-parent source policy, including the eventual held-out groups; an explicit
pilot artifact can additionally supply both partitions. This does not detect
paraphrases or establish benchmark decontamination.

Training selection is deterministic without replacement, source-proportional within
answer bands, with soft example-count weights 20/20/25/20/15% for 1–128, 129–256,
257–512, 513–768 and 769–1024 positions. Exhausted bands redistribute to available
bands. Dev follows its eligible mixture rather than training enrichment. Whole
examples fit under 85M train/15M dev combined-token caps, at most 100M total and
one million examples. Unfilled remainders/shortfalls and zero long-answer coverage
are reported. No group moves to fill a quota.

Each prepared data folder contains `capacity.json`, `capacity.ref.json`, and
`capacity.txt`, bound to the exact data/preprocessing identities. Reports distinguish
eligible/selected train/dev distributions, positions, exposures, exclusions and scan
completeness, codec IDs/order, achieved shares, pass equivalents and storage estimates.
The deterministic codec subset reaches at least 10M **training** answer positions
by whole examples; rounding adds fewer than one canvas. Up to 256 source/length-aware
dev examples are selected; the configured validation allowance evaluates up to 32.
The controller and standalone teacher entry point recompute/verify this same subset
before teacher allocation. Partial scans or insufficient capacity block the wave.
The raw release's example count is not a capacity measurement.

Released targets retain `source_type: reference` while `answer_origin` independently
records synthetic, human, mixed or unknown as supported. The pinned collection card
identifies WildChat GPT-4 responses as synthetic; unsupported component origin claims
remain unknown. Each example retains component attribution/terms, row/group/shard,
dataset revision and preprocessing identity through features and targets. The
collection is ODC-BY; component terms vary and include noncommercial data. Neither
OASST nor No Robots is automatically an independent benchmark for a model trained
on overlapping mixture components.

## Completion, resume and accounting

The default is a fresh codec and fresh pinned-Qwen embedded generator. OASST
recovery checkpoints do not transfer to Tülu or another codec. Exact wave resume
checks software/config/source/codec identities and verifies completed outputs,
handoffs and complete recovery shards. It never scans historical runs. Codec and
generator resume only their wave's explicit complete checkpoint; inference exports
cannot resume training. Interrupted data/teacher/target preprocessing restarts into
a new immutable run; previous failures and charges remain.

`runs/waves/<id>/manifest.json` identifies the wave. `status.json` summarizes the
current stage, qualification, completed stages, checkpoint-based objective progress, child results, recovery
references and failures. `events.jsonl` is append-only; `attempts/` retains child
launch/process records, bounded stdout/stderr and immutable receipts. Native children
remain under `runs/<stage>/<id>/`, so the normal monitor discovers their authoritative
JSONL metrics. A lock prevents concurrent advancement. Stale locks/crash reservations
require operator inspection; they are never silently deleted or refunded.

Controller cancellation/timeout stops its owned process tree, including the Windows
virtual-environment launcher and actual Python interpreter. An interrupted update
is not saved as a completed boundary; resume uses the last verified checkpoint.

The controller's progress fields refer to its last verified complete boundary;
live progress stays in each child's canonical metrics. For standalone generation
from the selected wave export, run
`ncm generate --checkpoint /ABS/WAVE/handoffs/generator.json --messages-file /ABS/messages.json`.

Objectives are explicit `steps`, `epochs` or `tokens`. Fixtures/pilot retain step
objectives. Extension codec success requires its data minimum and three passes;
generator success requires the token objective and an executed full-fine-tuning
update. At a terminal optimizer boundary, final held-out evaluation and a complete
terminal checkpoint occur even between normal 100-step cadences; selected inference
weights remain separate. Export/finalization failures, cancellation, and resource
or premature safety-cap stops cannot become successful completion.

Whole-update token rounding is reported, not hidden. The bound is the maximum
selected-example source length times batch size times accumulation minus one; when
the token target lies within one update of warmup, one additional full update can
be necessary. Hard work budgets reserve this margin plus validation, normalization,
scale fitting, samples and preprocessing. Attempted partial work is charged but
never becomes an exact completed-step checkpoint.

All native stages share phase `tulu3-extension-v1`: 12,096,000 aggregate reserved/
spent seconds, 20M attempted updates/work units, 5B source and answer positions,
20B canvas positions and 3TB retained storage. Per-run wall ceilings: data 86,400;
teacher/codec/targets 604,800 each; generator 1,209,600 seconds. Per-run storage:
20/150/10/120/750GB respectively. These are finite safety ceilings, not duration or
fit forecasts; stage YAMLs give every work cap. New wave directories retain spending.
Each source/model acquisition phase has 21,600 seconds per invocation/86,400 aggregate;
aggregate transfer limits are 2.5GB/25GB. Crash wall reservations remain charged.
Controller overhead has a separate 7,200-second invocation/28,800 aggregate ceiling,
32 aggregate attempts and 4MiB combined log limit per child. Child waiting time is
excluded from controller charges, avoiding duplicate training-time reservations.

10M BF16 width-4096 teacher positions alone estimate 81.92GB; 10M width-384 codes
estimate 7.68GB. Reports use actual selected counts and include metadata, four
retained/pending full generator recovery versions (~330GB), selected export and
128GB additional headroom. Kernel scratch, activations and OS needs remain hardware
measurements. See the separate Spark profile before a real wave.

Primary metadata: [pinned dataset card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/b14afda60f1bbebe55d5d2fa1e4df5042f97f8be/README.md),
[pinned shard metadata](https://huggingface.co/api/datasets/allenai/tulu-3-sft-mixture/tree/b14afda60f1bbebe55d5d2fa1e4df5042f97f8be/data),
and [langid upstream documentation](https://github.com/saffsd/langid.py).

Actual software validation and preserved failures are recorded in the
[dated receipt](tulu3_extension_validation_2026-10-29.json). Real Tülu capacity,
real-data codec fit, full Qwen training/recovery, Spark fit, quality and speed remain
unmeasured until independently authorized operator runs establish them.
