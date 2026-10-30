# Native data and artifact contracts

Native examples use `native_answer_example_v1`: original ID, source group, train/dev
split, source type, separate prompt/answer token arrays, genuine EOS/censoring flags,
and source/model/tokenizer/template/license receipts. Old known-answer-anchor and
15-position schemas are rejected. There are no evaluator test answers in this pipeline.

Preparation uses the pinned non-thinking chat template for a user prompt followed
by the assistant generation prefix. Deployment calls the identical helper. Real
OASST preparation joins approved English assistant/parent pairs with an on-disk
SQLite index and assigns entire conversation trees deterministically to train/dev.
This initial single-parent representation does not claim full-dialogue evaluation.
No corpus-sized teacher/state/code tensor is loaded into RAM.

Complete references include actual tokens and a terminal EOS. Loss padding after
EOS is ignored. Overlength references default to exclusion; explicit prefix censoring
does not invent EOS. No validity/length flag enters student attention. Generator
inference computes the whole canvas and returns through its configured EOS policy.
The primary reference terminator is the tokenizer EOS; pinned Qwen generation also
recognizes the configured alternative end-of-text token.

The frozen teacher receives prompt plus preceding reference tokens. Its state for
answer y[t] is the final-normalized hidden state at position P-1+t. The first token
is fully supervised and is not a known student anchor. Extraction uses the backbone
output directly, without all-layer hidden outputs or full-sequence vocabulary logits.
The frozen original vocabulary head is applied directly, without a second norm, for
alignment tests and bounded codec consistency losses.

The codec is `continuous_multiscale_pyramid_v1`: one encoder/decoder shared across
all resolutions. Masked means use fixed canvas intervals, not true-length-relative
bins. Empty bins are zero and excluded from losses. Cumulative targets are pooled
fine codes; progressive residuals subtract resized preceding cumulative targets.
Both caches and predictions use raw encoder coordinates. Saved train-only channel
scales (floor 0.001) affect auxiliary losses, not residual arithmetic. State MSE uses
a positive train-only variance statistic. Held-out data never fits these statistics.

Data/features/targets have separate `native_data_v1`, `native_features_v1`, and
`native_targets_v1` manifests with a hashed streamed index and bounded per-example
tensor shards. Only valid teacher/code positions are stored. Logical canvas padding,
answer-length histogram, unique prepared positions, and repeated training exposures
are separate fields. “Unique source tokens” means distinct prepared-example token
positions; shared prompt prefixes in different examples are not raw-message deduplication.

Cache identities bind original/tokenized source, teacher/tokenizer pins and template,
prediction-state alignment, selected codec hash/scales, canvas, schedule, dtype, and
schema. Missing or incompatible targets fail preprocessing; generator training never
loads an online teacher as fallback. The optional codec teacher subset can stop at
10M training answer positions by whole examples, with any small overshoot reported.
A broader generator pool streams teacher states into fine codes in a separate process.

At BF16, 10M valid width-4096 states are 81.92 GB and width-384 codes are 7.68 GB
before metadata. These are decimal-byte estimates, not storage measurements.
Pinned source receipts and actual stored bytes remain authoritative. Unavailable
unique data is a shortfall; epochs never create new unique examples.

## Tülu complete-context extension

The dedicated six-shard Tülu lock and versioned source/language/selection policy are
described in [extension training](extension_training.md). One released row becomes
one example: optional system message, all preceding user/assistant turns and the
final user context, with the final assistant entirely outside the prompt. The shared
`prompt_tokens()` accepts either the historical user string or validated messages.
Standalone generation adds mutually exclusive `--messages-file` / `--prompt-file`
routes using exactly that renderer; a messages file is a JSON list of text
`{"role": "user", "content": "..."}` objects ending with a user turn.

Message/character bounds precede tokenization. Complete contexts above 512 tokens
or targets above 1024 including EOS are excluded. Raw special delimiters/interior
special answer tokens are rejected, then exactly one tokenizer EOS is appended.
No text is shortened to fit. The explicit admission order and every exclusion
category are retained. Language filtering uses pinned CPU langid evidence, not ASCII;
uncertain natural-language answers are excluded, with documented code/math handling.

Disk-backed connected groups join known conversation variants, exact prompt/answer
duplicates and alternative targets for the same complete rendered prompt before
seed-17 grouped splitting. Explicit hash-bound OASST inputs supply exact overlap
prompts from both partitions. No historical latest-run search or split movement is
permitted. Selection is deterministic without replacement; dev is representative
of its eligible mixture rather than training length enrichment.

`source_type: reference` means a released reference, independently of
`answer_origin: human | synthetic | mixed | unknown`. Component origin, attribution,
terms, dataset revision, original row/group and shard identity persist through
teacher/features/targets. Unsupported origin/terms claims stay unknown. Tülu's
collection license does not replace component terms; OASST/No Robots mixture overlap
prevents treating them automatically as independent benchmarks.

`capacity.json`/`capacity.ref.json` bind counts, selection and qualification to the
exact data manifest and full preprocessing identity. Source shards, renderer,
message/EOS/language/dedup/split/selection settings and overlap inputs all affect
that identity. Capacity distinguishes complete from bounded partial scans, eligible
from selected train/dev, distinct positions from repeated exposures, and unknown
from qualified. Both the wave and standalone teacher recompute the committed
codec subset before allocating the teacher.
