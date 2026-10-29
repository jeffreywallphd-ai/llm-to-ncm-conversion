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
