# Embedded Qwen hierarchy

This repository implements a proposed native parallel-answer generator initialized
from pinned Qwen3-8B weights. Its quality, retained pretrained capabilities,
full-8B training feasibility, throughput, and speed improvement are **not measured**.
Small CPU diagnostics establish software behavior, not competitive model quality.

The previous speculative drafter and its studies are preserved on
[archive/hierarchical-drafter](https://github.com/jeffreywallphd-ai/llm-to-ncm-conversion/tree/archive/hierarchical-drafter),
at `c5e5329172e63026995a440aa0591a5e15804ab4`.
See [migration](docs/migration.md). Main now has no external drafter, known answer
anchor, AR verifier, acceptance loop, or autoregressive correction.

## Current corrections

The native training/monitoring review corrections C1–C8 are implemented. BF16 now
uses `fp32_moments_bf16_stochastic_v1`: completed tiled FP32 AdamW arithmetic,
FP32 moments, private checkpointed stochastic write-back and no master copy.
Old nearest-write recovery is incompatible; compatible inference exports remain
independent. Codec validation uses fixed `codec_stratified_coarse_indices_v1`
coverage. Phase labels describe executed work; replay reports truthful snapshot
outcomes and counts only derived views. Inference has a separate bounded 128-slot
head policy and one final ID transfer.

See [correction report](docs/native_corrections.md) and
[new validation receipt](docs/native_corrections_validation.json) for actual results,
preserved failures, BF16 CPU recovery evidence and unexecuted hardware checks.
Current final installed-package suite: **114 passed, 1 CUDA skip**; lint,
connected monitoring smoke and preparation plan-only passed.
Historical 51/78-test receipts below remain evidence for their respective revisions.

## Architecture and training

The original 36 Qwen blocks execute once per answer canvas. Blocks 1–8 process
the prompt and four learned answer queries. Each of blocks 9–24 is followed by
two successive hierarchy updates, giving 32 updates from four to 1024 answer
positions. Blocks 25–36 refine the full 1024-position stream. Original transformer
residuals remain intact; every hierarchy update also preserves hidden/compact
residual paths. The compact stream is 384 wide in the real configuration.

A shared continuous codec is trained offline on frozen Qwen final-normalized
prediction states. It supplies cumulative targets after each update pair: sixteen
auxiliary losses. The generator receives only the prompt and its padding mask.
Teacher states, reference tokens, answer lengths, validity masks, and codec targets
stay outside its forward API. The optional module warmup is followed by full
original-backbone fine tuning, including embeddings, final norm, and vocabulary
head. No LoRA is used.

Inference loads only the generator export and tokenizer. It predicts all canvas
positions in one traversal, returns through the first configured EOS, and reports
truncation when EOS is absent. Fixed capacity does not imply successful long-answer
learning. See [architecture](docs/architecture.md) and [data contracts](docs/data_contracts.md).

## Install and bounded validation

```sh
uv sync --locked --python 3.12 --extra cpu --extra prepare --group dev
uv run --no-sync ncm doctor --profile cpu
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv run --no-sync ncm validate --config configs/validation/cpu_native.yaml
```

Installation, help, and preparation planning do not download research models or
start training. Validation uses tiny Qwen production classes and explicitly
labeled fixtures. It performs codec/generator training, save/reload/resume,
fixed-batch learning diagnostics, and fresh-process generation, with finite stage
and aggregate budgets. Failed attempts and their charges are preserved.

<!-- BEGIN NATIVE VALIDATION -->
| Check | Actual result | Provenance / limit |
|---|---|---|
| Native migration at `b496b3e`: tests / installed wheel | 51 passed, 1 CUDA skip in each | CPU software fixtures; includes 1024-slot geometry |
| `ruff check .` | Passed | Native source/tests |
| Connected validation `20261029T113627Z-62a290a040` | Completed, save/resume and fresh process export | CPU; data `0966c13f6a461872a737de7e59f8a9b213ed4f15154fd5fcd5c28e6493b210aa` |
| 40-step codec `20261029T113647Z-8068ba5639` | Loss 1.0300 → 0.8714 | Same fixed batch; no generalization claim |
| 40-step generator `20261029T113652Z-3e5dcd0e7b` | Loss 3.9327 → 2.6797 | Same fixed batch; no generalization claim |
| Fresh prompt-only generation | Immediate EOS; empty decoded text, all 16 fixture slots computed | One traversal; poor early output, no repair |
| Real-data codec / full Qwen3-8B / speed | Not executed / not measured | Missing full weights and suitable training hardware/storage |

Exact artifacts, failures, loss components and environment: [native validation](docs/native_validation.json).
<!-- END NATIVE VALIDATION -->

## Six workflows

Live training monitoring is available through a read-only local TensorBoard view.
Canonical JSONL remains authoritative; the event projector and web server run in
independent CPU-only processes. Install the optional `monitoring` extra alongside
the appropriate CPU or existing CUDA environment. See [monitoring](docs/monitoring.md)
for metric definitions, all 16/32 mappings, samples, resume histories and limits.

```sh
uv sync --locked --python 3.12 --extra cpu --extra prepare --extra monitoring --group dev
# Terminal 1: bounded CPU software fixture, no model download
uv run --no-sync ncm monitor smoke --config configs/monitoring/smoke.yaml
# Terminal 2: can start before training; then open http://127.0.0.1:6006
uv run --no-sync ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006
```

The same monitor observes `codec train` and `generator train` with the stage configs
below. [Monitoring validation](docs/monitoring_validation.json) records its actual
checks. Historical monitoring revision `3072d45`: **78 passed, 1 CUDA skip**; lint and its documented smoke passed. Current correction results are linked above.
Full-model/GPU monitoring overhead remains unmeasured.

| Stage | Purpose |
|---|---|
| [01 data](workflows/01_data/README.md) | Verified source preparation, whole-tree split, separate prompt/reference tokens |
| [02 teacher features](workflows/02_teacher_features/README.md) | Frozen post-normalization prediction states, without full-vocabulary logits |
| [03 codec](workflows/03_codec/README.md) | Shared encoder/decoder training, held-out selection, frozen representation export |
| [04 latent targets](workflows/04_latent_targets/README.md) | Frozen fine-code cache; separate from generator training |
| [05 generator](workflows/05_generator/README.md) | Prompt-only embedded model, optional warmup, full fine tuning and recovery |
| [06 generation](workflows/06_generation/README.md) | Standalone export and bounded parallel generation |

An authorized real pilot starts with an acquisition preview and explicit retrieval:

```sh
ncm doctor --profile spark
ncm prepare --config configs/validation/real_pilot.yaml --plan-only
ncm prepare --config configs/validation/real_pilot.yaml
ncm data prepare --config configs/data/real_pilot.yaml
ncm teacher extract --config configs/validation/teacher_features_pilot.yaml
ncm codec train --config configs/validation/codec_pilot.yaml
ncm targets build --config configs/validation/latent_targets_pilot.yaml
ncm generator train --config configs/validation/generator_pilot.yaml
ncm generator train --config configs/validation/generator_pilot.yaml --resume /ABS/CHECKPOINT/current.json
ncm generate --checkpoint /ABS/EXPORT/manifest.json --prompt-file /ABS/prompt.txt
```

Named `runs/handoffs/*.json` receipts connect stages by exact manifest hashes.
There is no historical-run scan or arbitrary latest checkpoint selection. Resume
uses a complete native recovery checkpoint; an inference export cannot resume
training. See [training](docs/training.md) for timing, storage, and precision rules.

Intended larger budgets remain separate: [10M codec positions](configs/codec/initial_10m_answer_tokens.yaml)
and [100M processed generator tokens](configs/generator/initial_100m_processed_tokens.yaml).
They are not automatic validation. The source-pool target is up to 100M unique
prepared tokens; current source availability and unique-data shortfalls are recorded.
Repetition increases exposures, never unique-data counts.

## Current limitations and extension

The available development environment has CPU PyTorch, no usable CUDA training
runtime, no cached full Qwen3-8B assets, and no DGX Spark. Real head/kernel,
memory, optimizer, checkpoint restoration, quality, and speed require the separate
[Spark feasibility procedure](docs/spark_setup.md). Hardware detection is not
full-training validation. Early native answers may be poor and have no guaranteed
relation to the original AR distribution.

Formal studies will be added later under [experiments](experiments/README.md),
consuming immutable native runs while importing shared `src/ncm_lab` components.
No ACL/IS design, ablation sweep, significance program, financial benchmark, or
publication claim is included in this migration.

The project retains [GPL-3.0](LICENSE). Former vendored code/notices are preserved
on the archive branch; no DFlash/DCore/financial runtime is carried into the native
package. Downloaded Qwen/OASST assets retain their separately recorded license terms.
