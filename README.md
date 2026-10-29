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
| `python -m pytest` / installed wheel | 51 passed, 1 CUDA skip in each | CPU software fixtures; includes 1024-slot geometry |
| `ruff check .` | Passed | Native source/tests |
| Connected validation `20261029T113627Z-62a290a040` | Completed, save/resume and fresh process export | CPU; data `0966c13f6a461872a737de7e59f8a9b213ed4f15154fd5fcd5c28e6493b210aa` |
| 40-step codec `20261029T113647Z-8068ba5639` | Loss 1.0300 → 0.8714 | Same fixed batch; no generalization claim |
| 40-step generator `20261029T113652Z-3e5dcd0e7b` | Loss 3.9327 → 2.6797 | Same fixed batch; no generalization claim |
| Fresh prompt-only generation | Immediate EOS; empty decoded text, all 16 fixture slots computed | One traversal; poor early output, no repair |
| Real-data codec / full Qwen3-8B / speed | Not executed / not measured | Missing full weights and suitable training hardware/storage |

Exact artifacts, failures, loss components and environment: [native validation](docs/native_validation.json).
<!-- END NATIVE VALIDATION -->

## Six workflows

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
