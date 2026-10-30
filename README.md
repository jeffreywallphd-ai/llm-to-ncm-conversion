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
Historical correction revision `792ba52`: **114 passed, 1 CUDA skip**; lint,
connected monitoring smoke and preparation plan-only passed.
Historical 51/78-test receipts below remain evidence for their respective revisions.

Recovery lifetime revision `31df0df`: **116 passed, 1 CUDA skip**; all eight focused recovery
tests, lint and the connected monitoring smoke passed. BF16/FP32 weak-reference
checks verify release before replacement allocation and exact continuation; see the
[dated receipt](docs/recovery_lifetime_validation_2026-10-29.json).

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
uv sync --locked --python 3.12 --extra cpu --extra prepare --extra monitoring --group dev
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

## Running tests

Run these commands from the repository root after the installation command above.
The `monitoring` extra is needed for the real TensorBoard event and HTTP tests.
On a machine with a working CUDA environment, follow [Spark setup](docs/spark_setup.md)
instead of installing the CPU Torch extra over it.

| Check | Command |
|---|---|
| Full test suite | `uv run --no-sync python -m pytest -q` |
| Recovery tensor lifetime and exact in-process continuation (BF16 and FP32) | `uv run --no-sync python -m pytest -q tests/integration/test_optimizer_recovery.py -k in_process_recovery` |
| All optimizer recovery, warmup-boundary and monitoring-invariance tests | `uv run --no-sync python -m pytest -q tests/integration/test_optimizer_recovery.py` |
| Tülu Parquet admission, qualification, wave processes and resume | `uv run --no-sync python -m pytest -q tests/integration/test_tulu_data.py tests/integration/test_tulu_wave.py tests/unit/test_extension_selection.py` |
| Optimizer rounding, numerical safety and private RNG | `uv run --no-sync python -m pytest -q tests/unit/test_optimizer_precision.py` |
| Data source contracts and software Parquet preparation | `uv run --no-sync python -m pytest -q tests/unit/test_real_data_contracts.py` |
| Live TensorBoard events/HTTP and replay correctness | `uv run --no-sync python -m pytest -q tests/integration/test_monitoring_tensorboard.py tests/integration/test_replay_corrections.py` |
| Unit tests only | `uv run --no-sync python -m pytest -q tests/unit` |
| Lint | `uv run --no-sync ruff check .` |

Tests use software fixtures; some integration tests perform bounded tiny-model
training and recovery. They do not download research weights or establish real-8B
feasibility. The CUDA kernel test skips when CUDA is unavailable. HTTP tests bind
temporary loopback ports: a sandbox socket denial needs normal localhost execution
permissions, not weakened assertions.

For a connected end-to-end CPU training/monitoring smoke, run:

```sh
uv run --no-sync ncm monitor smoke --config configs/monitoring/smoke.yaml
```

This intentionally runs small codec/generator training, save/reload/resume and fresh
generation. The longer CPU diagnostic is
`uv run --no-sync ncm validate --config configs/validation/cpu_native.yaml`.
Both have finite stage/aggregate budgets and produce ignored `runs/` artifacts.
If the system temporary directory is unavailable, create a workspace parent such
as `.cache/test-runs` and pass a new empty `--basetemp` path under it to pytest;
pytest clears that selected path, so keep previous evidence in separate directories.

The recovery lifetime follow-up releases the completed gradient report immediately
after the validated update, before the recovery probe allocates another model.
Actual before/after observations and current verification are in the
[dated lifetime receipt](docs/recovery_lifetime_validation_2026-10-29.json).

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

## Training stages: CPU checks, OASST pilot, then Tülu extension

Choose the workflow explicitly. CPU checks use tiny software fixtures. The OASST
pilot checks initial operations. After inspecting that pilot and the bounded Spark
feasibility profile, an operator can start the larger Tülu extension. Installation,
help, planning and pilot completion never start it automatically.

| Stage | OASST initial testing | Larger Tülu extension |
|---|---|---|
| Acquire | `ncm prepare --config configs/validation/real_pilot.yaml` | Wave first verifies/acquires tokenizer, data and notices, then qualifies data before weights |
| Prepare data | `ncm data prepare --config configs/data/real_pilot.yaml` | Complete conversations, grouped dedup/split, representative length-aware selection, capacity report |
| Teacher | `ncm teacher extract --config configs/validation/teacher_features_pilot.yaml` | Exact source/length-aware subset with at least 10M distinct training answer positions; frozen pinned teacher |
| Codec | `ncm codec train --config configs/validation/codec_pilot.yaml` | Fresh shared codec, three passes, held-out selection and train-only scales |
| Targets | `ncm targets build --config configs/validation/latent_targets_pilot.yaml` | Selected frozen codec; broader pool streamed into fine codes |
| Generator | `ncm generator train --config configs/validation/generator_pilot.yaml` | Fresh pinned-Qwen initialization, 1M warmup tokens within nominal 100M processed training tokens, full fine tuning and export |

The complete larger sequence has one command, plus offline planning and exact resume:

```sh
ncm wave plan --config configs/waves/tulu3_extension.yaml
ncm wave run --config configs/waves/tulu3_extension.yaml
ncm wave run --config configs/waves/tulu3_extension.yaml --resume /ABS/WAVE/manifest.json
ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006
```

On the locked CPU development environment, prefix commands with `uv run --no-sync`.
On Spark retain CUDA Torch and install `pip install -e '.[upstream,prepare,monitoring]'`;
see [hardware/storage prerequisites](docs/spark_setup.md). CPU installation/tests
are documented above. **Real Tülu capacity and full-model feasibility remain unknown**
until the operator runs those checks; the release size alone does not qualify it.

The dataset is `allenai/tulu-3-sft-mixture` at
`b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`, with six Parquet shards pinned by SHA256
and verified during acquisition (1,412,954,868 bytes). Preparation preserves the entire context and holds out the
final assistant target. It admits at most 512 prompt and 1024 answer tokens including
EOS, excludes overlength references, applies a pinned CPU English heuristic and
component policy, and checks exact OASST overlap. The collection is ODC-BY with
varying component terms, including noncommercial data, and mixed/unknown origins.
Released references do not imply human authorship or independent OASST/No Robots
benchmark status.

Each wave prints its `runs/waves/<id>/manifest.json`. Its `status.json` links exact
child runs, per-wave handoffs, capacity reports, recovery points and failures; normal
training metrics remain under `runs/<stage>/<id>/`. Insufficient/partial capacity
blocks before teacher allocation. Resume verifies this same wave; it never transfers
an OASST checkpoint to different data/codec. Partial preprocessing restarts with
retained charges. Training resumes a complete optimizer boundary. Successful
objective completion forces final validation/checkpoint/export between normal
cadences; safety/resource exhaustion remains incomplete.

See the [full operator guide](docs/extension_training.md),
[extension workflow](workflows/07_extension_training/README.md), and
[dated validation](docs/tulu3_extension_validation_2026-10-29.json) for exact stage
configs, data policies, finite resource ceilings and actual fixture evidence.

Tülu extension verification: **143 passed, 1 CUDA skip**; Ruff and connected
monitoring smoke passed. The connected wave used local Tülu-shaped Parquet and tiny
production classes, including exact resume and owned-process cancellation checks.
These software results do not measure real corpus capacity or full-model training.

## Independent stage workflows and monitoring

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
| [07 extension wave](workflows/07_extension_training/README.md) | Qualified Tülu preparation and fresh, resumable complete training sequence |

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

The old OASST-only `initial_100m_*`/`initial_10m_*` configs are preserved historical
envelopes; they do not establish a 100M source pool or 10M distinct training answer
positions. The Tülu wave above supersedes them as the recommended larger route.
Repetition increases exposures, never distinct-data counts. Real extension budgets
and CPU validation remain separate explicit operations.

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
