# Embedded Qwen hierarchy

A proposed parallel-answer generator initialized from pinned Qwen3-8B weights.
It computes a fixed answer canvas in one backbone traversal, with four learned
queries and 32 embedded hierarchy updates. The prompt is its only model input;
reference answers and frozen teacher/codec targets are used only for supervision.
Inference needs the generator export and tokenizer, with no autoregressive repair.

**Full-8B training feasibility, answer quality and speed remain unmeasured.** CPU
fixtures verify software behavior. See [architecture](docs/architecture.md) and
[data contracts](docs/data_contracts.md) for the scientific design and limitations.

## Hierarchical eight-token block workflow

The semi-autoregressive workflow now uses the tested native embedded hierarchy:
four queries, all 32 updates, latent width 384 and all 36 Qwen layers per block.
It prepares block data, extracts frozen teacher features, trains/selects the codec,
builds frozen code targets, trains/selects the full-backbone hierarchical generator,
and evaluates full-answer rollouts. The custom mask-token baseline is removed.

```bash
python -u scripts/run_block8.py --plan-only
# Fresh full workflow, including codec and generator training.
python -u scripts/run_block8.py --execute
```

Execution needs verified local assets and finite per-stage budgets; wall limits
are enabled by default. Use a fresh experiment: former custom block8 schemas and
checkpoints are incompatible. Full-8B Spark fit, quality and speed remain unmeasured.
See the [hierarchical block guide](docs/block8.md) and
[architecture plan](docs/hierarchical_block8_plan.md) for preparation, losses,
cache semantics, recovery and the limits of the evidence.

Training prints each completed optimizer step. Average step time and remaining
days/hours/minutes refresh every ten updates and repeat unchanged between refreshes.
Follow an existing experiment read-only from another terminal, including major
data-preparation phases without detailed preparation counters:

```bash
python -u scripts/monitor_block8.py --experiment /ABS/EXPERIMENT/manifest.json
```

See [step progress](docs/block8.md#step-progress-and-remaining-time-estimates) for
older running experiments, pending estimates and recovery behavior. Reports are
triggered by completed work; there are no timer-only progress messages.

## Setup

Run commands from the repository root. On **Spark/Linux**, keep the working
Python/CUDA environment and install the project extras without replacing CUDA Torch:

```sh
python -m pip install -e '.[upstream,prepare,monitoring,chat]'
ncm doctor --profile spark
```

Follow [Spark setup](docs/spark_setup.md) for hardware, memory and storage checks.
Passing preflight does not establish that full training or checkpoint recovery fits.
For a **separate CPU development environment**, install the locked dependencies:

```sh
uv sync --locked --python 3.12 --extra cpu --extra prepare --extra monitoring --extra chat --group dev
uv run --no-sync ncm doctor --profile cpu
```

In that environment, prefix the later Python/`ncm` commands with `uv run --no-sync`.
Installation and planning do not download research assets or start training.

Preview the pinned Qwen/OASST/tokenizer acquisition, then explicitly acquire and
verify the required assets. The second command downloads them and records notices:

```sh
ncm prepare --config configs/validation/real_pilot.yaml --plan-only
ncm prepare --config configs/validation/real_pilot.yaml
```

Pilots and the staged experiments require these local assets. They stop if verification fails;
their preparation steps do not silently download replacements.

## Tests and CPU diagnostics

Run the full suite and lint after changes:

```sh
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
```

For narrower checks, choose the relevant command:

| Check | Command (prefix with `uv run --no-sync` in the CPU environment) |
|---|---|
| Unit tests | `python -m pytest -q tests/unit` |
| Stage 1 qualification, selection, recovery and literal README Bash workflow | `python -m pytest -q tests/unit/test_stage1_policy.py tests/unit/test_stage1_data.py tests/integration/test_stage1_workflow.py tests/integration/test_stage1_bash.py` |
| Stage 2 planning, data gates and unattended training to chat | `python -m pytest -q tests/unit/test_stage2_config.py tests/integration/test_stage2_workflow.py` |
| Optimizer precision and exact recovery/lifetime | `python -m pytest -q tests/unit/test_optimizer_precision.py tests/integration/test_optimizer_recovery.py` |
| Canvas sizes, pilot launch and chatbot | `python -m pytest -q tests/unit/test_canvas_presets.py tests/integration/test_canvas_pipeline.py tests/unit/test_pilot_chat.py tests/integration/test_chat_and_pilots.py` |
| TensorBoard events, HTTP and replay | `python -m pytest -q tests/integration/test_monitoring_tensorboard.py tests/integration/test_replay_corrections.py` |
| Tülu admission and staged wave/resume | `python -m pytest -q tests/integration/test_tulu_data.py tests/integration/test_tulu_wave.py tests/unit/test_extension_selection.py` |

Tests use local software fixtures, including bounded tiny-model training. They do
not download full weights or establish research results. CUDA-only checks skip
when CUDA is unavailable; HTTP tests need permission to bind temporary loopback ports.

For connected training/monitoring or a longer CPU diagnostic, choose one:

```sh
# Tiny codec/generator training, recovery, export and monitoring events.
uv run --no-sync ncm monitor smoke --config configs/monitoring/smoke.yaml

# Longer bounded native diagnostic, including fixed-batch learning checks.
uv run --no-sync ncm validate --config configs/validation/cpu_native.yaml
```

Both save artifacts under ignored `runs/` directories and retain failed attempts.

## Pilots: 2 training steps or 100 training steps

Use pilots to check operations and inspect early learning before Stage 1. Both
retain the full 8B backbone and optimizer-state requirements. Answer capacity changes
sequence geometry, not parameter count. EOS occupies one position; prompts allow
512 tokens. Complete overlength answers are excluded, not cut short with fake EOS.

| Preset | Answer positions |
|---|---:|
| xsmall | 8 |
| small | 32 |
| **medium (default)** | **64** |
| large | 256 |
| xlarge | 1024 |

`--prepare` verifies/builds matching local data, frozen teacher features, a codec
and latent targets before generator training. The pilot codec itself receives only
**two updates**; a 100-step generator pilot does not imply a well-trained codec.
Insufficient complete-answer data stops before teacher allocation. Use Stage 1 for
a longer codec fit.

The **2 training step pilot** is the initial operational check:

```sh
python scripts/run_2_step_pilot.py --fresh --prepare --no-chat
```

The **100 step pilot** targets 100 total generator updates, with validation,
checkpoints and samples every 25 updates. Choose **one** fresh run below;
these are alternatives, not a sweep:

```bash
# Eight positions: at most seven content tokens plus EOS.
python scripts/run_100_step_pilot.py --fresh --canvas-size xsmall --prepare --no-chat
```

```bash
# 32 positions for short answers.
python scripts/run_100_step_pilot.py --fresh --canvas-size small --prepare --no-chat
```

```bash
# Recommended first run: medium, 64 positions.
python scripts/run_100_step_pilot.py --fresh --canvas-size medium --prepare --no-chat
```

```bash
# 256 positions; qualify data and hardware fit separately.
python scripts/run_100_step_pilot.py --fresh --canvas-size large --prepare --no-chat
```

```bash
# 1024 positions; qualify data and hardware fit separately.
python scripts/run_100_step_pilot.py --fresh --canvas-size xlarge --prepare --no-chat
```

Omitting the size also selects medium:

```bash
python scripts/run_100_step_pilot.py --fresh --prepare --no-chat
```

The numeric alias `64` is equivalent to named medium:

```bash
python scripts/run_100_step_pilot.py --fresh --canvas-size 64 --prepare --no-chat
```

For a model-free preview of preparation and training settings, without executing them:

```bash
python scripts/run_100_step_pilot.py --fresh --canvas-size medium --prepare --no-chat --plan-only
```

The installed equivalent is
`ncm pilot train --steps 100 --fresh --canvas-size medium --prepare --no-chat`.
Omit `--prepare` to require already verified matching prerequisites. For an explicit
approved data configuration, add `--prepare --data-config PATH`; see the
[pilot guide](docs/pilot_chat.md) for source limits, capacity blockers and handoffs.

To continue a compatible two-step run to **100 total steps** (98 more after two),
use its exact checkpoint path. Do not add `--fresh` or `--prepare`:

```sh
python scripts/run_100_step_pilot.py --resume /ABS/2-STEP-RUN/checkpoints/current.json --no-chat
```

Recovery verifies the same geometry, data, codec and implementation. Changing size
requires a fresh experiment; historical source checkpoints are not automatically
approved for recovery. Independent inference exports remain usable.

Pilots run in the foreground. Ctrl+C stops owned children and preserves recovery
information. Run/phase/outer training clocks are disabled explicitly; elapsed time
and all finite work, storage, memory and integrity safeguards remain active.
`--no-chat` exits after export and prints its path. Omit it to launch the local
chat server only after successful training.

## Stage 1 single-pass experiment — run everything

Stage 1 is a **separate experiment** with a longer codec fit, followed by full
backbone generator training and held-out evaluation. These are starting settings,
not validated optima or a guarantee of useful answers. All five sizes retain
Qwen3-8B, compact width 384, four queries and all 32 hierarchy updates.

| Setting | Codec | Generator |
|---|---:|---:|
| Optimizer updates | 1,000 | 500 |
| Physical batch / accumulation | 4 / 4 | 1 / 8 |
| Nominal effective batch / presentations | 16 / 16,000 | 8 / 4,000 |
| Parameters | FP32 | BF16 |
| Constant learning rates | 1e-4 | pretrained 1e-5; added modules 1e-4 |
| Weight decay / gradient clipping | 0.01 / 1.0 | 0.01 / 1.0 |
| Validation interval / dev examples | 100 / 128 | 50 / 64 |
| Periodic checkpoint interval | 100, plus improved/final | 100, plus improved/final |
| Trained checkpoint selection | fixed-plan combined codec dev objective | mean per-example dev text CE |

The frozen teacher uses BF16; the selected codec builds FP32 targets with an
explicit consumption cast for the generator. Step-zero measurements are separate
from trained selection. Generator samples use eight fixed dev prompts at step zero
and every 50 steps. There is no module warmup or research recovery probe.

Before copying the block:

- Use **Bash on Spark/Linux**, anywhere inside this repository, with the working
  Python/CUDA environment and the setup requirements above. Keep dependency setup
  separate; never install the CPU extra over validated CUDA Torch.
- The pinned Qwen/OASST/tokenizer assets must already be cached. Missing assets
  stop with the separate acquisition instructions shown under Setup.
- Qualification requires **2,000 train / 200 dev / 200 locked test** complete pairs
  from a bounded 100,000-row approved OASST scan. Capacity shortages stop before
  teacher loading; no duplicated examples fill quotas.
- Medium/64 is recommended first. Change only `CANVAS_SIZE` for another preset;
  this starts one experiment, with its own data and hardware-fit requirements.
- Execution is foreground in a subshell. A failure stops later commands without
  closing or changing your interactive shell. No server or terminal multiplexer starts.
- This creates a **new experiment**. For interruptions, use its printed manifest
  with the recovery command below instead of pasting the fresh-run block again.

<!-- BEGIN REQUIRED README STAGE1 BLOCK -->
```bash
(
  set -euo pipefail
  REPO_ROOT="$(git rev-parse --show-toplevel)"
  cd "$REPO_ROOT"

  # Change only this value to select xsmall, small, medium, large, or xlarge.
  CANVAS_SIZE="medium"
  RUN_ID="$(python -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])')"
  EXPERIMENT_DIR="$PWD/runs/stage1/$CANVAS_SIZE/$RUN_ID"
  MANIFEST="$EXPERIMENT_DIR/manifest.json"
  printf '\nStage 1 experiment: %s\nResume manifest: %s\n\n' "$EXPERIMENT_DIR" "$MANIFEST"

  # Resolve and save the experiment plan; this does not load research models.
  python -u scripts/run_stage1.py --canvas-size "$CANVAS_SIZE" --seed 17 --experiment-dir "$EXPERIMENT_DIR" --plan-only

  # Prepare complete answers and verify train/dev/test capacity before GPU training.
  python -u scripts/run_stage1.py --experiment "$MANIFEST" --stage prepare --execute

  # Extract frozen, aligned teacher features in a separate process.
  python -u scripts/run_stage1.py --experiment "$MANIFEST" --stage teacher --execute

  # Train the FP32 codec: 1,000 updates; batch 4; accumulation 4.
  python -u scripts/train_stage1_codec.py --experiment "$MANIFEST" --execute

  # Build targets from the qualified SELECTED codec, not the old pilot codec.
  python -u scripts/run_stage1.py --experiment "$MANIFEST" --stage targets --execute

  # Train the BF16 single-pass generator: 500 updates; batch 1; accumulation 8.
  python -u scripts/train_stage1_generator.py --experiment "$MANIFEST" --execute

  # Evaluate the selected generator and write the complete experiment report.
  python -u scripts/evaluate_stage1.py --experiment "$MANIFEST" --split test --execute

  printf '\nStage 1 complete.\nManifest: %s\nReport: %s\n' "$MANIFEST" "$EXPERIMENT_DIR/reports/stage1_summary.md"
)
```
<!-- END REQUIRED README STAGE1 BLOCK -->

No script silently skips data or codec qualification. A failure/review stop returns
nonzero, prevents later commands and the final success message, and preserves the
experiment and partial reports. Codec non-improvement requires review. Operational
completion can still yield poor answers; reports distinguish these outcomes.

For **recovery only**, substitute the exact printed manifest path. Completed stages
are verified without repeating training, extraction or evaluation; incomplete work
resumes only from complete checkpoints bound to that manifest:

```bash
python -u scripts/run_stage1.py --resume /ABS/EXISTING/EXPERIMENT/manifest.json --execute
```

After planning, this convenience command runs the same staged workflow:

```bash
python -u scripts/run_stage1.py --experiment /ABS/EXISTING/EXPERIMENT/manifest.json --stage all --execute
```

The codec/generator wrappers in the main block can also run independently once
prerequisites are complete; add `--resume-checkpoint PATH` only for an exact owned
checkpoint. Evaluation never trains or changes selection.

New planning accepts `--config PATH`, `--data-config PATH`, `--codec-steps N`,
`--generator-steps N`, `--seed N` and `--canvas-size`; the default size is medium.
A no-action call defaults to planning. Existing manifests reject conflicting
new-run settings. Stage 1 disables execution clocks while retaining elapsed
observations and finite work/storage/I/O caps derived after data qualification.
See the [Stage 1 guide](docs/stage1.md) for independent stage configs, resource
requirements, review overrides, total-step extension and the isolated overfit diagnostic.

## Stage 2 single-pass experiment — run everything and launch chat

Stage 2 keeps the **medium 64-position canvas** (up to 63 answer tokens plus EOS)
and starts from the pinned Qwen3-8B initialization. It requires **16,000 distinct
training pairs, 200 dev pairs and 200 locked test pairs**. The FP32 codec runs
1,000 updates (batch 4 × accumulation 4); the BF16 generator runs 4,000 updates
(batch 1 × accumulation 8, two nominal passes over the training set).

From the Spark environment in [Setup](#setup), run the full experiment on pinned
OASST:

```sh
python -u scripts/run_stage2.py --execute
```

OASST capacity is unverified. To start with the larger corpus instead, explicitly
stage the pinned Tülu assets once, then run the same experiment on its bounded
short-answer cohort:

```sh
ncm prepare --config configs/preparation/tulu3_sources.yaml &&
python -u scripts/run_stage2.py --source tulu --execute
```

The launcher qualifies the chosen source, extracts teacher features, fits and
checks the codec, caches targets, trains the generator, evaluates the selected
export on the locked test, then serves that export in chat. There are no scheduled
pauses. Failed data, codec or resource checks stop the workflow without switching
sources or reducing the target. A failed codec gate cannot be overridden in
Stage 2. Elapsed time is recorded; training clocks are
disabled, and finite work, storage and checkpoint-write limits still apply.
The one-to-two-day runtime is a planning target; Spark runtime is unmeasured.

For an offline preview, run `python -u scripts/run_stage2.py --plan-only` (add
`--source tulu` for Tülu). The launcher prints the new manifest path and writes
the final report under `reports/stage2_summary.md` in the experiment directory.
Resume an interrupted run, or reopen chat after a completed run, with:

```sh
python -u scripts/run_stage2.py --resume /ABS/EXISTING/EXPERIMENT/manifest.json --execute
```

Open the printed chat URL (by default [http://127.0.0.1:7860](http://127.0.0.1:7860)).
For remote Spark, forward that port with SSH (for example,
`ssh -N -L 7860:127.0.0.1:7860 USER@SPARK_HOST`). Ctrl+C closes the foreground
server; use `--chat-port 0` if the default port is occupied. See the
[Stage 2 Codex prompt](docs/CODEX_STAGE2_EXPERIMENTS_PROMPT.md) for source,
evaluation and verification details.

## Stage 3 single-pass research-candidate experiment — one command

Before launching:

- Run from the repository root in the existing working Spark Python/CUDA environment.
- Use the second Spark or a separate development checkout. Do not update code beneath the ongoing Stage 2 process.
- Pinned Qwen weights, tokenizer, approved Tülu shards and overlap assets must be local and verified. Missing assets stop execution. Acquire Qwen/tokenizer with `ncm prepare --config configs/validation/real_pilot.yaml` and Tülu sources with `ncm prepare --config configs/preparation/tulu3_sources.yaml` separately; execution never installs packages or changes revisions.
- This starts a new experiment. Retain its printed manifest for recovery. Heavy stages run in owned foreground processes without execution-clock deadlines, with finite work, retry, storage and memory safeguards.
- Defaults are **Tülu, large/256, seed 17, teacher-consistent targets**. There is no automatic size or seed sweep. `--execute` authorizes the declared offline distillation and training; help/planning performs neither.

<!-- BEGIN REQUIRED README STAGE3 COMMAND -->
```bash
python -u scripts/run_stage3.py --execute
```
<!-- END REQUIRED README STAGE3 COMMAND -->

This one command prepares source candidates, assigns group-safe splits, constructs
sequence-level teacher targets, qualifies the final selections, extracts aligned
teacher features, trains/selects the codec, builds latent targets, trains/selects
the generator, compares the selected student and original AR Qwen on locked
prompts, writes reports, then launches the **selected Stage 3 export** in local
chat. Generation remains single-pass, whole-answer, with no AR repair.

The target is 100,000 distinct training pairs (explicit minimum 90,000 with an
underfill warning), 2,000 dev and 2,000 locked original-reference test pairs.
The 256-position answer canvas includes EOS. Train/dev teacher outputs must end
naturally within that capacity; truncated teacher output never gains a fabricated
EOS. Original references and source/component licenses remain recorded separately.

The FP32 codec runs 10,000 updates at batch 4 × accumulation 8; the BF16 generator
runs 20,000 updates at batch 1 × accumulation 16, with FP32 Adam moments and the
existing stochastic BF16 write-back. Each stage has 320,000 nominal example
presentations. Learning rates warm up for 500/1,000 updates respectively, then
cosine-decay to 10% of peak. The original backbone remains trainable throughout.
Finite codec quality regressions warn and continue under Stage 3's separate
advisory policy; invalid/nonfinite artifacts stop execution. Stage 1/2 retain
their existing gates and constant rates.

Alternatives:

```bash
# Offline preview; omitting an action also plans only.
python -u scripts/run_stage3.py --plan-only

# Explicit defaults, equivalent scientific recipe.
python -u scripts/run_stage3.py --source tulu --canvas-size large --execute

# Complete evaluation/reports without serving chat.
python -u scripts/run_stage3.py --execute --no-chat

# Recover exactly the recorded experiment.
python -u scripts/run_stage3.py --resume /ABS/EXISTING/EXPERIMENT/manifest.json --execute

# Reopen completed chat on an available local port; no repeated training/evaluation.
python -u scripts/run_stage3.py --resume /ABS/EXISTING/EXPERIMENT/manifest.json --execute --chat-port 0
```

Source preparation has separate finite limits: a 32 GB full-scan SQLite index
and a 4 GB selected candidate cache. Payloads are compressed losslessly; grouping
still covers the complete approved scan before selection. The offline plan shows
an 88 GB source free-disk admission allowance, and failures retain a
`source_storage.json` diagnostic. After updating to this storage fix, start a new
experiment and preserve the old failed manifest/index; do not edit its caps or
bypass identity checks. See [source storage details](docs/stage3.md#source-index-storage-and-the-november-7-failure).

The manifest lives under `runs/stage3/large/<UTC-run-id>/`; reports are
`reports/stage3_summary.md` and `.json`. Selected and terminal steps may differ.
Chat is loopback-only, with independent session limits; a chat error leaves
training/evaluation completed. No duration, hardware fit, convergence or speedup
is guaranteed. Stage 2 is historical context, **not a matched ablation**.
See [Stage 3 recipes, resource accounting, recovery and manual chat](docs/stage3.md).

## Monitoring and testing an export

Run TensorBoard in a **separate terminal**, then open
[http://127.0.0.1:6006](http://127.0.0.1:6006). For Stage 1, replace `runs` with the
experiment's `native_runs` directory to narrow the view:

```sh
ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006
```

This is a read-only CPU view; canonical JSONL remains authoritative. See
[monitoring](docs/monitoring.md) for metrics, samples and replay behavior.

To test a trained export, use the exact export manifest printed by the run.
The server derives canvas size and selected optimizer step from that artifact:

```sh
ncm chat serve --checkpoint /ABS/EMITTED-EXPORT/manifest.json --device cuda --profile spark
```

To preview the interface **without loading a model**, run:

```sh
ncm chat serve --preview
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860); use an SSH tunnel for Spark.
The default server allowance is two hours/100 requests. All canvas positions compute
before EOS trimming, so short returned text does not imply less model work.
See the [pilot/chat guide](docs/pilot_chat.md).

## Other staged workflows and reference material

The larger **Tülu extension** is an explicit separate workflow with its own dataset
terms, qualification, budgets and 1024-position geometry. It never starts after a
pilot or Stage 1 automatically. First inspect the offline plan:

```sh
ncm wave plan --config configs/waves/tulu3_extension.yaml
```

To explicitly acquire assets, qualify data and execute the complete training wave:

```sh
ncm wave run --config configs/waves/tulu3_extension.yaml
```

For recovery, use that wave's exact printed manifest instead of starting a new wave:

```sh
ncm wave run --config configs/waves/tulu3_extension.yaml --resume /ABS/WAVE/manifest.json
```

See the [extension guide](docs/extension_training.md) for capacity and license
restrictions. Individual stage guides cover [data](workflows/01_data/README.md),
[teacher](workflows/02_teacher_features/README.md), [codec](workflows/03_codec/README.md),
[targets](workflows/04_latent_targets/README.md), [generator](workflows/05_generator/README.md)
and [generation](workflows/06_generation/README.md).

Implementation history and measured software evidence live in the
[implementation status](docs/implementation_status.md), [decision log](docs/decision_log.md),
[native corrections](docs/native_corrections.md), [memory accounting](docs/memory_accounting_fix.md)
and the [Stage 1 software validation receipt](docs/stage1_validation_2026-11-04.json).
Real CUDA/Spark training, recovery, quality and speed need separate measured evidence.

The old drafter is preserved on
[archive/hierarchical-drafter](https://github.com/jeffreywallphd-ai/llm-to-ncm-conversion/tree/archive/hierarchical-drafter);
see [migration](docs/migration.md). The project retains [GPL-3.0](LICENSE) and
third-party notices. Downloaded model/data assets keep their separately recorded terms.
