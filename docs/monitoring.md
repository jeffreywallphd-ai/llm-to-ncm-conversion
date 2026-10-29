# Local training monitoring

This is an additive, read-only TensorBoard view of native codec and embedded
generator training. It does not change the architecture, objectives, coefficients,
optimizer, selection rule, or training data. It starts no released-model download.
GPU compatibility and monitoring overhead on Spark are not measured.

## Start training and the browser view

For a CPU development environment:

```sh
uv sync --locked --python 3.12 --extra cpu --extra prepare --extra monitoring --group dev
uv run --no-sync ncm monitor doctor
```

The optional extra pins TensorBoard 2.20.0, Pillow 12.1.0 and setuptools 80.10.2.
TensorBoard 2.20 requires `pkg_resources`; the compatible setuptools pin is
deliberate. TensorFlow is not required. Standard CLI/help and canonical training
records work without the optional extra. On Spark, preserve the working CUDA Torch:
install the upstream, prepare and monitoring extras in the verified CUDA environment,
**without the cpu extra**. See [Spark setup](spark_setup.md).

In terminal one, run a bounded software demonstration:

```sh
uv run --no-sync ncm monitor smoke --config configs/monitoring/smoke.yaml
```

In terminal two (it can be started before training):

```sh
uv run --no-sync ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006
```

After the launcher reports HTTP readiness and a healthy projector state, open
[TensorBoard](http://127.0.0.1:6006). `waiting_for_runs` is a valid empty state;
`catching_up` means replay is in progress. `live` means the projector has consumed
available complete records, not that the trainer is making progress at that moment.
`status.json` includes source time, data age, byte offsets and errors. Long optimizer
steps do not produce invented interim losses. Training failure and monitor failure
are separate. The launcher checks `/data/logdir` and `/data/environment`, including
the exact intended event directory and pinned server version, before readiness.

The smoke command executes tiny Qwen production classes on CPU, two-step training,
resume and three-step fixed-batch diagnostics with one held-out sample at scheduled
validation boundaries. It has a 120-second per-stage / 1800-second aggregate ceiling.
It is software validation, not Qwen3-8B training or an answer-quality demonstration.

For real training, after the explicit data → teacher → codec → targets handoffs,
terminal one instead runs an existing stage command:

```sh
ncm codec train --config configs/validation/codec_pilot.yaml
ncm generator train --config configs/validation/generator_pilot.yaml
```

Use the same monitor command in terminal two. Existing configurations get the
defaults below. A stage may set `monitoring: configs/monitoring/default.yaml` or
provide the same settings inline. No model-code editing or invented checkpoint path
is needed to monitor. `--run-id` and `--attempt-id` filter large run roots. The
launcher never attaches to or terminates an unrelated occupied port.

## Authoritative records and attempts

Each invocation already has a fresh native run directory; monitoring adds a fresh
attempt ID. It preserves this layout rather than moving historical artifacts:

```text
runs/<stage>/<run_id>/
  resolved_config.json, launch.json, environment.json
  attempt.json, attempt_lineage.json
  monitor_metrics.jsonl, monitor_events.jsonl, monitor_samples.jsonl
  metrics.jsonl, events.jsonl, summary.json, manifest.json
  checkpoints/, artifacts/
runs/_monitoring/view-<id>/
  server.log, launcher.json
  projection/projection.json, projection/status.json
  projection/tensorboard/<stage>/<run_id>/<attempt_id>/events.out.tfevents.*
```

Only the streams explicitly declared by versioned `attempt.json` are projected.
Older sealed runs without that declaration are left intact. The earlier migration
receipt is historical evidence for commit `b496b3e`, not a claim that those old
artifacts contained monitoring. JSONL is authoritative; event files/images are
rebuildable. Each record carries schema, stream/sequence, run/attempt, stage/phase,
split, fixture/real origin, UTC and elapsed time, config hash, true optimizer step,
and metric values/counts/units/reductions/status. Resolved inputs bind dataset and
frozen codec identity. Architecture metadata maps IDs to blocks/updates/resolutions.
The configuration hash includes monitoring settings; recovery objective identity
does not use viewing-session state.

Resume gets a new attempt with parent attempt, exact restored checkpoint hash and
numerical counters. Earlier-checkpoint recovery retains abandoned later history in
the original attempt and uses the actual restored step. There is no step offset,
`purge_step`, checkpoint mutation, or TensorBoard budget restoration. Persistent
phase charges include repeated/failed work and remain separate from model counters.
The strict implementation identity still rejects checkpoints from a different
implementation revision; no cross-version resume compatibility is claimed.

Replay always creates a new directory and refuses to overwrite an existing view:

```sh
ncm monitor replay --runs-dir runs --attempt-dir /ABS/RUN --output-dir /ABS/NEW_VIEW
```

Add `--follow` for live appends. Preserve run directories when sharing evidence;
projection alone is not a complete run or a recovery checkpoint.

## Metric definitions

The loss implementation is the authority. The logger receives detached components
from its actual call, performs no extra backward passes, and does not independently
define a training objective. `*/loss/*` shows the actual unscaled microbatch
objective, first averaged according to the trainer's actual accumulation group and
then over completed optimizer steps in the reporting interval. `*/data/*` and group
MSEs retain loss numerators and observation counts for count-weighted summaries.
They can differ when microbatches have unequal counts. Interval endpoints and
microbatch/accumulation information are recorded. No AMP scaler is used currently;
records say so. Attempted budget steps and completed optimizer updates are distinct.

Generator text CE uses slot `t` against answer token `t`, including genuine EOS.
`data/text_ce` contains summed NLL and valid-token count. It is parallel-token CE;
it is not standard autoregressive perplexity. Normalized auxiliary loss is the
equal active-group mean. Its overall coefficient, weighted aggregate and individual
group contributions are recorded. In the default nonempty sixteen-group case:

```text
group contribution = lambda_aux / 16 * normalized cumulative-code MSE
total objective = parallel token CE + sum(group contributions)
```

For group `g=1..16`, supervision follows block `8+g` and update `2g`, at the
configuration's corresponding resolution. Repeated tiny resolutions keep distinct
IDs. Raw squared error, normalized squared error, valid vectors/scalars, means,
actual coefficients, normalization hash and scale-floor convention are retained.
Normalization uses frozen training-derived positive scales floored at 0.001 in raw
encoder coordinates. It compares cumulative predictions with cumulative targets.
Empty means have zero observations and an unavailable value, never a perfect error.

Codec loss remains `0.5*full_normalized_MSE + 0.5*sampled_cutoff_normalized_MSE +
beta*head_KL`. Raw MSE uses the frozen training variance. Components before/after
weighting, counts and observed cutoff IDs are retained. The head evaluates only
the sampled-cutoff reconstruction at at most 32 positions in chunks of at most 8;
the direction is `KL(teacher || reconstruction)`, temperature 1. Cutoff-specific
summaries do not merge a changing mixture into apparent progress. No extra 32-way
reconstruction/head evaluation is launched. Teacher states, encoder codes and
reconstructed states have bounded strided scalar statistics (up to 4096 values).
The variance is a pooled scalar sample variance, not semantic diversity or a
reliable across-example collapse measure. Mean-predictor relative error is
unavailable; no held-out reference statistics are fitted by this patch.

All 32 logical hierarchy updates expose effective gates, resized latent/hidden
base RMS, gated latent-refinement RMS and actual hidden bridge RMS. Zero-base
ratios are unavailable. Observations use all inference-visible slots and never
receive a gold mask/length/target. Capture occurs outside checkpointed original
blocks, so backward recomputation does not duplicate observations. For accumulated
microbatches the displayed activity is the last sampled logical forward; its
observation count is declared. Only CPU numbers survive each reduction.

Gradient summaries inspect existing accumulated, unscaled gradients before clipping.
They cover queries, shared refinement groups, representative original blocks,
embeddings, final norm/head, and other new parameters. Unique parameter identities
are counted once. Shared pairs are named explicitly. Frozen, absent, finite zero,
finite nonzero and nonfinite states remain distinct. Combined-objective gradients
do not establish which auxiliary term caused an update. No full weight snapshots,
per-loss gradient calls, graphs or activation histories are queued.

## Cadence, samples and resource limits

Defaults are in [default configuration](../configs/monitoring/default.yaml): scalar
interval 10 completed steps or the next completed step after 30 seconds; first
three and final available steps visible; hierarchy/gradient details first three
then every 100 steps; prompt samples at scheduled validation no more often than
500 steps, at most four prompts; 60-second cooperative sample-call budget;
2048 display characters. Histograms/graph tracing/full embedding dumps are off.
The smoke configuration uses every-step visibility and one prompt.

Samples use fixed held-out prompt IDs selected by metadata separately from the
training iterator. They use the current model with its actual full configured
canvas, no second model, teacher, targets, gold response input, verifier or repair.
Per-module mixed train/eval modes and Python/NumPy/Torch CPU/initialized CUDA RNG
states are restored. Reference display is off. Actual text, EOS position, returned
length, full computed canvas, generation time and independent display-shortening
flags are recorded. The sample timer is cooperative before/between prompts; an
in-flight forward finishes, any overrun is recorded, and further prompts stop.
Source/canvas work and time are charged to the existing aggregate training budget.

Resource tags distinguish nonpadding source tokens, supervised answer tokens,
allocated canvas and returned sample length. Host step wall time is not a CUDA
device-event compute measurement. Generation reuses the existing synchronized
timer. CPU process RSS/lifetime peak/available host memory and CUDA allocated,
reserved/peak values are recorded when supported; unavailable counters stay null.
Spark host/device capacities are one physical pool and must never be summed.
Validation, sampling and checkpoint durations/status are recorded as events.

Canonical records are newline-terminated, bounded to 1 MiB, immediately flushed for
visibility and fsynced at checkpoint/clean close. This is not a guarantee that every
record survives sudden power loss. Nonfinite numbers are explicit null values with
`nonfinite` status plus an event. Required canonical storage failures follow the
controlled-stop policy and preserve the original error where possible.

The projector uses a separate spawned CPU-only process (`CUDA_VISIBLE_DEVICES=''`
before imports), bounded buffers and a TensorBoard writer with queue 128 / flush
5 seconds. It incrementally reads complete lines, reports corruption, discovers
new declared attempts and keeps at most 128 attempts / 2048 heatmap observations
per series per view. Use filters for more attempts. TensorBoard reloads about every
5 seconds. Rendering uses Pillow, bounded CPU arrays and explicit missing cells.
Raw JSONL is unsmoothed; TensorBoard's UI smoothing/downsampling changes the view,
not measurements. Individual scalar curves are the precise interactive view.

Each view has a cooperative 2 GiB derived-output cap, an 8 GiB aggregate monitoring
tree cap and 2 GiB free-disk headroom by default. Limits are checked around bounded
projection work, so queued writes/one batch can overshoot; they are not filesystem
quotas. Set `--headroom-bytes` to the actual checkpoint/source reservation on the
training machine. The viewer also needs disk bandwidth and CPU/RAM; it is not free
or immune to a full disk. Canonical/checkpoint artifacts are never deleted to keep
the view running. Failure leaves visible status and canonical training continues.

The launcher has a 30-second readiness timeout and a finite 24-hour default view
lifetime (`--max-seconds`, maximum seven days). Replay defaults to one hour. Ctrl-C
terminates only the launcher-owned projector/server; training is never signalled.
Closing the browser has no effect on either training or recorded data.

For remote Spark, run the server on its loopback interface and tunnel from a laptop:

```sh
ssh -L 6006:127.0.0.1:6006 YOUR_USER@YOUR_SPARK_HOST
```

Then use the local browser address. No public deployment or log upload is involved.

## Interpretation and validation

Auxiliary improvement with flat text CE is a prompt to inspect fusion/loss balance.
Tiny injected refinements alongside improving text CE invite checking hierarchy
participation. Train/dev divergence invites checking generalization/data differences.
These are investigation prompts, not diagnoses or evidence of hierarchy benefit.

Exact smoke IDs, commands, event/server checks, test outcomes and limitations are
recorded in [monitoring validation](monitoring_validation.json). This patch measures
CPU software behavior only. No real 8B run, hardware-overhead estimate, competitive
quality claim or formal study is added.

Implementation references: [PyTorch SummaryWriter](https://docs.pytorch.org/docs/stable/tensorboard.html),
[TensorBoard 2.20 core HTTP plugin](https://github.com/tensorflow/tensorboard/blob/2.20.0/tensorboard/plugins/core/core_plugin.py),
[event reader](https://github.com/tensorflow/tensorboard/blob/2.20.0/tensorboard/backend/event_processing/event_accumulator.py).
