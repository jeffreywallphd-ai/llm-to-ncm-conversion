# Shared continuous codec

Purpose: shared continuous codec through shared native production classes, without an
academic experiment registry.

Inputs: Frozen teacher feature manifest with held-out groups. Outputs: Selected frozen codec, normalization/scales, recovery shards. Stage handoffs are explicit named
`runs/handoffs/*.json` references bound to exact manifest hashes.

From the repository root, after preceding dependencies and explicit asset preparation:

```sh
ncm codec train --config configs/validation/codec_pilot.yaml
```

The separate future budget configuration is `configs/codec/initial_10m_answer_tokens.yaml`; it is excluded from automatic validation.

Budget: pilot stages use finite per-run time/work/disk caps from
`configs/budgets/real_pilot.yaml`, with a persistent shared aggregate ledger.
Generation defaults to a finite 120-second, one-canvas limit. CPU validation uses
`configs/budgets/cpu_native.yaml` (180 seconds per stage, 10,800 aggregate seconds)
and predeclared 40-step fixed-batch diagnostics. Large envelopes are separate and
must be calibrated on actual hardware. Initialization, validation, failed attempts,
and checkpoint I/O are charged; no limit silently expands.

Resume/recovery: Use --resume /ABS/RUN/checkpoints/current.json. Complete optimizer/RNG/cursor and source contracts are checked.

Current status: implemented, with native CPU production-path tests. Exact current
software-fixture run IDs/outcomes are recorded in
[native validation](../../docs/native_validation.json). Real-data Qwen codec fitting,
real-8B full fine tuning/recovery, quality, and speed are **not executed/not measured**.
Fixtures establish software behavior only. A 1024-slot tiny-width shape test does
not establish substantive long-answer learning. See
[data contracts](../../docs/data_contracts.md), [training](../../docs/training.md),
and [Spark prerequisites](../../docs/spark_setup.md) for stage-specific limitations.

<!-- BEGIN STAGE RESULTS -->
CPU software-fixture runs: `20261029T113637Z-a6dac79462` (completed), `20261029T113638Z-788dbec959` (completed), `20261029T113647Z-8068ba5639` (completed). The predeclared 40-step fixed-batch diagnostic loss changed from 1.029954 to 0.871352. Real-Qwen validation remains not executed.
<!-- END STAGE RESULTS -->

## Live monitoring

Canonical monitoring is enabled by default, with configurable cadence in
`configs/monitoring/default.yaml`. In another terminal, run
`ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006` after installing
the optional monitoring extra. Closing the browser/server does not stop training.
See [monitoring](../../docs/monitoring.md) for objective versus count-weighted
summaries, cutoff/group counts, gradient states, fixed prompt samples and resume
attempts. [Monitoring validation](../../docs/monitoring_validation.json) records
actual CPU fixture integration results; real-8B monitoring overhead is unmeasured.

Monitoring smoke `20261029T120129Z-05391a88a6` completed with canonical codec runs
`20261029T120133Z-7df3585fb4`, `20261029T120134Z-d33df4e982`, `20261029T120139Z-e7f7c601ec`. The final installed-wheel suite passed 78 tests with 1 CUDA skip.
Records and observed samples are in the monitoring validation receipt; all are CPU software fixtures.

## Native correction policies

The [C1–C8 report](../../docs/native_corrections.md) documents the corrected source
binding, `fp32_moments_bf16_stochastic_v1` optimizer/private RNG/recovery contract,
fixed `codec_stratified_coarse_indices_v1` selection, executed-phase monitoring and
truthful replay. Frozen codecs honor trained dtype; target-cache casts are explicit.
Inference/sample head chunks default independently to 128 positions with a declared
temporary-memory cap and one final ID transfer. Training CE/KL limits stay small.
Old nearest-write recovery cannot resume under the new numerical contract; historical
files/scores remain intact. Compatible inference exports remain optimizer-independent.
Current CPU/BF16 evidence is in the [new receipt](../../docs/native_corrections_validation.json);
no full-8B, Spark or GPU speed claim follows from those fixtures.
