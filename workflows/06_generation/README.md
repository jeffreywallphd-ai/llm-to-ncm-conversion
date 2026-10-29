# Prompt-only native generation

Purpose: prompt-only native generation through shared native production classes, without an
academic experiment registry.

Inputs: Independent native export and prompt text; no teacher/codec/cache. Outputs: Parallel tokens/text, EOS/truncation and complete canvas/time telemetry. Stage handoffs are explicit named
`runs/handoffs/*.json` references bound to exact manifest hashes.

From the repository root, after preceding dependencies and explicit asset preparation:

```sh
ncm generate --checkpoint /ABS/EXPORT/manifest.json --prompt-file /ABS/prompt.txt
```

The deployed model computes all answer slots before EOS truncation; an absent EOS is reported explicitly.

Budget: pilot stages use finite per-run time/work/disk caps from
`configs/budgets/real_pilot.yaml`, with a persistent shared aggregate ledger.
Generation defaults to a finite 120-second, one-canvas limit. CPU validation uses
`configs/budgets/cpu_native.yaml` (180 seconds per stage, 10,800 aggregate seconds)
and predeclared 40-step fixed-batch diagnostics. Large envelopes are separate and
must be calibrated on actual hardware. Initialization, validation, failed attempts,
and checkpoint I/O are charged; no limit silently expands.

Resume/recovery: Generation does not resume AR tokens. Another canvas or autoregressive repair is outside scope and is not silently invoked.

Current status: implemented, with native CPU production-path tests. Exact current
software-fixture run IDs/outcomes are recorded in
[native validation](../../docs/native_validation.json). Real-data Qwen codec fitting,
real-8B full fine tuning/recovery, quality, and speed are **not executed/not measured**.
Fixtures establish software behavior only. A 1024-slot tiny-width shape test does
not establish substantive long-answer learning. See
[data contracts](../../docs/data_contracts.md), [training](../../docs/training.md),
and [Spark prerequisites](../../docs/spark_setup.md) for stage-specific limitations.

<!-- BEGIN STAGE RESULTS -->
Fresh-process fixture run `20261029T113712Z-ef1644f373` completed. It returned token `[2]` (EOS), empty decoded text, no truncation, one traversal and all 16 allocated positions computed. Forward/head time was 0.049650 seconds on CPU; this is not an 8B throughput estimate.
<!-- END STAGE RESULTS -->

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
