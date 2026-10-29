# Embedded generator training

Purpose: embedded generator training through shared native production classes, without an
academic experiment registry.

Inputs: Frozen target manifest; generator starts from separate pinned pretrained weights. Outputs: Full-model native recovery and independent selected inference export. Stage handoffs are explicit named
`runs/handoffs/*.json` references bound to exact manifest hashes.

From the repository root, after preceding dependencies and explicit asset preparation:

```sh
ncm generator train --config configs/validation/generator_pilot.yaml
```

The separate future budget configuration is `configs/generator/initial_100m_processed_tokens.yaml`; it is excluded from automatic validation.

Budget: pilot stages use finite per-run time/work/disk caps from
`configs/budgets/real_pilot.yaml`, with a persistent shared aggregate ledger.
Generation defaults to a finite 120-second, one-canvas limit. CPU validation uses
`configs/budgets/cpu_native.yaml` (180 seconds per stage, 10,800 aggregate seconds)
and predeclared 40-step fixed-batch diagnostics. Large envelopes are separate and
must be calibrated on actual hardware. Initialization, validation, failed attempts,
and checkpoint I/O are charged; no limit silently expands.

Resume/recovery: Use --resume /ABS/RUN/checkpoints/current.json. Recovery is exact at optimizer boundaries on deterministic CPU diagnostics; partial accumulation is not restored.

Current status: implemented, with native CPU production-path tests. Exact current
software-fixture run IDs/outcomes are recorded in
[native validation](../../docs/native_validation.json). Real-data Qwen codec fitting,
real-8B full fine tuning/recovery, quality, and speed are **not executed/not measured**.
Fixtures establish software behavior only. A 1024-slot tiny-width shape test does
not establish substantive long-answer learning. See
[data contracts](../../docs/data_contracts.md), [training](../../docs/training.md),
and [Spark prerequisites](../../docs/spark_setup.md) for stage-specific limitations.

<!-- BEGIN STAGE RESULTS -->
CPU software-fixture runs: `20261029T113642Z-504c1356a0` (completed), `20261029T113645Z-9fd277a719` (completed), `20261029T113652Z-3e5dcd0e7b` (completed). The predeclared 40-step fixed-batch diagnostic loss changed from 3.932727 to 2.679719. Text loss 3.506689 → 2.582922; latent loss 4.260381 → 0.967971. Real-Qwen validation remains not executed.
<!-- END STAGE RESULTS -->
