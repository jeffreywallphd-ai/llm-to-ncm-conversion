# Frozen teacher prediction states

Purpose: frozen teacher prediction states through shared native production classes, without an
academic experiment registry.

Inputs: Native data handoff and cache-only pinned Qwen. Outputs: Valid final-normalized P-1+t state shards. Stage handoffs are explicit named
`runs/handoffs/*.json` references bound to exact manifest hashes.

From the repository root, after preceding dependencies and explicit asset preparation:

```sh
ncm teacher extract --config configs/validation/teacher_features_pilot.yaml
```

The separate future budget configuration is `configs/teacher/initial_10m_features.yaml`; it is excluded from automatic validation.

Budget: pilot stages use finite per-run time/work/disk caps from
`configs/budgets/real_pilot.yaml`, with a persistent shared aggregate ledger.
Generation defaults to a finite 120-second, one-canvas limit. CPU validation uses
`configs/budgets/cpu_native.yaml` (180 seconds per stage, 10,800 aggregate seconds)
and predeclared 40-step fixed-batch diagnostics. Large envelopes are separate and
must be calibrated on actual hardware. Initialization, validation, failed attempts,
and checkpoint I/O are charged; no limit silently expands.

Resume/recovery: The initial implementation restarts extraction into a new immutable run; previous work stays charged. It does not silently rerun a teacher for codec epochs.

Current status: implemented, with native CPU production-path tests. Exact current
software-fixture run IDs/outcomes are recorded in
[native validation](../../docs/native_validation.json). Real-data Qwen codec fitting,
real-8B full fine tuning/recovery, quality, and speed are **not executed/not measured**.
Fixtures establish software behavior only. A 1024-slot tiny-width shape test does
not establish substantive long-answer learning. See
[data contracts](../../docs/data_contracts.md), [training](../../docs/training.md),
and [Spark prerequisites](../../docs/spark_setup.md) for stage-specific limitations.

<!-- BEGIN STAGE RESULTS -->
CPU software-fixture runs: `20261029T113634Z-d1fb3269e1` (completed). Real-Qwen validation remains not executed.
<!-- END STAGE RESULTS -->
