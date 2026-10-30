# Data preparation

Purpose: data preparation through shared native production classes, without an
academic experiment registry.

Inputs: Verified source training release and pinned tokenizer. Outputs: Native prompt/reference records and grouped train/dev manifest. Stage handoffs are explicit named
`runs/handoffs/*.json` references bound to exact manifest hashes.

From the repository root, after preceding dependencies and explicit asset preparation:

```sh
ncm data prepare --config configs/data/real_pilot.yaml
```

The preserved historical budget configuration is `configs/data/initial_100m_source_pool.yaml`; it is excluded from automatic validation.

Budget: pilot stages use finite per-run time/work/disk caps from
`configs/budgets/real_pilot.yaml`, with a persistent shared aggregate ledger.
Generation defaults to a finite 120-second, one-canvas limit. CPU validation uses
`configs/budgets/cpu_native.yaml` (180 seconds per stage, 10,800 aggregate seconds)
and predeclared 40-step fixed-batch diagnostics. Large envelopes are separate and
must be calibrated on actual hardware. Initialization, validation, failed attempts,
and checkpoint I/O are charged; no limit silently expands.

Resume/recovery: Never resume by appending to a completed immutable artifact; rerun explicitly under the same charged phase.

Current status: implemented, with native CPU production-path tests. Exact current
software-fixture run IDs/outcomes are recorded in
[native validation](../../docs/native_validation.json). Real-data Qwen codec fitting,
real-8B full fine tuning/recovery, quality, and speed are **not executed/not measured**.
Fixtures establish software behavior only. A 1024-slot tiny-width shape test does
not establish substantive long-answer learning. See
[data contracts](../../docs/data_contracts.md), [training](../../docs/training.md),
and [Spark prerequisites](../../docs/spark_setup.md) for stage-specific limitations.

<!-- BEGIN STAGE RESULTS -->
The connected CPU fixture dataset has identity `0966c13f6a461872a737de7e59f8a9b213ed4f15154fd5fcd5c28e6493b210aa` (8 train, 4 dev examples; split by whole group). Bounded raw-JSON preparation and grouped splitting are separately covered by the test suite. Real OASST preparation was not executed.
<!-- END STAGE RESULTS -->

## Tulu extension stage

Complete-context Tulu data, capacity report and OASST overlap index; no teacher allocation.

```sh
ncm data prepare --config configs/data/tulu3_extension.yaml
```

These independent configs use `runs/handoffs/tulu3-standalone/`. For per-wave
resolved handoffs and sequential processes, use the complete
[extension wave](../07_extension_training/README.md) after initial testing.
Resource/safety interruption remains incomplete; native training resumes only
a complete checkpoint, while partial preprocessing restarts with retained charges.
Actual extension software results and real-data/GPU unknowns are recorded in the
[dated receipt](../../docs/tulu3_extension_validation_2026-10-29.json).
