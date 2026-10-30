# Complete Tülu extension wave

Purpose: after operator inspection of the OASST pilot and Spark profile, run the
fresh extension preparation → qualified frozen teacher → codec → targets → full
generator sequence through shared `src/ncm_lab` implementations.

Inputs: pinned Tülu/Qwen asset locks, CPU preparation dependencies, suitable verified
CUDA hardware/storage, and explicit operator start. Outputs: per-wave resolved
configs/handoffs, capacity report, immutable child runs, selected exports and
terminal recovery checkpoints. No real capacity or GPU result is claimed here.

```sh
ncm wave plan --config configs/waves/tulu3_extension.yaml
ncm wave run --config configs/waves/tulu3_extension.yaml
ncm wave run --config configs/waves/tulu3_extension.yaml --resume /ABS/WAVE/manifest.json
ncm monitor serve --runs-dir runs --host 127.0.0.1 --port 6006
```

Plan is offline and read-only. Run stops on failed/blocked/cancelled/resource-stopped
children; a sub-10M or partial scan blocks before teacher/model-weight acquisition.
Resume verifies only this wave's completed outputs and explicit native training
recovery. Partial preprocessing restarts with retained charges. New wave directories
do not reset the shared finite phase ledger. Pilot handoffs remain independent.

See [operator instructions](../../docs/extension_training.md) for installation,
source/language/overlap policies, exact budgets, stage commands and stopping behavior.
[Validation](../../docs/tulu3_extension_validation_2026-10-29.json) records actual
software-fixture results. Real Tülu capacity, full-model feasibility, quality and
speed remain unmeasured.
