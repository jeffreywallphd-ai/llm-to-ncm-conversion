# SHR-003: Shared baseline references

Purpose and hypothesis: Measure the actual AR, released DFlash, and pinned DCore reference procedures on common official requests.

Methods and controls: References are reused explicitly across independent adapted seed-pair comparisons. Released references are not newly trained repetitions.

Data and split: Complete designated GSM8K, IFEval, original LongBench GovReport, and HumanEval+ splits. Missing official inputs/scorers or code isolation block execution.

The [single study declaration](../../acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml)
and [locked contrast definitions](../../../publications/acl/contrasts.yaml) generate
this folder's matrix plus required dependencies/consumers. Seed pairs 17/17,
29/29, and 43/43 independently fit codecs and students. Five randomized timing
blocks are nested within each pair; shared references remain shared. ACL-001
cost profiling uses three recorded repeats after its warmup. Detailed source,
endpoint, failure, pairing, and cost rules are in the
[V3 correction report](../../../docs/corrections_v3_report.md).

Finite budgets: stage runs reserve at most 100 attempted steps, 40M processed
positions, 1,800 seconds, and 16 GB; evaluation reserves 10,000 steps, 50M positions,
1,800 seconds, and 256 MB. Audit and robustness analysis have separate finite
budgets. Conservative phase caps, reserved totals, and download cap zero are
checked by planning. These candidate caps require real calibration before a full
study; they do not guarantee completion of a data tier or official split.

From the repository root:

```sh
ncm matrix plan --spec experiments/shared/SHR-003_flat_drafter_baseline/configs/matrix.yaml --output results/generated/SHR-003-v3-plan.json
# Only after assets, measured calibration, and study authorization:
ncm matrix run --plan results/generated/SHR-003-v3-plan.json
ncm results analyze --experiment SHR-003 --bindings MATRIX_OUTPUT/analysis-bindings.json
```

Planning is offline and model-free. `MATRIX_OUTPUT` is the exact returned execution
directory; there is no latest-run or historical scan selection. For a deliberately
exploratory analysis, use `--runs EXACT_RUN_ID ...` instead of `--bindings`.
`ncm acceptance` exercises a bounded connected CPU fixture path without research
downloads. The complete candidate plan and all eight subset plans have static
validation; distinct production contracts have bounded CPU tests.

Outputs: immutable per-run resolved configuration, source/input/dependency hashes,
resource counters, raw observations, failures and selection receipts as applicable;
matrix role bindings; new immutable per-seed grouped analyses with unmatched and
failed requests retained. Codec tables distinguish selection, held-out diagnostics,
normalized error units, intermediate costs, and full selected-procedure costs.
ACL-004 writes `strata.json` identifying the shared source runs and denominators.

Results status: implemented software with bounded CPU validation. Released-model
GPU/Spark validation and research measurements are **not measured**. No speed,
quality, broad generalization, or publication-ready result follows from fixtures.
Research claim selections remain empty. Official data/scorer environments, cached
pinned models, complete source contracts, real calibration, and HumanEval+
isolation remain external prerequisites. IS-only FIX-13/14 are deferred.
