# Implementation handoff — 2026-10-28

The first technical-paper software architecture is implemented and CPU validated. The unchanged autoregressive target supplies post-final-normalization prediction-state labels to an independently reconstructable nested codec. A frozen selected codec then supervises causal latent heads, refinements and live bridges in hierarchical DFlash. Runtime generation uses predicted codes. Native target conversion remains future work.

Validation baseline HEAD: `c60a8a1849410e7266dcdca494362bf108eecceb` on main. Implementation branch: `codex/multiscale-latent-remediation`. The validation receipt records the pre-commit snapshot; Git history records the subsequent implementation commit and main integration. GPL-3.0 and vendored notices are preserved, including the distinct pinned DCore source. The user attachment is not repository implementation source.

## Implemented flow and controls

The shared `src/ncm_lab` code covers placement, response boundaries, multiple EOS, exact greedy verification, incremental KV/features and rollback, deterministic student/codec resume, actual development selection, bounded storage, provenance and input isolation. Codec levels are 4/8/15 with independent reconstructions; width 256 is the declared research starting point. Causal heads predict after layers 2/3/4 and inject into layers 3/4/5, keeping the original skip separate.

Controls include released/adapted flat, hierarchy with/without skip, fixed broadcast, earlier/later/near-budget schedules, selected reduced flat depth, pooled-prefix conditioning, linear/independent/single/channel/pooling codecs, capacity-only, frozen-codec CE-only and auxiliary-only transfer. Named methods resolve through one factory. The selected external baseline is DCore at `e2f43607fb2531341b09e41889f593c75456a3f2`; its direct source forward matches the CPU adapter. No released-weight or SGLang reproduction is claimed. DFlare remains close prior art without an adapter.

Production stages connect teacher generation, codec search/selection/freezing, optional latent cache, student transfer/selection, held-out latent/oracle diagnostics, intervention analysis, development cost selection, parity, official evaluation, analysis and export. All eight shared/ACL experiment matrices now select actual study jobs and dependency closures. The original eleven IDs remain, with ACL-005 added.

The full offline plan has **779 jobs**, paired training seeds 17/29/43, explicit reused AR/released/external controls and randomized ready-job order. It preserves failed/blocked jobs and binds implementation, protocol, inputs and budgets. Its provisional upper envelope is 29,884,000,000 processed source positions, 1,400,700 wall seconds, and 3,096,096,000,000 storage bytes. These are conservative reservation caps, not predicted requirements or a convergence schedule. Calibrate and lock the intended study before execution. The tested calibration guard blocked every research job; none launched.

## Requirement status

Every item has its actual change, files, acceptance evidence, implementation/validation status and remaining external prerequisites in the [machine-readable checklist](remediation_checklist.json). All research-evidence fields remain `none`. FIX-13/14 are explicitly deferred IS-only implementation work.

| ID | Implementation | Validation | Entry point |
|---|---|---|---|
| FIX-01 | complete | cpu_validated | [pyproject.toml](../pyproject.toml) |
| FIX-02 | complete | cpu_validated | [models/factory.py](../src/ncm_lab/models/factory.py) |
| FIX-03 | complete | cpu_validated | [decoding/policy.py](../src/ncm_lab/decoding/policy.py) |
| FIX-04 | complete | cpu_validated | [models/factory.py](../src/ncm_lab/models/factory.py) |
| FIX-05 | complete | cpu_validated | [training/records.py](../src/ncm_lab/training/records.py) |
| FIX-06 | complete | cpu_validated | [training/checkpoints.py](../src/ncm_lab/training/checkpoints.py) |
| FIX-07 | complete | cpu_validated | [training/loop.py](../src/ncm_lab/training/loop.py) |
| FIX-08 | complete | cpu_validated | [models/factory.py](../src/ncm_lab/models/factory.py) |
| FIX-09 | complete | cpu_validated | [decoding/hf.py](../src/ncm_lab/decoding/hf.py) |
| FIX-10 | complete | cpu_validated | [experiments/matrix.py](../src/ncm_lab/experiments/matrix.py) |
| FIX-11 | complete | cpu_validated | [evaluation/tables.py](../src/ncm_lab/evaluation/tables.py) |
| FIX-12 | complete | cpu_validated | [data/artifacts.py](../src/ncm_lab/data/artifacts.py) |
| FIX-13 | not_implemented | not_validated | [data/loaders.py](../src/ncm_lab/data/loaders.py) |
| FIX-14 | not_implemented | not_validated | [business/pipeline.py](../src/ncm_lab/business/pipeline.py) |
| FIX-15 | complete | cpu_validated | [reporting/evidence.py](../src/ncm_lab/reporting/evidence.py) |
| FIX-16 | complete | cpu_validated | [reporting/portable.py](../src/ncm_lab/reporting/portable.py) |
| FIX-17 | complete | cpu_validated | [runtime/prepare.py](../src/ncm_lab/runtime/prepare.py) |
| PAPER-01 | complete | cpu_validated | [docs/research_plan.md](../docs/research_plan.md) |
| PAPER-02 | complete | cpu_validated | [models/codecs.py](../src/ncm_lab/models/codecs.py) |
| PAPER-03 | complete | cpu_validated | [models/latent_adapters.py](../src/ncm_lab/models/latent_adapters.py) |
| PAPER-04 | complete | cpu_validated | [training/codec.py](../src/ncm_lab/training/codec.py) |
| PAPER-05 | complete | cpu_validated | [experiments/acl/ACL-005_multiscale_codec_and_transfer](../experiments/acl/ACL-005_multiscale_codec_and_transfer) |
| PAPER-06 | complete | cpu_validated | [experiments/acceptance.py](../src/ncm_lab/experiments/acceptance.py) |

## Executed validation

- Source checkout: **127 passed, 1 skipped**, 26.37 s. The skipped test requires CUDA.
- Installed wheel in a disposable local clone with the final source overlay: **127 passed, 1 skipped**, 25.63 s. Three warnings originate in unchanged vendored TAT-QA regular expressions.
- `ruff check .` and authored-source Git whitespace checks passed. The wheel contains DCore, DFlash, FinQA and TAT-QA license notices.
- A saved-tiny-model production DAG runs actual teacher, codec, latent-cache, transfer, held-out diagnostic, evaluation, parity, profile and cost-selection executors. Separate regressions compare uninterrupted/resumed codec candidates and student optimizer state/order/weights.
- The full research plan and all eight experiment plans resolved offline. Launching the guarded research plan produced 779 blocked jobs and **zero research executions**.
- The final engineering archive validates 192 files. It was unpacked outside the checkout, then hash-checked and reanalyzed through the trusted installed CLI. No archive code was imported or executed. Fixtures remain research-ineligible.

Exact software commands:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check .
uv build --wheel --offline --quiet --out-dir .cache/remediation-wheel-final
uv pip install --python .venv/Scripts/python.exe --target .cache/remediation-installed-final --no-deps --offline --reinstall .cache/remediation-wheel-final/ncm_lab-0.1.0-py3-none-any.whl
.venv/Scripts/python.exe -m ncm_lab.cli acceptance
.venv/Scripts/python.exe -m ncm_lab.cli results codec-analysis --runs 20261028T125454Z-a9d35633b35d --output results/generated/remediation-final/figures --figures
```

The clone test set `NCM_REPO_ROOT` to `.cache/remediation-clean-check`, `PYTHONPATH` to the trusted `.cache/remediation-installed-final`, and ran the original venv Python from that clone. This verifies package imports independently of the editable checkout. Full command outcomes and hashes are in [remediation_validation.json](remediation_validation.json).

Final fixture run: [20261028T125454Z-a9d35633b35d](../experiments/acl/ACL-005_multiscale_codec_and_transfer/runs/20261028T125454Z-a9d35633b35d/manifest.json). Manifest SHA256: `9ba241eea21d1d002d95d4390497dad84e7812372342ee1614dbd8d457d06902`. [Engineering archive](../publications/exports/20261028T125454Z-a9d35633b35d-engineering.zip), SHA256 `f08d70667fc774c1b078daf6447eb0fefe23e266f178cd0bf030e4835a31b69b`. The run recorded 13 optimizer updates, 20 budgeted operations, 1359 target-processed positions and 48 teacher-generated fixture tokens. Executor time was 5.281 s and stored artifacts were 3,384,117 bytes at resource recording. These tiny-model observations support software validation only.

[Tables and figures](../results/generated/remediation-final/figures/tables.json) contain 12 rate/distortion rows, 9 learning observations, 6 interventions, 3 oracle/predicted comparisons and one dependency-cost row. Missing measurements remain null. [Resolved study plan](../results/generated/remediation-final/study-plan.json), digest `aedc571ef242adb95a3bb387261f791405007430e5cd552aafcb1401dfad9413`.

## Failed attempts retained

The initial baseline had 71 passing tests, one failure and one skip. CRLF conversion of the two released training samples differed from existing LF receipts; newline-only restoration recovered their pinned hashes exactly. No gold, receipt hash or outcome was adjusted.

The expanded executor test exposed a zero-duration tiny Windows profile and division-by-zero failure. Profiling now uses `perf_counter` and cost selection rejects nonpositive timings. The failed run remains intact. An early disposable wheel install hit sandbox PE-resource access restrictions; the approved offline install succeeded.

Automatic approval review rejected execution of an extracted generated validator under the repository's host-execution rule. The final path uses the trusted installed CLI for hash verification and data-only reanalysis. The optional packaged script was not executed in final validation and requires a verified external isolation boundary.

## Next bounded work

For another CPU integration:

```sh
uv run --no-sync ncm acceptance
```

For actual cached-model work, use a separate compatible CUDA/Spark environment; preserve its GPU PyTorch rather than syncing the CPU extra. Preparation is a separate explicit asset operation:

```sh
ncm doctor --profile spark
ncm prepare --profile spark --assets-root /large/research-assets --include-external-baseline --dry-run
# Only when deliberately retrieving the budgeted assets:
ncm prepare --profile spark --assets-root /large/research-assets --include-external-baseline
```

A completed preparation prints an immutable attempt directory. Its `launch/verification.yaml` is cache-only released-flat parity; `teacher.yaml` and `teacher_dev.yaml` produce canonical train/dev references; `pilot.yaml` supplies bounded training/memory/storage calibration. With ATTEMPT set to that printed directory:

```sh
ncm experiment run --id SHR-002 --config ATTEMPT/launch/verification.yaml --budget ATTEMPT/launch/pilot_budget.yaml
ncm experiment run --id SHR-001 --config ATTEMPT/launch/teacher.yaml --budget ATTEMPT/launch/teacher_budget.yaml
ncm experiment run --id SHR-001 --config ATTEMPT/launch/teacher_dev.yaml --budget ATTEMPT/launch/teacher_budget.yaml
ncm experiment run --id ACL-001 --config ATTEMPT/launch/pilot.yaml --budget ATTEMPT/launch/pilot_budget.yaml
ncm matrix plan --spec ATTEMPT/launch/codec_pilot_matrix.yaml --output results/generated/new-codec-pilot-plan.json
ncm matrix run --plan results/generated/new-codec-pilot-plan.json
```

The generated codec DAG explicitly links codec fitting and selection, frozen transfer, student selection and adapted parity. Optional DCore parity uses a distinct checkpoint/config. The bounded calibration pilot remains separate from the locked study's mandatory full-suite overlap audit. Teacher caps produce a small pilot corpus; larger data tiers require genuinely more disjoint response tokens, never repeated-token fill.

Validate a produced codec/student selection, evaluate and assess evidence as follows. Replace uppercase placeholders with the recorded actual paths; never invent checkpoint or manifest hashes:

```sh
ncm checkpoint validate-selection --path CODEC_RUN/codec-selection.json
ncm checkpoint validate-selection --path STUDENT_RUN/selection.json
ncm matrix plan --spec experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml --output results/generated/new-locked-study-plan.json
ncm matrix run --plan results/generated/new-locked-study-plan.json
ncm results analyze --experiment ACL-005
ncm results analyze --experiment ACL-002
ncm results attest --run EVALUATION_RUN --claim-class performance
ncm results attest --run FULL_TEST_RUN --claim-class full_test
ncm results select --study acl --run EVALUATION_RUN --attestation ACTUAL_ATTESTATION.json --claims ACL-005-C4
ncm report build --study acl
ncm publication export --study acl --anonymize --output publications/exports/new-acl-review.zip
ncm publication validate-export --path publications/exports/new-acl-review.zip --resolver PORTABLE_SHA256_CACHE --reanalyze
```

Before resolving the locked study, fill its real calibration reference, canonical teacher/source receipts, immutable benchmark inputs, official evaluator roots and actual cost evidence. Re-resolve after any dependency or input changes. Provision pinned official IFEval and original LongBench scorer environments; their API calls and source hashes are implemented. HumanEval+ scoring remains blocked until verified external isolation exists. An attestation describes a specific evidence class; author selection and claim interpretation remain separate decisions.

## External prerequisites and scope limits

This Windows host has PyTorch 2.13.0+cpu and Transformers 5.15.0. The GPU probe finds an RTX 4060 with 8,188 MiB, but CUDA is unavailable to this environment. No Spark, Qwen3-8B weights or audited generated-code sandbox was available. Only small public source/metadata files were fetched during continuation, not research weights or benchmark payloads.

Released-drafter structural estimates are 1,048,626,432 parameters; 2,097,252,864 bytes of BF16 weights; approximately 6,291,758,592 bytes for weights plus two same-dtype Adam moments; and approximately 14,680,770,048 bytes for latest + atomic replacement + selected deployment weights before new modules/metadata. These are meta-device calculations, not measured released checkpoints or GPU memory. Actual peak memory, step rate, convergence, corpus sufficiency, official reproduction and Spark throughput are unknown.

Code intentionally outside first-paper scope: IS-specific FIX-13 financial grouping/relationship repairs and FIX-14 bundle/workflow/workload dispatch. HumanEval+/arbitrary generated-Python execution is an unsupported capability until a verified boundary is provisioned. DFlare and second-target/concurrent-serving adapters are not implemented and cannot support broad generalization claims.

Implemented code still awaits released-weight GPU parity/calibration, actual locked training/data audits, official evaluator runtimes, repeated research runs and positive replay of genuinely selected measured-research exports. No research speed, quality, business, novelty or publication-readiness conclusion is established. No full research training, model download, benchmark sweep, remote provisioning or submission was launched.
