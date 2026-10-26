# Progressive query expansion for speculative drafting

This repository studies whether a drafter can start with fewer **unknown future
query states**, expand them inside its existing transformer layers, and reduce
generation cost while a frozen autoregressive target verifies every token.
It implements NCM-inspired hierarchical speculative drafting. It does **not**
convert Qwen into a native Next-Concept Modeling generator. The target is unchanged.

The [ACL study](publications/acl/README.md) tests the architecture, controls,
ablations and generalization on general language/reasoning/code tasks. The later
[IS journal study](publications/is_journal/README.md) tests financial evidence,
calculations, workpapers, validation, repair and deployment policy. It requires
new system-design evidence, beyond financial examples of the ACL decoder.
See [study boundaries](publications/publication_relationship.md).

## Architecture

| Layer | Flat future states | Progressive future states | Total with known anchor |
|---|---:|---:|---:|
| 1 | 15 | 4 | 5 |
| 2 | 15 | 4 | 5 |
| 3 | 15 | 8 | 9 |
| 4 | 15 | 15 | 16 |
| 5 | 15 | 15 | 16 |

The known bonus-token anchor is separate and is not predicted. Expansion copies
parents and adds learned child offsets. A zero-gated, active low-rank projection
lifts layer 2 into layer 5. The 51 versus 80 query-position-layer updates are a
work proxy, **not a speed prediction**. See [architecture](docs/architecture.md).

## Install and verify on CPU

Python 3.12 and `uv` are recommended. The CPU environment is isolated from any
working GPU installation. From this checkout:

```sh
uv sync --locked --extra cpu --extra scoring --extra prepare --group dev
uv run --no-sync ncm doctor --profile cpu
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv run --no-sync ncm data check
uv run --no-sync ncm smoke
```

On Windows the installed CLI is `.venv\Scripts\ncm.exe`; on Linux/macOS it is
`.venv/bin/ncm`. The smoke workflow runs two tiny trainable-model steps with
checkpoint resume, released training-data checks, prediction scoring, immutable
run validation and separate reports. It downloads no weights or data.

See [setup](docs/setup.md), [student quickstart](docs/student_quickstart.md), and
[Spark setup](docs/spark_setup.md). Setup was validated on Windows x86-64 with
CPU PyTorch. The detected GPU is an RTX 4060 (8188 MiB); no DGX Spark was available.
Spark/CUDA parity, kernel compatibility and throughput remain unverified.

## Experiment index

| ID | Protocol |
|---|---|
| SHR-001 | [Environment and integrity](experiments/shared/SHR-001_environment_and_integrity/README.md) |
| SHR-002 | [Target and verification](experiments/shared/SHR-002_target_and_verification/README.md) |
| SHR-003 | [Flat and strong baselines](experiments/shared/SHR-003_flat_drafter_baseline/README.md) |
| ACL-001 | [Component profiling and pilot](experiments/acl/ACL-001_component_profile_and_pilot/README.md) |
| ACL-002 | [Main quality and speed](experiments/acl/ACL-002_main_quality_speed/README.md) |
| ACL-003 | [Hierarchy and skip ablations](experiments/acl/ACL-003_hierarchy_and_skip_ablations/README.md) |
| ACL-004 | [Generalization and robustness](experiments/acl/ACL-004_generalization_and_robustness/README.md) |
| IS-001 | [Financial answers](experiments/is_journal/IS-001_financial_answer_accuracy/README.md) |
| IS-002 | [Full-filing workflow](experiments/is_journal/IS-002_full_filing_workflow/README.md) |
| IS-003 | [Workpapers and validation](experiments/is_journal/IS-003_workpapers_and_validation/README.md) |
| IS-004 | [Operating frontier](experiments/is_journal/IS-004_operating_frontier/README.md) |

## Data and scoring

[Manifests](data/manifests) pin sources and preserve original gold. Tiny licensed
FinQA/TAT-QA **training-split** samples have source/hash receipts. Financial gold
is never fabricated. FinQA uses the pinned official execution/program evaluator
behind a bounded DSL validator; TAT-QA uses the pinned official normalization,
EM/F1 and scale implementation. DocFinQA scorer parity remains blocked.

Model prompts use explicit allowed fields. Gold answers/programs/evidence,
future target states, and oracle routing features cannot enter prompts or repair.
FinQA/DocFinQA share a benchmark family. Official splits do not establish unseen
company generalization. HumanEval+ requires verified isolated execution.
See [data provenance](docs/data_provenance.md) and [evaluation](docs/evaluation_protocol.md).

## Plan, run, analyze, reproduce

One command can retrieve and prepare the first pilot's pinned models and data:

```sh
ncm prepare --profile spark
```

This downloads 18.54 GB from an empty cache and needs about 20.82 GB free for
preparation. Use `--assets-root PATH` for another disk, or `--dry-run` for an offline
preview. It checks hashes, prepares train/dev prompts, and prints ready-to-use launch
configs. Teacher generation and training remain separate bounded commands.
See [preparation](docs/prepare.md) for budgets, recovery, installation and optional
financial datasets. The current development drive lacks space for the full assets.

```sh
ncm experiment list
ncm experiment plan --id ACL-001 --profile spark
ncm experiment run --id ACL-001 --config configs/cpu_profile.yaml --budget configs/budgets/cpu_profile.yaml
ncm results analyze --experiment ACL-001
ncm report build --study acl
ncm report build --study is_journal
ncm publication export --study acl --anonymize --output publications/exports/acl-review.zip
```

The CPU profile is an artificial-tensor diagnostic. Research configurations check
cached models, data and correctness prerequisites and record blockers. They do not
download artifacts implicitly. Every run needs finite per-run **and phase** budgets.
Validate a returned ID with `ncm results validate --run RUN_ID`. Use immutable
configurations, artifact hashes and parent runs to reproduce results. See
[workflow](docs/experiment_workflow.md) and [compute plan](docs/compute_plan.md).

## Key findings

<!-- BEGIN MEASURED RESULTS -->

No empirical findings yet.
<!-- END MEASURED RESULTS -->

Engineering validation: CPU tests exercise real upstream DFlash blocks, variable
tensor lengths, feature leakage, autograd, exact finite-model verification,
actual Transformers cache rollback, official training-example scoring, and
checkpoint/report integration. These establish software behavior, not real-model
quality or deployment speed. See [implementation status](docs/implementation_status.md).

Untested hypotheses: proposal-quality retention, runtime advantage, useful skips,
financial-workflow benefits, routing gains and generalization.

## Limitations and blockers

Released Qwen3-8B target/drafter weights are not cached here. No real-model GPU
parity, training, baseline reproduction or research timing was run. Optimized
hierarchy integration into SGLang/vLLM, genuine isolated code execution, the full
official general-task evaluation stack, and journal serving/load integration need
additional validation. No publication-ready or novelty claim is made. Human
experiment owners/reviewers and author roles remain unassigned.

## Repository map and citation

`src/ncm_lab/` separates contracts, data, models, decoding, training, evaluation,
business workflow, runtime, experiments and reporting. `configs/` holds finite
execution settings; `experiments/` holds protocols and ignored raw runs;
`results/index.json` points to derived summaries. `publications/` holds distinct
study/claim manifests. `third_party/` holds exact sources and configurations.
Notebooks are optional; authoritative work uses the package CLI.

Project code retains the original [GPL-3.0 license](LICENSE). Vendored DFlash,
FinQA and TAT-QA code retains adjacent MIT notices. Dataset/model terms are
separate. Cite upstream work from [verified bibliography](docs/references.bib)
and exact revisions from [source lock](third_party/sources.lock.yaml). There is
no project research paper to cite yet. Native conversion is [future work](docs/native_ncm_roadmap.md).
