# Implementation status

Scope is a runnable CPU research reference and study infrastructure. Research
performance and publication readiness have not been established.

| Capability | Status | Evidence / practical limit |
|---|---|---|
| Package, CLI, locks, CPU workflow | implemented_cpu_validated | Python 3.12.14, torch 2.13.0+cpu, Transformers 5.15.0; CLI plans, data checks, smoke and exports exercised |
| One-command model/data preparation | implemented_cpu_validated (acquisition path) | `ncm prepare`; pinned file-size/hash lock, resumable bounded HTTP, OASST parquet preprocessing, generated teacher/pilot configs and artifact references; full model download not attempted on the small local disk |
| Upstream DFlash adapter | implemented_cpu_validated | Actual pinned model blocks; flat wrapper agrees with direct upstream output on tiny random configurations; no released-weight benchmark |
| Progressive layers/expansion/skip | implemented_cpu_validated | Actual [5,5,9,16,16] stage lengths, nested uneven groups, main-path gradients, trainable gate/factors, frozen target/head tests |
| Feature alignment/isolation | implemented_cpu_validated | Per-anchor strict cutoff, forbidden-feature mutation, document/block isolation and valid later prefixes; target training processes one document per call |
| Greedy verifier / target KV rollback | implemented_cpu_validated | Exhaustive finite proposals plus actual tiny Transformers cache cropping; real GPU precision parity pending |
| Finite stochastic verifier | implemented_cpu_validated | Actual q and positive residual checked algebraically/statistically; initial model study remains greedy |
| Online training/save/resume | implemented_cpu_validated | Two optimizer steps with checkpoint recovery and explicit counters; no Qwen3-8B adaptation or learning results |
| Released financial loaders/scorers | implemented_cpu_validated | Two FinQA training questions and twelve TAT-QA questions from two original contexts; official gold self-checks, safe DSL and scale tests |
| Evidence, BM25, contracts, one repair, routing | implemented_cpu_validated | Deployment interfaces exclude gold; malformed-output/failure tests; no business outcome measurements |
| Fixed stronger dense retriever | implemented_cpu_validated (cache smoke only) | Existing MiniLM encoder at configs/retrieval/minilm.yaml loaded locally for a bounded query; benchmark recall/quality unmeasured |
| Workpaper/load driver | implemented_cpu_validated | Contract/repair and controlled async timeout accounting tests; external serving integration and end-to-end real workload runs pending |
| Run provenance / reports / review export | implemented_cpu_validated | Immutable resolved artifacts and hashes, fixture exclusion, selected-study filtering, preserved notices and identity-redaction tests |
| All eleven core protocols | implemented_cpu_validated | Registry and every README/config/analysis entry point resolve; no research results selected |
| Spark/CUDA parity | not gpu_parity_validated | Actual host is Windows AMD64 with RTX 4060 8188 MiB, driver 610.88; installed torch is intentionally CPU-only |
| Optimized SGLang/vLLM hierarchy integration | incomplete; excluded from performance claims | Source/config audit done; custom engine bridge and parity/timing evidence absent |
| Baseline benchmark reproduction | not benchmark_reproduced | Qwen3-8B target/drafter weights absent; DCore engine not installed |
| Code execution / DocFinQA official scoring | blocked | Docker daemon unavailable; isolated executor not audited; author scorer parity not established |
| Official IFEval/EvalPlus/LongBench harnesses | blocked | Separate pinned evaluator environments/data and code isolation required; no approximate official metrics substituted |

## Setup evidence and failed/blocked attempts

The initial setup CPU test suite completed with **61 passed, 1 optional CUDA test skipped**.
Preparation adds separate acquisition/data/recovery tests; see [prepare.md](prepare.md).
The follow-up full suite passed with 72 tests and one optional CUDA test skipped;
lint and whitespace checks passed.
Tests do not certify a released checkpoint or research outcome.

- `ncm smoke` completed and validated fixture run
  `20261025T233607Z-51f937b78796`, under
  `experiments/shared/SHR-001_environment_and_integrity/runs/`. It trained/resumed
  two steps, checked original training gold and wrote both study reports. It is
  publication-ineligible and supplies no model-quality/speed finding.
- The bounded artificial CPU component profile completed as
  `20261025T233619Z-0f0e3e121059`, under ACL-001's structured runs directory.
  Timings are software diagnostics, never a deployment-speed claim.
- The real ACL-001 pilot preflight recorded blocked run
  `20261025T233441Z-1b660ae94e94`: missing readiness artifact, prepared OASST teacher
  data, locked target snapshot and locked drafter snapshot. No target forward or
  training ran. Exact attempted command is below.
- `docker info --format '{{.OSType}}'` failed because the Docker Desktop Linux
  engine named pipe was unavailable. Next action: on a suitable machine, configure
  and audit a network-disabled non-root bounded sandbox before code scoring.
- The restricted Windows execution shell prevented asyncio's local self-pipe socket
  from initializing. The load-driver tests are run outside that shell restriction;
  they require no remote network, weights, paid API or generated-code execution.
- The upstream API's rotary configuration moved to rope_parameters; the adapter
  checks the loaded modern configuration rather than assume rope_theta exists.

```sh
ncm experiment run --id ACL-001 --config experiments/acl/ACL-001_component_profile_and_pilot/configs/run.yaml --budget configs/budgets/pilot.yaml
```

Before a real pilot, obtain a verified CUDA/Spark environment and explicitly budgeted
locked models (local disk was only about 6 GB free at inspection), prepare train-only
teacher data, and record readiness/parity evidence. This blocked phase reservation is
retained; do not erase its history. Use a deliberately chosen new phase ID for later
budgets if reservations are exhausted. No empirical findings exist, and neither
proposed paper is submission-ready.

`ncm prepare --profile spark` now automates pinned model/data retrieval, prompt
preparation and launch-config generation. Its default first-pilot selection is 18.54 GB;
the current development drive fails the 20.82 GB free-space preflight, so no weights
were fetched here. A bounded real-data check successfully acquired and hashed official
OASST/GSM8K train files and prepared 108 train / 20 dev prompts. Teacher generation,
GPU parity, optimized engines and full-study launch still require their own runs/work.
