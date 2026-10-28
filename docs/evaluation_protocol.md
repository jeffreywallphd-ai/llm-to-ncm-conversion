# Evaluation and operational accounting

Start with greedy target-output parity. The finite reference exhaustively covers
small proposal vocabularies; the actual Transformers cache path checks rollback.
Real weights/backend/precision need their own parity evidence over prompt lengths,
rejection positions, EOS, tokenized stops and output boundaries. CPU random-weight
parity is not GPU checkpoint parity. Initial research sampling is greedy only.

FinQA canonical scoring uses the pinned official five-decimal execution comparison
and symbolic program equivalence. The allowlisted interpreter rejects unsupported
syntax, future references, oversized literals/exponents and arbitrary code before
calling the official implementation. Unsupported numeric cases are explicit parse
failures, not a new numerical tolerance. TAT-QA uses official scale, EM and F1 behavior.
GSM8K uses the official #### answer delimiter. IFEval's strict official harness,
EvalPlus and original LongBench GovReport evaluation require their pinned environments;
approximate substitutes are not reported as official scores. ROUGE is not factuality.

Generated Python requires network-disabled non-root isolation, read-only inputs,
bounded scratch/processes/CPU/memory/time/output, and no host secrets or sensitive
mounts. Docker presence alone is insufficient. Host AST filters/subprocesses are
not isolation. Code-execution evaluation stays blocked until an audited boundary exists.

Record input/output lengths, truncation, parse failures, timeouts, retries, cache
policy, TTFT, prefill, generation, verification and full request timing where measured.
Unavailable components are null with a reason. CUDA timings synchronize the device;
do not add overlapping component times and call them end-to-end latency. Separate
cold setup, warmup/compilation and steady state, randomize method order, retain raw
observations and record thermal/runtime conditions where available.

Failed requests remain in denominators. Queue throughput uses actual workload elapsed
time rather than summed overlapping request durations. Correct requests/hour and
fully correct workpapers/hour are different metrics. Report p50/p95/deadlines, coverage
and quality among returned/flagged outputs. Bundle only original same-report questions,
without duplication; compare cache-controlled individual and joint generation.

Resample paired report/context/tree groups, with independent training seeds and
timing repetitions separately analyzed. Predeclare endpoints; report denominators,
output length distributions and uncertainty. Do not average ratios across tasks
without specifying weights. Identical verified outputs have identical deterministic
task scores; stochastic judges cannot manufacture decoder quality differences.

Validation distinguishes parses, executes, answer/calculation consistency, reference
existence, actual evidence support, and financial period/unit appropriateness.
Structural checks establish only some of these. A blinded fixed-rubric support audit
needs recorded sampling and human verification; it is not analyst acceptance.

## Revision-2 implementation update â€” 2026-10-27

Current evaluation uses one normalized multi-EOS greedy policy and hash-bound inputs/checkpoints. IFEval strict/loose and LongBench GovReport official scorer calls are implemented against provisioned, hash-verified source. Official environments and full-test data remain unvalidated. HumanEval+ is blocked before loading until a verified external execution boundary exists. Oracle codec diagnostics and fixture outputs are excluded from deployable performance claims.
# V3 endpoint and analysis contracts

GSM8K exact match and IFEval prompt strict accuracy are binary endpoints. IFEval
instruction strict accuracy is total correct instructions divided by total
instructions. Original LongBench GovReport ROUGE-L is continuous; it has no binary
accuracy or correct-request throughput. A missing required score is an explicit
scoring failure, separate from measured zero. Report scored-only denominators and
the declared zero-for-failures estimand separately. HumanEval+ remains blocked
until a verified external execution boundary is provisioned; no generated Python
is executed on the host.

V3 analyses require exact `--runs` or immutable matrix `--bindings`. Distinct tiers,
budgets, selected weights/objectives, independent seed pairs, runtime conditions,
and pilot/final scopes cannot silently merge. Locked comparisons explicitly vary
treatment weights while holding declared request/policy/runtime fields fixed.
Reject duplicate observations, retain unmatched requests and failures, and sort
groups before within-seed bootstrap resampling. Timing blocks are nested within
independent codec/student pairs; there is no inferred across-seed aggregate.
See [correction report](corrections_v3_report.md) for exact commands and coverage.
