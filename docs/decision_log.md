# Material setup decisions

2026-10-25: Preserve GPL-3.0 and initial commit; use feature branch
`codex/research-repository-setup`. No commits/pushes or submissions were requested.

Use a pinned upstream DFlash model snapshot with a local layer adapter. DCore supplies
an inspected training objective/reference and candidate strong baselines; its shared
controlled-study contribution is prior art. Do not install its whole distributed stack
as a setup side effect. Optimized serving integration remains explicitly unverified.

Choose CPU Python 3.12.14 from the available bundled interpreter; the host default is
3.15. Use isolated CPU torch 2.13.0+cpu and Transformers 5.15.0 as inspected upstream.
Accommodate the newer rope_parameters interface rather than assume the old rope_theta
attribute. The actual GPU is an RTX 4060, not Spark; no CUDA weights were acquired.

Retain inspected CE positional decay gamma=7, but normalize by valid supervised
future positions rather than upstream batch reduction. Apply this documented change
to both adapted flat and hierarchical controls. This is not exact recipe reproduction.
One source document per target call avoids unvalidated packed-target attention.

Use bounded original training-data prefixes for scorer validation (48 KiB streamed).
Do not download 78 MB FinQA train or multi-GB DocFinQA just to inspect a schema. Full
release counts/hashes remain unaudited. DocFinQA official scorer parity is unresolved.

Do not claim deployment speed from eager references, random-weight/tensor fixtures,
or the calculated query-position work proxy. Keep all setup runs publication-ineligible.
Protocol/CLI for externally optimized workflow/load sweeps fail on precise prerequisites;
their serving integration is additional work, not a silently completed benchmark.

Follow-up: add `ncm prepare` to remove manual pilot-asset acquisition and data-path
wiring. The asset lock pins sizes and SHA256/Git blob IDs for safetensors, tokenizer,
original training data and optional financial releases. Downloads have finite byte,
storage and elapsed-time caps, hash verification and explicit HTTP-range recovery.
Use official OASST train parquet, requiring pinned PyArrow, and derive whole-tree dev
holdout. Test payloads require a separate explicit flag. Teacher data is referenced
by hash from its canonical run rather than copied. Acquisition does not replace GPU
PyTorch or run generation/training. Full weights were not fetched on the small disk.

## Revision-2 implementation update — 2026-10-27

2026-10-27: revision-2 remediation explicitly moves the auxiliary continuous prediction-state codec and causal transfer into the first technical study. Preserve original CE-only method identities. Use a shared placement factory, restricted student-only checkpoints, independently reconstructable levels, predicted-only runtime conditioning and one incremental SDPA verifier. Preserve the GPL/vendor sources, failed/legacy runs and original experiment IDs. The new document supersedes codec deferral, not the prohibition on native-target claims or automatic research execution.

## Continued implementation — 2026-10-28

Use DCore-DFlash-Qwen3-8B-Budget16 at e2f43607fb2531341b09e41889f593c75456a3f2 as the implemented external control. Preserve its untouched source and MIT notice, and disclose that the model card does not specify a weights license. DFlare remains an unimplemented comparison, not a substitute identity. No released weights were downloaded.

Keep all required controls, including the pooled-prefix control, and evaluate focused ablations separately from locked final tests. Share AR/released reference timings explicitly across trained-seed contrasts. Development cost selection can fail if no depth meets tolerance. The 779-job plan is a provisional finite envelope, blocked by real calibration and input/evaluator prerequisites, not an instruction to launch an uncalibrated study.

Measure short durations with perf_counter: Windows monotonic returned a zero-duration tiny profile and exposed a division-by-zero failure. Preserve that failed run; reject nonpositive timing observations. Account for preparation, codec search and resumed dependencies once rather than inferring missing costs.

Pin IFEval scorer dependencies and original LongBench release metadata from primary sources. No benchmark payload was retrieved for this audit. Official scoring runtime validation remains outstanding.
