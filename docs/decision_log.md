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
