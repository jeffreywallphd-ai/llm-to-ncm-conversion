# Development and scientific integrity

Preserve GPL-3.0 and vendored third-party notices. Keep shared implementation in
`src/ncm_lab`; experiments configure it. Do not start model downloads, training,
benchmark sweeps, remote provisioning, or submissions during planning/install.
Every executable run needs finite per-run and aggregate phase budgets.

Never expose evaluator gold to prompts, retrieval, repair, or routing. Keep whole
reports/conversation trees together; FinQA and DocFinQA are one benchmark family.
Never execute generated Python on the host. Use the allowlisted DSL or a verified
external isolation boundary. Fixtures are software tests and cannot support claims.

Run `python -m pytest` and `ruff check .` after scientific changes. Test actual tensor
lengths, anchor cutoffs, upstream flat parity, frozen weights/autograd, rejection,
EOS, rollback, and publication provenance. Preserve failed runs and immutable
resolved configurations. Do not edit outcomes to make a hypothesis work.

This is hierarchical speculative drafting with an unchanged AR target, not native
NCM conversion. GPU compatibility, reproduction, and speed require separate measured
evidence. Do not infer novelty from names or masks. Record deviations in the decision
log, blockers in implementation status, and actual human contributions in the ledger.
