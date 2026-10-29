# Development and scientific integrity

Preserve GPL-3.0 and vendored third-party notices. Keep shared implementation in
`src/ncm_lab`; workflows compose it. Do not start model downloads, training,
benchmark sweeps, remote provisioning, or submissions during planning/install.
Every executable run needs finite per-run and aggregate phase budgets.

Keep reference tokens, true answer lengths/masks, teacher states and target codes
outside the generator's prompt-only forward API. Keep whole conversation trees
together across splits. Never execute generated Python on the host. Fixtures are
software tests and cannot support research or business claims.

Run `python -m pytest` and `ruff check .` after scientific changes. Test actual tensor
lengths, all hierarchy updates, attention/position geometry, label alignment,
full-backbone gradients, frozen teacher/codec targets, EOS and sharded recovery.
Preserve failed runs and immutable
resolved configurations. Do not edit outcomes to make a hypothesis work.

Main implements a proposed native embedded hierarchy, with one original-backbone
traversal per answer canvas and no drafter or AR correction. The old drafter is
preserved on archive/hierarchical-drafter. GPU compatibility, full-8B optimizer and
recovery feasibility, quality, and speed require separate measured evidence. Do not
infer novelty or semantic concepts from names, pooling, or masks. Record deviations
in docs/decision_log.md, blockers in implementation status, and actual contributions.
