# Native migration record — 2026-10-29

The actual pre-migration commit and reviewed baseline are both
`c5e5329172e63026995a440aa0591a5e15804ab4`. The tracked tree was clean; the user's
untracked `.codex-remote-attachments` were preserved. No hard reset, blind stash,
history rewrite, force push, or unrelated blanket commit was used.

Preservation branch: `archive/hierarchical-drafter`. It was created locally at that
exact SHA and published to origin after explicit user approval. `git ls-remote
--heads origin archive/hierarchical-drafter` returned the same full SHA. Its Git
history preserves old source, tests, study configs, provenance, and third-party
notices. Old generated/ignored outputs were left in place and are not consumed by
the new native path. There is no giant active legacy source directory.

Native migration branch: `codex/embedded-hierarchy-migration`. The ordinary commit
containing this document is the native migration reference; retrieve its full SHA
with `git rev-parse codex/embedded-hierarchy-migration`. The user requested tests,
commit, and a local fast-forward of main. No main remote push or branch-protection
bypass is part of this local update. A later remote publication follows repository
policy. No PR was required for the requested local fast-forward.

Retained infrastructure: GPL-3.0, pinned Qwen/OASST source metadata, bounded verified
HTTP acquisition, content hashes, grouped splitting principles, immutable configs,
independent phase accounting, failure preservation, CLI/package identity and pinned
dependency tooling. These were adapted to native stages and sharded full-state IO.

Replaced infrastructure: DFlash/DCore runtime/vendor packages, AR verification and
known-answer anchors, independent 4/8/15 codecs, filtered drafter-only checkpoints,
academic registries/matrices/reporting, finance/evaluator modules, and old setup/CI
commands. Their notices/history remain on the archive. No former code is relabeled
as a native generator. Legacy caches, drafter/codec weights and recovery schemas fail
closed and require their archived implementation; no compatibility migration is claimed.

The new design uses shared continuous codecs and a full-Qwen embedded hierarchy.
Initial engineering choices include interval-overlap resize, integer-center RoPE,
two updates sharing group projections, 0.1 gates, small MLPs, train-only scale floors,
and tiled FP32-moment AdamW without master weights. These are not proven optimal or
properties established for an original NCM. Deliberate diagnostic deviations are
tiny width/canvas and software data; a separate full-geometry tiny-width test uses
all 32 updates and 1024 slots. Real 8B training remains a distinct unexecuted gate.

See native_validation.json for actual new test/run evidence. Old 127/170 counts,
drafter acceptance, speed assumptions, and prior publication status are not native
evidence. A failed initial fresh-process tokenizer check is retained; the correct
BatchEncoding input_ids interface is now shared by preparation and deployment.
