# Native implementation status

| Milestone | Implementation | Actual evidence |
|---|---|---|
| 36 original blocks, 32 updates, 1024 canvas, 384-wide real spec | implemented | Tiny-width full geometry test; no real-8B claim |
| Frozen teacher alignment and shared progressive codec | implemented | Native CPU gradients/alignment/cache checks |
| Sixteen cumulative losses and full-backbone tuning | implemented | Tiny production-class optimizer updates |
| Complete sharded recovery and independent export | implemented | CPU deterministic recovery, corruption and interruption checks |
| Bounded fixed-batch codec/generator learning | implemented | See native_validation.json; fixture only |
| Pinned real-data codec fit | implemented path | Not executed: full Qwen assets/CUDA unavailable |
| Real 8B optimizer/recovery feasibility | implemented probe | Not executed: CUDA/Spark/assets/storage prerequisites |
| Downstream quality, throughput, speed improvement | unestablished | Not measured |

The current receipt and six workflow READMEs identify actual native run IDs and
limitations. This is an executable initial research implementation, not proof that
parallel generation retains the pretrained model's quality. Source/target quotas
can be unavailable. No financial, business, generalization, novelty, or publication
claim is made. Formal studies remain future extensions.

Read-only canonical/TensorBoard monitoring is implemented and CPU validated:
Historical revision `3072d45` passed 78 tests with 1 CUDA skip; actual event/HTTP/browser checks and a
connected installed-wheel smoke completed. See [monitoring validation](monitoring_validation.json).
Histogram collection is disabled/unsupported; no real-8B or matched overhead claim is made.

Current C1–C8 status: source binding, stochastic BF16 numerical recovery, fixed codec
validation coverage, executed-phase labels, replay storage/outcomes, aggregate
device checks and independent inference head policy are implemented. The correction
receipt records current full-suite outcomes and actual BF16 production-class CPU
codec/generator runs. Real-data fitting, full-8B recovery, Spark/CUDA allocation,
quality and speed remain unexecuted/unmeasured. See
[native corrections](native_corrections.md) and its [receipt](native_corrections_validation.json).

Final correction suite: 114 passed, 1 CUDA skip in 80.40 seconds;
installed source bytes match the wheel, smoke completed, and preparation plan started
no downloads. BF16 codec/generator recovery was executed on CPU fixtures.

## Tulu extension implementation

The dedicated pinned multi-shard route, complete-context rendering, disk-backed
source/language/overlap/group/selection policy, capacity gate, sequential wave CLI
and explicit objective completion are implemented. See
[operator guide](extension_training.md) and [dated software validation](tulu3_extension_validation_2026-10-29.json).

Real extension capacity remains **unknown**. No real Tulu scan, 10M-position codec
fit, 100M-token generator wave or Qwen weight download ran during implementation.
Genuine operational blockers remain pinned full assets/qualified data, suitable
validated CUDA/Spark hardware, and sufficient simultaneous cache/checkpoint/free
storage. Full-model fit, retained quality, throughput and speed remain unmeasured.
The historical OASST-only larger envelopes do not establish enough distinct data.

Final extension software verification: 143 passing tests, one CUDA skip, Ruff and
connected monitoring smoke passed. Local two-shard Tülu-shaped waves exercised
qualification, separate stage processes, terminal export, exact same-wave recovery,
retained preprocessing charges, corrupted-output rejection and Windows child-tree
cancellation. These are software fixtures, not measurements of released Tülu capacity.
