# Native training and monitoring corrections — 2026-10-29

These are additive corrections to reviewed revision
`3072d4546024bf4616b1aa85408e3654d7ae589b`. The pinned Qwen backbone, 36-block
placement, 32 sequential hierarchy updates, 1024-position canvas, shared width-384
codec, sixteen cumulative auxiliary losses, prompt-only forward API, original
residuals, and inference decoding rule are retained. No drafter, verifier, repair,
LoRA, new benchmark or formal study is introduced.

The archive reference remains `archive/hierarchical-drafter` at
`c5e5329172e63026995a440aa0591a5e15804ab4`. Historical runs and the earlier validation
receipts are retained. New evidence is in
[native_corrections_validation.json](native_corrections_validation.json).

| Item | Finding and implementation decision | Primary files | Regression / observed focused result | Remaining limit |
|---|---|---|---|---|
| C1 | Shipped OASST configs selected LICENSE. Bind the actual locked train Parquet/SHA; validate ID, canonical path, revision, split and format before hashing/tokenizing/parsing. Explicit-hash fixture/JSONL routes remain. | `data/prepare.py`, two `configs/data` files | Shipped config/lock checks, early license rejection and actual tiny Parquet joins/filtering/grouped preparation passed. | Software Parquet fixture; real Qwen data fitting not run. |
| C2 | Direct nearest BF16 writes lost `1e-5` updates. Use tiled completed FP32 arithmetic, FP32 moments and adjacent-value stochastic BF16 write-back; private checkpointed RNG, no master copy. Restore nonpersistent RoPE buffers in the declared dtype. | `training/precision.py`, `optimizer.py`, `checkpoints.py`, `engine.py`, `embedded/model.py`, typed configs | Old cancellation reproduced locally; probabilities, exact endpoints, signed/subnormal cases, FP32 reference parity, BF16 codec and exact generator/warmup recovery passed. | Same-device/environment exact continuation tested on CPU; Spark kernel/fit/convergence unmeasured. |
| C3 | Dev examples repeatedly used index 27 (501). Bind a fixed stratified index plan and independent per-example head RNG. | `training/validation_plan.py`, `losses.py`, `engine.py` | Determinism, small-set range coverage, all 31 coarse indices when allowance permits, preserved RNG and independently reconstructed selection scores passed. | Tiny sets explicitly report incomplete coverage. |
| C4 | Last warmup step used next-update phase/counts. Capture executed phase/counts before accumulation; flush at transition; checkpoint phase prepares the next update. | `training/engine.py` | One-step/accumulation/long-cadence boundaries, end-at-boundary, BF16 save/resume and monitoring invariance passed. | Prepared full trainability is not evidence a full update occurred. |
| C5 | Custom replay aggregate accounting included unrelated siblings. Count only recognized derived projection/view roots, including owned server logs for normal sessions; check free space separately. | `monitoring/projector.py` | Large unrelated file ignored; genuine multiple views, individual/aggregate exhaustion, headroom and unchanged canonical bytes passed. | Cooperative filesystem caps; queued work can finish before the next check. |
| C6 | Missing and incomplete replay could report success. Validate declared source/identity, capture byte extents, expose complete/empty/incomplete/follow-stopped/invalid-source outcomes and propagate them through CLI. | `monitoring/projector.py`, `cli.py` | Missing/empty/complete sources, forced deadline/backlog, partial-line completion exactly once, corruption, overwrite rejection and live/HTTP discovery passed focused tests. | Snapshot completion covers captured bytes, not later appends. |
| C7 | Per-tile host scalar conversions synchronized the device. Reduce finite flags/norms and prospective-arithmetic checks on device; transfer bounded aggregates and reuse the report for clipping/update. | `training/precision.py`, `optimizer.py`, `engine.py`, `monitoring/diagnostics.py` | Late nonfinite rejection leaves all weights/moments/RNG untouched; scalar extraction count does not grow with tile count. | Bounded diagnostic synchronizations remain; no measured CUDA speed claim. |
| C8 | Inference used eight positions/call and transferred each chunk. Give inference a typed 128-position default and declared allocation cap; keep IDs on device until one final transfer. | `inference/policy.py`, `export.py`, generation/sample/CLI paths | Controlled-margin IDs/EOS/truncation agree across chunks; partial/full chunks, memory guard, single ID transfer and 36 logical block calls passed. | BF16 near-tie outputs may differ with GEMM shapes; GPU speed effect unmeasured. |

## Numerical and recovery contract

The selected policy is `fp32_moments_bf16_stochastic_v1`; the optimizer state schema
is `fp32_adamw_bf16_sr_v2`. FP32 diagnostics retain a conventional deterministic
FP32 write path. BF16 nearest write-back is not an available fallback. Configs bind
`optimizer_policy`, `optimizer_seed` (1729 by default), and
`optimizer_tile_elements` (1,048,576 by default).

For each bounded tile, the optimizer computes new FP32 moments, bias-corrected
AdamW/epsilon and decoupled decay in FP32. It rounds only the completed value. For
adjacent finite BF16 endpoints `a <= x <= b`, the upper choice has probability
`(x-a)/(b-a)`. Exactly representable values stay fixed. Exact signed zero keeps its
sign; underflow toward zero keeps the input sign. Nonfinite/unsupported overflow
is rejected, without clamping. A dedicated `torch.Generator` supplies independent
per-element draws and is separate from all training/validation/monitoring RNGs.

No-gradient/frozen parameters acquire no moments, decay or random draws. The
checkpoint validates numerical policy, seed, Torch RNG algorithm/device, exact RNG
state, draw count, traversal identity, parameter groups, tile size, moments and
per-parameter steps. Tile-size/order changes fail closed. This is not cross-device
bitwise reproducibility. Restricted checksummed shards stream into one destination;
no persistent FP32 master, giant state bytes blob or second full model is added.

Gradient flags and float64 norm reductions stay on device across tiles, with one
small aggregate transfer before clipping. The same validated report is reused by
the optimizer; changed gradients invalidate it. A bounded prospective-arithmetic
pass checks all tiles before committing any parameter/moment/RNG mutation. A second
bounded pass commits the update. This trades arithmetic for pre-update safety.
Other runtime failures fail the attempt; recovery uses the last complete checkpoint.

Actual optimizer diagnostics separately report intended FP32 update L2, stored
parameter change L2 and active scalar count. Stochastic rounding can leave individual
scalars unchanged for several steps. The older eight-element representative flags
are debugging observations, not acceptance criteria for BF16 effectiveness.

Old nearest-write recovery checkpoints cannot resume under this policy. The error
is explicit; files, spending and historical selection observations remain intact.
Ordinary compatible inference exports do not load an optimizer and remain usable.
Teacher/data artifacts can be reused when their content/semantic identities match.
Old codec selection scores are historical; they must not be compared with the new
validation plan as if the criterion were unchanged. Frozen codec loading honors its
declared trained dtype; an intentional target-cache dtype cast is recorded separately.

## Validation and phase policy

Codec selection uses `codec_stratified_coarse_indices_v1`. It fixes held-out example
IDs, index/resolution assignments and per-example head seeds before training. At
least 31 admitted dev examples cover all 31 coarse indices; smaller allowances span
coarse/middle/fine indices as far as possible and report actual coverage. Repeated
tiny resolutions keep distinct indices. One dev example establishes one cutoff only.

The training sampling/objective is unchanged. Validation performs full reconstruction,
one planned coarse reconstruction and bounded sampled head KL per example. Its score
remains the equal mean of actual per-example weighted objectives. Metadata binds the
plan/policy/aggregation; events retain components reproducing the selection score.
Count-weighted data summaries stay a separate monitoring view. No dev statistics fit
normalization or reference means.

Each accumulation window captures its executed phase and trainability counts.
Crossing warmup does not unfreeze partway through the window. The completed warmup
metrics are emitted/flushed as warmup, followed by a transition with previous/next
phase, last completed step and `effective_from_step`. Only then is trainability
prepared for the next update. A checkpoint may contain `phase=full` while
`last_executed_phase=warmup`; that does not claim a full optimizer step.

## Memory, inference and continuation

Meta-device construction counts the actual production modules:

| Inventory | Parameters |
|---|---:|
| Original Qwen | 8,190,735,360 |
| Native generator | 8,250,600,480 |
| Shared codec | 6,901,504 |

The native BF16 weight/gradient + FP32 moment lower bound is **99,007,205,760 bytes**.
Recovery weights/moments excluding gradients are **82,506,004,800 bytes**. A full FP32
master would bring the first subtotal to **132,009,607,680 bytes** and is not used.
The corrected policy adds tile-bounded scratch (conservative estimate 160 bytes per
tile element) and small RNG state. Activations, buffers, kernels, allocator and OS
still need memory. These are estimates, not Spark feasibility measurements. Spark
host/device memory is one pool. Preflight retains the previous/selected/new checkpoint
versions; four recovery versions alone are about 330.02 GB before metadata.

Inference has its own `inference.head_chunk_positions` (128) and
`inference.max_head_temp_bytes` (268,435,456). The resolved chunk is capped at the
actual canvas. It accounts for two logits-sized buffers and full-canvas int64 IDs;
backend GEMM workspace remains an additional measured hardware requirement. Training
CE/codec KL chunk limits remain eight and are independent. For example:

```sh
ncm generate --checkpoint /ABS/EXPORT/manifest.json --prompt-file /ABS/prompt.txt --head-chunk-positions 256 --max-head-temp-bytes 536870912
```

Monitoring samples use the same inference settings, preserve training state/RNG and
record the resolved projection/memory policy with actual timing. The full canvas
still executes before EOS trimming; there is no autoregression or answer repair.

Bounded software continuation:

```sh
python -m pytest -q tests/unit/test_optimizer_precision.py tests/integration/test_optimizer_recovery.py
ncm monitor smoke --config configs/monitoring/smoke.yaml
ncm prepare --config configs/validation/real_pilot.yaml --plan-only
```

On suitable hardware with already prepared pinned assets and exact native handoffs:

```sh
ncm doctor --profile spark
ncm profile train --config configs/validation/spark_8b_profile.yaml
```

That profile remains a finite 3600-second / three-full-step, complete recovery probe.
It now selects the corrected stochastic policy explicitly. No real-data codec,
actual 8B optimizer/recovery, CUDA/Spark throughput or model quality claim has been
established here. No 10M/100M training, model download or provisioning was launched.

## Numerical references

[Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192) grounds the diagnosis
of cancellation under nearest-rounded writes. The adjacent-probability implementation
here is independently written and tested. [Ozkara et al., AISTATS 2025](https://proceedings.mlr.press/v258/ozkara25b.html)
provides related research context. Its [official optimizer](https://github.com/amazon-science/adamw-sr)
uses different optimizer-state precision/shared randomness; this repository does not
claim to reproduce that optimizer or its performance, import its frameworks, or
increase learning rates to conceal lost updates. No third-party optimizer source was
copied; existing project/license notices are preserved.

## Recovery tensor-lifetime follow-up — 2026-10-29

Revision `792ba52` retained the completed `GradientReport` in the training frame.
Its active parameter list strongly owned the old weights and their gradients even
after the probe deleted `loss`, `model` and `optimizer`. Garbage collection and
`empty_cache()` cannot release tensors that still have a live owner.

The narrow repair in `training/engine.py` is `del report` immediately after the
successful `optimizer.step(report)`. Finiteness validation, stale-report checks,
diagnostics, clipping and the validated update still finish first. The existing
loss deletion and one-destination restoration remain intact. The report API,
optimizer precision/state schema, model topology, RNG behavior and failure semantics
are unchanged. An optimizer error still fails the attempt.

The new production integration test captures only weak references after the
successful second full-training step. Before calling the replacement constructor,
it asserts that old parameters, gradients and FP32 moments are all gone; it checks
again on the replacement model's forward. The observer neither clears references
nor runs garbage collection. Counts and bytes are derived from the fixture.

Against the original behavior, the BF16 test failed before replacement allocation:
578 parameters and 578 gradients remained live; old moments were already released.
With the fix, all three live counts are zero before allocation and during every
observed replacement forward, in BF16 and FP32. Both three-step probes complete.
Terminal recovery checkpoints match uninterrupted runs exactly for weights,
moments, steps, private optimizer RNG/draw count, sampler/cursor, recorded global
RNG and third-step loss, allowing intentional run/attempt provenance differences.

Numerical continuation alone did not prove release; this ownership check adds that
missing invariant. It does not require process RSS or cached/reserved memory to
return to zero and does not establish full-8B/Spark feasibility. Recovery remains
bound to the implementation hash, so pre-fix checkpoints are not automatically
compatible with changed source. Prior receipts and the failed pre-fix run are
preserved. See the separate
[dated lifetime validation receipt](recovery_lifetime_validation_2026-10-29.json)
for commands, run IDs, counts, bytes, failures/skips and current gates.
