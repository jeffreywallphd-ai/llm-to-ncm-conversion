# Native decisions — 2026-10-29

The user explicitly replaced the unchanged-target drafter goal with a native
embedded Qwen hierarchy. Preserve the old work at a verified archive branch before
removing active drafter/study code. Do not reconstruct an academic study on main.

Use the specified 8/16/12 block allocation, 32 sequential updates, 1024-position
canvas, shared width-384 codec, sixteen cumulative losses, and full fine tuning.
Use fixed interval-overlap geometry and integer-center positions; no gold length
or target code enters the student. Keep longer skips disabled. Initial gates are
nonzero and group projections are shared only within successive pairs.

Teacher states use P-1+t, including the first answer token and genuine EOS. Cache
only valid states/fine codes with content/geometry/precision identity. Larger target
preparation is a separate streaming process, never a generator-side teacher fallback.
Normalize using training-only statistics in raw encoder coordinates.

Keep exact checkpointed/chunked vocabulary losses. The original FP32-moment,
no-master decision used direct nearest BF16 writes; review reproduced cancelled
small updates, so that write-back choice is superseded. The supported correction
is `fp32_moments_bf16_stochastic_v1`: tiled completed FP32 AdamW arithmetic,
FP32 moments and a single adjacent-probability stochastic BF16 write-back, with
private checkpointed RNG and no persistent master. Two bounded passes preserve
pre-update rejection safety. This is not the external AdamW-SR implementation or
a convergence/performance claim. Publish complete bounded shards behind atomic
manifests, reject old recovery policy explicitly and retain historical evidence.

Report software fixtures, small-batch learning, real-data codec fitting, and actual
8B optimizer/recovery as distinct milestones. Preserve negative/failed attempts and
cumulative budget charges. No initial large run is launched to finish implementation.

Monitoring is additive and read-only: canonical strict JSONL is projected by separate
spawned CPU-only TensorBoard processes. Each invocation/resume has a distinct attempt;
older checkpoint recovery retains abandoned history and aggregate charges. Losses,
optimizer and checkpoint selection remain authoritative. A short diagnostic now
validates at its declared final step if it ends before the ordinary ten-step cadence.
TensorBoard 2.20 requires pinned pkg_resources-compatible setuptools; Windows nested
event IO uses extended paths. Histograms remain disabled and explicit enabling is
rejected; no all-parameter dump is supplied. GPU/Spark overhead is unmeasured.

Codec selection now binds fixed stratified coarse-index assignments rather than
the per-example Random(0) reset that selected index 27/resolution 501. Training
objective/sampling stays unchanged. A completed warmup window is recorded before
unfreezing for the next update. Replay captures extents, exposes partial/deadline
outcomes and counts recognized derived views instead of unrelated siblings.
Inference gets its own typed head allocation/chunk policy and one final ID transfer;
training head limits remain independent. BF16 resume also restores nonpersistent
RoPE buffers in their declared dtype. See the C1–C8 correction report/receipt.

## 2026-10-29 — Tulu extension and complete wave

- Keep the historical immediate-parent OASST pilot unchanged; recommend a separately pinned complete-context Tulu extension. No real corpus/weight acquisition or real training is part of implementation.
- Use langid 1.1.6 and its hash-bound embedded CPU classifier, with explicit uncertain exclusions and code/math evidence policy. This is a heuristic; unknown component origins/terms remain unknown.
- Use disk-backed exact dedup/connected groups, seed-17 group split and source/length-aware sampling without replacement. Dev does not inherit training enrichment. Capacity is unknown until exact preparation, and the 10M training-only quota gates teacher allocation.
- Thin wave control invokes separate existing CLI stage processes, with per-wave snapshots/handoffs, persistent phase ledgers and explicit native recovery. Preprocessing retries restart and remain charged. Controller waiting is excluded from overhead accounting.
- Explicit objective completion forces terminal validation/recovery/export; resource and safety stops remain incomplete. Preserve stochastic BF16 policy, private RNG and the completed-gradient-report lifetime fix.
- Engineering token/work/storage ceilings are declared in extension configs, not automatically increased or treated as performance forecasts. No architecture deviation or formal benchmark study was introduced.

## 2026-11-02 — Spark availability accounting

Use Linux MemAvailable for raw host availability instead of immediately free pages.
Keep byte-accurate host totals/free/availability separate from conservative visible
cgroup-v2 limit-minus-current headroom and the selected CUDA device total. Never
add shared pools, swap/cache estimates or a CUDA free-memory ceiling. Preserve zero,
unknown capacity, the existing zero RAM reserve, all baseline inventory and disk
guards, and distinguish unsupported discovery from a known unreadable finite limit.

Keep parsing/probing in one standard-library module and numeric metric projection
separate from structured diagnostics. A failed integration run exposed the old
metric-only assumption; preserve that evidence and test the corrected consumer.
The optional verified-hash reuse is a separate I/O change, not a memory-capacity
claim. No architecture, dataset, optimizer, loss, schedule or resource-cap deviation
was introduced; real Spark/full-model peak validation remains unexecuted.

## 2026-11-03 — Pilot completion without clock termination

The user explicitly requested disabling both run and aggregate-phase wall limits
for the shared real pilot, superseding the blanket clock-budget wording in AGENTS.
Keep positive numeric clocks inactive, default other jobs to enforcement and keep
phase identity/numeric caps, work/storage, memory and integrity rules unchanged.
Zero wall reservation is not a known zero duration. Record each attempt policy and
actual settlement; unresolved disabled-clock elapsed accounting blocks a later
finite-clock admission. Preserve legacy combined spending without inventing its split.

Pin the c303212 predecessor and one exact reviewed successor in an explicit recovery
receipt; require the named CLI option, verify old self-identity/all shards and every
other contract. At-target resume completes validation/recovery/export without more
optimizer updates. No model math, optimizer, data or schedule deviation is introduced.
Retain larger-wave clocks and reject disabled children under their outer deadlines.
Genuine tiny BF16 checkpoints captured before edits provide software continuation
evidence only; the reported Spark checkpoint and real hardware remain untested.
