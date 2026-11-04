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

## 2026-11-03 — Named 2/100-step pilots and export-only chat

Keep the old two-step config as a compatible entry point and add named two/100-step
configs with separate exports. The 100-step target is total, preserving existing
data/codec, phase spending, optimizer and full-backbone scope. Evaluate/save/sample
at 25-step boundaries, with unchanged held-out selection. Chat uses that selected
export, which may precede the last optimizer update. No scientific math changes.

Run training in a separate child and wait for sealed objective completion/handoff
before loading one inference model. No training execution clock is reintroduced.
Use existing open-source Werkzeug for a loopback-only, bounded UI; preview has no
model imports/inference and cannot masquerade as model output. Preserve prompt-only
generation, native parallel argmax/EOS, request accounting and text-only rendering.

Keep the original clock receipt immutable; append one exact reviewed source edge
from 0c9fdd5 with an explicit pilot-chat-v1 option and strict identity/shard/state
checks. Genuine tiny pre-edit checkpoints validate compatibility; no unavailable
Spark checkpoint, full-model 100-step run or quality/speed result is inferred.

## 2026-11-04 — Independent canvas pilots

Centralize five answer capacities (8/32/64/256/1024), retain pinned Qwen3-8B,
width 384 and all 32 updates, and default new named pilots to medium/64. Preserve
explicit 1024 schedules/exports and deliberate legacy/Tulu workflows. Repeated
resolutions execute separate refinements. Do not implement curriculum or cross-size
checkpoint transfer under a canvas flag.

Build data/teacher/codec/targets in separate owned local children. Qualification
requires complete references including EOS and usable stable whole-group train/dev
splits before teacher loading. Keep the codec's two-step objective separate from
100 total generator updates. Namespace convenience pointers by full experiment
fingerprint, publish only sealed successes, guard writers and retain phase charges.
Pilot clocks stay disabled without an outer execution deadline; non-time safeguards
and elapsed observations remain. No-chat and planning avoid optional chat startup.

This scientific source change is not covered by the historical clock/chat recovery
receipts. Leave them immutable and reject older-source training recovery honestly;
no narrowly reviewed legacy transition is claimed. Same-source recovery restores
architecture and immutable inputs, with normal complete optimizer/RNG/identity
checks. Exact saved inference geometry remains independent of fresh-run defaults.
Use bounded tiny software tests and meta parameter inspection; real-data capacity,
all-size Spark behavior, quality, speed and convergence remain unverified.
