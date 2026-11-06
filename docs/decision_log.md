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

## 2026-11-04 — Separate Stage 1 codec/generator experiment

Add a manifest-bound Stage 1 workflow alongside the existing pilots. Keep their
two-step codec and legacy combined generator selection unchanged. Stage 1 uses a
1,000-update FP32 codec, independently configured BF16 teacher, FP32 cached targets
and a 500-update BF16 full-backbone generator. Its versioned generator selection
uses mean per-example dev text CE; auxiliary loss remains a training objective and
separate diagnostic. Initial measurements never select an untrained checkpoint.

Retain equal actual microbatch-objective reduction, including partial groups,
rather than silently switching to a global-token or global-example objective.
Measure initialization and all 31 codec cutoff assignments with a fixed plan and
preserved RNG. Compare combined/full/coarse/head components with explicit numerical
tolerances. Finite non-improvement seals trained artifacts but blocks dependencies
for review; a separately recorded diagnostic override cannot excuse invalid data,
nonfinite states or corrupt artifacts. Report train-only scales and raw variance.

Qualify complete EOS-inclusive references before teacher allocation. Connect whole
conversation/duplicate groups before partitioning and filtering, keep source splits
and dev/test IDs independent of training seed, and store locked test records under
a schema refused by training writers. The bounded approved OASST scan targets
2,000/200/200 pairs without automatic source substitution or quota changes.

Use synchronous owned children and exact manifest-owned receipts/checkpoints.
Disable all Stage 1 execution clocks explicitly, including sample/controller
deadlines, while retaining elapsed observations, cancellation and finite work,
storage, attempts and cumulative tensor/checkpoint I/O. Derive allowances after
qualification from actual counts and meta parameter inventories; distinguish
resident disk, cumulative writes and baseline memory from unmeasured activations.
An explicit pre-test total-step extension preserves prior work and charges.

Recover a complete native boundary from the exact recorded child launch even if
the controller was interrupted before copying that pointer into manifest state.
Validate launch ownership and scientific context before binding it; never search
global runs or repeat already completed updates to compensate for missing controller
bookkeeping. Locked-test access through optional AR inference also closes the
training-extension window.

Keep optional original-Qwen AR evaluation separate, opt-in and incapable of
repairing NCM outputs. Isolate the 32-pair overfit diagnostic. No real research
assets, model training, sweep or server is started during this implementation.
Stage 1 is not covered by historical source-transition recovery receipts.

Actual fixture execution of the literal README block exposed Windows long-path
failures in nested cache/checkpoint/export paths. Normalize actual Windows I/O
through extended paths while preserving ordinary serialized identities and the
existing bounded deletion/ownership checks. Preserve failed attempts. Condense the
README into an explained command guide, retaining the exact combined block and
moving scientific/operator detail to [Stage 1](stage1.md).

## 2026-11-04 — Initialize the shared OASST SQLite index before source scanning

Source inspection confirms Stage 1 opens its native-run index before either
artifact writer creates `artifacts`. Create only the supplied index parent at the
shared reader boundary, using the existing Windows I/O path helper, before SQLite
connects. Report directory creation and connection failures with their intended
path and chained original cause. Keep existing schema-collision rejection, SQL,
transactions, source filters, scan bounds, partitions and scientific settings.

Protect all connection-owned setup, ingestion and queries with explicit closing.
Both Stage 1 and legacy preparation own and close their reader generators, including
when a consumer breaks or raises. No attempt/index reuse, source compatibility
bypass, data migration or architectural deviation is introduced. Historical failed
runs remain untouched; the operator should start a new experiment using the
unchanged README combined block after installing the fix.

Add local Parquet/SQLite lifecycle regressions, legacy early-stop cleanup and actual
Stage 1 prepare-only publication/failure-path tests. Test and lint execution were
explicitly waived by the user because this system has no configured development
Python environment; these new checks and the existing suite remain unexecuted.
No dependencies were installed, research assets downloaded or real training run.

## 2026-11-04 — Remove timer-only Stage 1 console messages

Remove only the 15-second liveness print and its `last_message` bookkeeping in
`stage1.workflow.execute_child`, shared by all six stages and individual wrappers.
Keep output offsets, raw worker forwarding and final drain, initial settings/log
locations, polling/backoff, resource checks, owned cleanup, exit/checkpoint capture
and result handling unchanged. Training-step reports, elapsed observations and
monitoring cadences remain untouched. Waiting alone is not progress evidence.

The full-package source hash changes even for this console-only correction.
Stage 1 checks that hash before admitting an existing experiment; historical
clock/pilot compatibility receipts do not authorize this transition. Preserve
these safeguards and old manifests. Existing work stays on its exact matching
checkout/environment; do not interrupt or discard it to change output. No new
migration policy, scientific change, automatic restart or instrumentation is added.

Verify silence with mocked long clock advances, real output/duplicate forwarding,
partial reads/trailing errors, failure/review outcomes, cleanup, non-time caps and
identity rejection. Reuse the full suite for actual tiny backend, wrapper/all-route,
monitoring and SQLite coverage. Record actual results separately; no research
downloads, full-model training or active-experiment interruptions are authorized.

Actual checks: targeted suite 44 passed; full suite 459 passed with one CUDA skip;
Ruff passed. The current run also executes the earlier SQLite tests whose execution
was waived at their original delivery. See
[console validation](timer_status_validation_2026-11-04.json). An AST comparison
confirms that no other implementation code changed, and README bytes match the base.

## 2026-11-05 — Fixed-target unattended Stage 2

Keep the 64-position geometry and Stage 1 objectives, selection and precision.
Use 16,000 distinct training pairs, 200 dev pairs and 200 locked test pairs,
with fixed 1,000/4,000 codec/generator updates. The single launcher keeps the
codec gate and locked final evaluation; no intermediate review, override or
step extension is part of Stage 2. A full run launches chat only after all
training and evaluation children have exited and their results are verified.

Keep pinned OASST as the default with the same group partition and exclude
the standard Stage 1 locked test's groups from Stage 2 test selection. Offer
the pinned Tülu short-answer slice as an explicit alternative rather than
automatically changing the training source. Keep the 1,024-position Tülu wave
and its distinct codec quota separate. Data qualification, finite work and
resource gates remain in force. Existing manifests still require the exact
implementation fingerprint; no historical attempts were changed.

## 2026-11-06 — Stage 3 research-candidate workflow

Added a separately versioned Tülu/256-position experiment with offline sequence
distillation, final post-distillation qualification, update-indexed LR warmup and
cosine decay, and separate selected-weight/recovery cadences. Stage 1/2 defaults,
constant-rate numerics and codec gates remain unchanged. Stage 3 finite codec
non-improvement is advisory; invalid artifacts remain fatal. No extension of an
existing schedule horizon or cross-experiment recovery bypass is permitted.

The default target/minimum of 100,000/90,000 training pairs, 2,000 dev/test,
25/35/40 accepted-target length shares and candidate cap are operational policies,
not measured optima. No approved component is assumed authoritative by default;
retaining an authoritative reference requires explicit component metadata, an
allowlisted component and the supported normalized-exact validation rule.
Locked test is excluded from distillation and selection. Final student/AR models
load serially, with frozen-contract outputs cached for retries.

Selected snapshots must never displace the two recent complete recoveries or the
selected-weight dependencies those recoveries need. Logical SQLite/JSON write
charges use a declared conservative page/journal allowance and are reported
separately from physical-device I/O. Conservative retained-disk estimates remain
upper allowances, not claims about fit. No source/model download, production
teacher generation, full-Qwen training, benchmark sweep or provisioning occurred.

The final retry regression additionally requires a fresh reference model's
latency warmup before its first uncached example, preserves committed outputs,
and creates the new attempt's artifact directory when student evaluation is
cached. Historical recipes reject injected Stage 3 policies; Stage 3 custom
configs reject wrong stage roles, missing fixed selection contracts, objective
changes, and backbone-freezing warmup before model allocation. The complete
final-source suite passed (498 passed, one CUDA skip), with a separate tiny CUDA
pass. See the dated validation receipt for the fixture-only scope and earlier
failures and corrections.
