# Real pilot without clock deadlines

`Budget.enforce_wall_time_limits` is a strict Boolean, default **true**. The shared
`configs/budgets/real_pilot.yaml` explicitly sets it to **false**. Both the run
deadline and phase wall admission/reservation limit are disabled. The generator
still declares **two optimizer steps**. This is not a larger timeout or unlimited
work: model/data, full-backbone trainability, loss, optimizer precision, memory,
work/storage caps and integrity checks are unchanged, including the Spark fix.

The existing 1,200/10,800-second numeric settings remain validated, preserved and
**inactive**. `phase_id` remains `native-real-pilot-v1`; spending is not reset.

## Scope and accounting

The shared budget is referenced by `configs/data/real_pilot.yaml` and validation
configs `real_pilot.yaml`, `teacher_features_pilot.yaml`, `codec_pilot.yaml`,
`latent_targets_pilot.yaml` and `generator_pilot.yaml`. Native data, extraction,
codec, targets and generator operations inherit the policy. The standalone
generator has no other internal execution deadline through loading, validation,
checkpoint hashing/writing, export or handoff publication.

The separate `spark_8b_profile.yaml`, smoke/CI jobs, larger configs and Tülu waves
retain clocks. Waves have outer wave/child deadlines; resolution rejects
disabled-clock children rather than hiding those limits. Run this pilot through
its standalone CLI. Asset acquisition retains its separate finite byte/disk/time
budget and is not an automatic part of generator resume. Monitoring refresh/view
lifetimes and optional sample timeouts remain auxiliary controls, without stopping
required training/finalization. An external scheduler or shell may impose its own cap.

Elapsed time and throughput remain observed. Disabled attempts reserve **zero**
seconds, admit previously over-spent phase clocks and settle actual elapsed time
on close. Enabled attempts retain finite reservations. Settlement subtracts only
the reservation actually made; repeated close is idempotent. Work and disk checks
remain active.

Old ledger `caps`, `wall`, `charges` and run history remain readable and retained.
The Boolean is not an immutable numeric phase cap, so policy-only changes need no
cap migration. Additive `wall_attempts` record policy, actual reservation, closed
elapsed duration and open/closed status. The legacy combined balance is retained
separately because its actual/reserved split cannot be reconstructed.

An abrupt disabled-clock crash leaves an **open, unknown-duration** entry, not a
completed zero-duration observation. Later disabled attempts may resume. Finite
admission fails while such an entry is unresolved: it cannot claim the full phase
duration is known. No duration guess, ledger reset or reservation refund occurs.
Handled cancellation remains `cancelled` and settles its observed elapsed time.

## Recovery procedure

Reviewed predecessor: commit `c3032129d59e4a9c2b3d2414bf66f1c805202521`, package
SHA256 `9404148f14f0496952e919c77119119b3c32e90b0c74b3947847f0cdce111ab6`.
The exact successor and per-file/source-manifest review are recorded in
[`configs/recovery/clock_policy_v1.json`](../configs/recovery/clock_policy_v1.json).
The receipt is outside package Python to avoid a self-referential hash.

`--recovery-compatibility clock-policy-v1` explicitly permits only that recorded
transition. The implementation pins its predecessor and receipt ID and requires
the current full-package hash to match the approved successor. A later source edit
requires a new reviewed decision, not automatic approval. There is no arbitrary
receipt path, hash override, skipped integrity or fresh-weight fallback.

All shards/pointers are verified. Original metadata, manifest identity and payload
identity must agree. Expected metadata must match after **only** the implementation
transition: model, architecture, source/data/codec, dtype, training contract,
numerical policy, runtime and phase remain exact. Optimizer groups/moments/private
RNG, constant scheduler/scaler, global RNG, cursor and counters use normal strict
restore. Original checkpoints are never rewritten. New attempt events record the
parent manifest/hash, predecessor/successor, receipt/hash and reason, including old
selected checkpoints. Retain the explicit option on future resumes if the selected
checkpoint still belongs to that predecessor.

The reported Spark checkpoint is **not present locally**. Its actual implementation
identity is unknown and recovery from it is untested. From the repository root in
the existing training environment:

```sh
git switch main
git pull --ff-only origin main
ncm preflight --config configs/validation/generator_pilot.yaml
ncm generator train --config configs/validation/generator_pilot.yaml --resume runs/generator/20261103T145320Z-ea07d0aaaa/checkpoints/current.json --recovery-compatibility clock-policy-v1
```

Resume succeeds only if the checkpoint matches the recorded predecessor (or the
current implementation) and every other contract passes. Unknown predecessors
fail with an explanation. If installation is not editable, reinstall this checkout
without replacing dependencies: `python -m pip install --no-deps -e .`. Preserve
the validated CUDA environment, original checkpoints, handoffs, datasets and ledger.

## Monitoring and completion

`resources/budget_wall_seconds_used` continues increasing. Remaining time is `null`
with `disabled_by_policy` status; TensorBoard shows status text without a countdown
scalar. `resources/enforce_wall_time_limits` is zero. Started events explain:
“Wall-clock limits: disabled; completion is governed by the configured work target.”
Preflight, resolved config, launch/attempt records and summary expose this policy.

`budgets/phase/wall_seconds_charged_and_reserved` is the known combined balance.
`budgets/phase/wall_seconds_reserved` means active finite reservations only and is
unavailable where a legacy balance prevents an exact split. Phase diagnostics
explicitly identify incomplete open disabled-clock elapsed accounting.

Success requires actual objective completion, held-out validation/selection,
durable recoverable state, complete independent export and publication of the
hash-bound `runs/handoffs/generator.json`. Inspect `status: completed`,
`completion.code: objective_attained`, actual steps, checkpoint/artifact paths and
the handoff. A target-boundary resume revalidates and writes recovery/export
without repeating optimizer updates. Work/storage, memory, integrity, numerical
errors and user interruption can still stop a run; none is relabeled completed.

## Implementation and validation

| Files | Purpose |
|---|---|
| `runtime/budget.py`, shared pilot budget | Policy, reservation/settlement, legacy/crash accounting |
| `runtime/provenance.py`, `runtime/doctor.py` | Durable policy and inactive numeric settings |
| `training/engine.py` | Clock metrics, strict resume integration, at-target finalization |
| `training/recovery_compatibility.py`, receipt | Exact approved source transition |
| `cli.py`, `runtime/runner.py` | Explicit option through existing resume command |
| `config/waves.py` | Reject hidden outer-clock conflicts |
| New clock-policy unit/integration tests | Fake clocks, safeguards, exact recovery, cancellation, TensorBoard |
| README, training/Spark/monitoring guides, status/contributions, AGENTS | Procedure and scoped policy exception |

Exact executed commands, results and preserved development failures are in the
[dated validation receipt](pilot_clock_validation_2026-11-03.json). To repeat the
software checks in the existing development environment:

```sh
python -m pytest -q tests/unit/test_clock_policy.py tests/integration/test_pilot_clock_completion.py
python -m pytest -q
ruff check .
```

Final full-suite result: **255 passed, one CUDA skip**, in 265.11 seconds. Ruff and
whitespace checks passed. The earlier sandbox-restricted failures are retained in
the receipt alongside the passing normal-local rerun with the same source.

Tiny BF16 checkpoints were captured under the genuine predecessor **before source edits**;
partial and at-target recovery preserved exact weights, optimizer moments/private
RNG, global RNG, sampler/cursor and counters. Repeatable synthetic-transition tests
also check positive/negative cases. These are CPU software fixtures. No real model,
corpus download, Spark/GPU training or reported-checkpoint execution occurred; no
fit, convergence, quality or speed conclusion follows.
