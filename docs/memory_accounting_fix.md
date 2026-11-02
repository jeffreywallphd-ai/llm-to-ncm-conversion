# Spark memory accounting correction — 2026-11-02

Starting checkout: `aa28fa605cc4030960bd4509e3375f0d56fbb798`. Its
`runtime/doctor.py::host_available()` used `SC_AVPHYS_PAGES * SC_PAGE_SIZE`, and
Spark admission took the smaller of that value and GPU zero's total. On
Linux/glibc that pages query uses `freeram`, rather than the kernel's reclaim-aware
`MemAvailable`. `runtime/memory.py` reused the same helper. Unknown preflight host
measurements were coerced to zero.

The supplied Spark observations are consistent with this error. They were not
simultaneous, so no exact reconciliation or attribution of all cache to NCM is
claimed. The reported attempts were blocked before loading or training the model.

## Changes

| File | Purpose |
|---|---|
| `src/ncm_lab/runtime/host_memory.py` | Standard-library snapshot, pure parsers and bounded read-only cgroup-v2 probe |
| `src/ncm_lab/runtime/doctor.py` | Shared host fields, current CUDA device, auditable capacity operands/margin/blockers |
| `src/ncm_lab/runtime/memory.py` | Same host definitions with separately scoped current/lifetime process RSS |
| `src/ncm_lab/training/engine.py` | Project only integer/unknown byte fields as metrics; structured diagnostics remain JSON metadata |
| `tests/unit/test_memory_accounting.py` | Synthetic host/cgroup/fallback/platform/visibility regressions |
| `tests/unit/test_spark_memory_preflight.py` | Baseline reproduction, safety, CLI, immutable decision and historical-reader coverage |
| `tests/integration/test_memory_monitoring.py` | Actual tiny trainer accepts structured memory diagnostics without changing numeric tags |
| `tests/conftest.py` | Synthetic procfs reader; new decision tests do not depend on real host/container capacity |

Each Linux report reads meminfo once, converts `kB` using 1024 and keeps
`MemTotal`, `MemFree` and `MemAvailable` separate. Cache/swap observations add no
capacity. Malformed required availability, duplicates and contradictions remain
unknown; unrelated optional fields do not invalidate valid availability. Zero is
zero. Missing `MemAvailable` can use an explicitly labeled conservative free-memory
fallback; failed procfs/sysconf calls return `null` when no usable measurement
exists. Windows keeps `GlobalMemoryStatusEx`; supported other platforms retain a
validated free-pages fallback. No psutil or dependency synchronization is required.

`host_available()` remains callable from the existing modules with its raw
integer-bytes-or-`None` contract. `host_available_bytes` retains that raw scope;
`host_effective_available_bytes` separately applies known process headroom.

The cgroup-v2 probe resolves `/proc/self/cgroup` against cgroup2 mount roots from
`/proc/self/mountinfo`, including escaped mount paths and visible ancestors. It
selects the widest matching view. Unmatched namespace-relative mappings require
proof of the current PID in the candidate's `cgroup.procs` before reading limits.
It never scans siblings or crosses the mount boundary. Reads and discovery are
bounded: 16 mounts, 64 ancestor levels and small scalar files.

Each finite `memory.max` is paired with **that group's** `memory.current`:
`max(limit - current, 0)`. The tightest visible headroom applies without reclaim
credit for charged cache. Unlimited leaves do not hide finite parents. Known
finite limits with unreadable usage, unreadable resolved constraints and truncated
ancestor probes block host-dependent real admission as indeterminate. Missing
controller files at the visible root do not fabricate zero or imply cgroup v1.
Unsupported/unresolved discovery retains the host estimate with explicit assessment
limitations, never a claim of verified unlimited capacity. Hidden ancestors remain
unverified; new cgroup-v1 support is not claimed.

Spark uses the minimum of corrected host availability, readable applicable cgroup
headroom and the current selected CUDA device's **total**. It never sums host/device
capacity or imposes CUDA free memory as a new ceiling. Ordinary discrete-GPU
admission still uses device total. Existing CUDA/device validity, CPU authorization,
genuine insufficiency, fixture, disk/checkpoint, budget and artifact guards remain.
Unknown required capacity blocks real admission rather than becoming a fake zero.

The unchanged BF16 generator baseline is **99,007,205,760 bytes** for weights,
gradients and two FP32 moments. The RAM policy reserve remains **zero**. Tile scratch
stays visible separately; the signed `remaining_after_baseline_bytes` is not
guaranteed activation space. Model, precision, loss, schedule, trainability and
resource caps are unchanged. Implementation-hash recovery integrity remains
enforced; old checkpoints are not automatically compatible with changed source.

## Reproduction and verification

This rounded input is a **synthetic fixture**, not an alleged exact Spark dump:

| Quantity | Bytes |
|---|---:|
| Immediately free, old admission input | 55,796,248,576 |
| Reclaim-aware available, corrected input | 119,936,961,536 |
| Unchanged baseline | 99,007,205,760 |
| Signed baseline margin with sufficient mocked device total | 20,929,755,776 |

The isolated old-policy control rejects the baseline. Corrected admission selects
`proc_meminfo.MemAvailable` and passes this baseline check. Smaller device/cgroup
capacity, zero headroom, invalid/unknown measurements, CUDA failure, disk failure
and finite-limit/unknown-current cases still block. Tests assert actual capacity
and limiting source, not just an empty blocker list. Cgroup cache receives no credit
and CUDA free memory is not queried.

Doctor and final observations share definitions. A later close observation may
differ but cannot overwrite the saved decision snapshot. Existing JSON status names
and historical monitoring records remain compatible. Full-suite verification caught
the trainer's old numeric-only assumption; the byte-metric projection and dedicated
trainer regression address that consumer while retaining structured diagnostics.
The earlier failed run records/log are preserved.

```sh
python -m pytest -q tests/unit/test_memory_accounting.py tests/unit/test_spark_memory_preflight.py tests/unit/test_asset_hash_reuse.py
python -m pytest -q tests/integration/test_memory_monitoring.py
python -m pytest -q
ruff check .
git diff --check
```

Exact executable paths, timeouts, counts, exits, environment and CLI outputs are in
[the dated validation receipt](memory_accounting_validation_2026-11-02.json).
Final full-suite result: **214 passed, one skipped** in 237.57 seconds; the skip is
the tiny CUDA test on the existing CPU PyTorch environment. Ruff and whitespace
checks passed. Local doctor returned zero; both real-config, non-training
preflights returned the existing blocked exit status two for unavailable CUDA,
unknown device total and insufficient full-checkpoint disk space.
Tests use synthetic files/mocks and tiny software fixtures. The existing CPU suite
includes bounded tiny-model training/recovery; no real model, corpus download,
system tuning, cache dropping, stress allocation or CUDA installation occurred.

## Separate optional asset I/O optimization

`teacher/model.py::verify_cached_assets()` now reuses each verified SHA256 returned
by `verify_asset()` rather than rereading the asset through `digest()`.
`tests/unit/test_asset_hash_reuse.py` independently tests the identical mapping,
one verification/read per asset per invocation, and unchanged SHA256, Git-blob,
size, missing/corrupt-file rejection. Every invocation still verifies content;
there is no timestamp or persistent trust cache. This removes duplicate I/O,
creates no physical RAM, and does not imply repeated reads made duplicate cache
copies.

## Later operator checks

These existing commands are non-training; they were also exercised locally:

```sh
ncm doctor --profile spark
ncm preflight --config configs/validation/spark_8b_profile.yaml
ncm preflight --config configs/validation/generator_pilot.yaml
```

A `completed` preflight means baseline admission passed, not completed training.
Legitimate blockers retain the existing nonzero CLI status.

The separate **manual, independently authorized** hardware probe remains:

```sh
ncm profile train --config configs/validation/spark_8b_profile.yaml
```

Its current config consumes `runs/handoffs/targets.json`, uses three full optimizer
steps, microbatch one, no warmup and recovery after step two. The per-run wall cap
is 3,600 seconds under `native-real-pilot-v1`; persistent phase caps also apply.
**It may publish/replace `runs/handoffs/generator.json`.** Inspect/preserve any
existing handoff before explicitly running it. This task did not execute the probe
or alter any handoff/checkpoint/history.

No actual Spark/GB10, full-8B recovery, activation/allocator peak, quality or speed
evidence was produced. Meminfo and cgroup reads race concurrent workloads and
process migration; the report is not an allocation reservation. Passing baseline
admission does not establish full-training fit. Namespace visibility and
conservative charged-cache treatment remain explicit limitations.

References: [Linux meminfo](https://docs.kernel.org/filesystems/proc.html#meminfo),
[glibc free-pages implementation](https://github.com/bminor/glibc/blob/master/sysdeps/unix/sysv/linux/getsysstats.c),
[cgroup-v2](https://docs.kernel.org/admin-guide/cgroup-v2.html),
[mountinfo](https://man7.org/linux/man-pages/man5/proc_pid_mountinfo.5.html),
and [NVIDIA unified-memory guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture).
