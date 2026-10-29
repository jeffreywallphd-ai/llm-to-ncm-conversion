# Spark/CUDA feasibility and continuation

The intended Spark profile has 128 GB shared physical memory. It is not the current
development machine. Keep an already validated CUDA PyTorch installation; install
the package without requesting the CPU extra:

```sh
pip install -e '.[upstream,prepare,monitoring]'
ncm doctor --profile spark
ncm prepare --config configs/validation/real_pilot.yaml --plan-only
```

Tested CPU dependencies are pinned in pyproject/uv.lock. ARM64/Blackwell SDPA and
the tiled optimizer remain unvalidated here. The stage configs identify Spark;
ordinary CUDA hosts can select their CUDA profile. Host offload is not an additional
independent memory pool on Spark. Available unified/device capacity must not be added.

Meta-device construction of the actual production classes counts 8,190,735,360
original and 8,250,600,480 native-generator parameters; the codec has 6,901,504.
BF16 generator parameters/gradients plus two FP32 moments require at least
99,007,205,760 bytes (99.01 decimal GB). A persistent FP32 master would raise that
subtotal to 132,009,607,680 bytes and is not selected. The corrected policy uses
tiled FP32 arithmetic with private stochastic BF16 write-back and adds bounded
scratch/RNG state. These are calculations, not a successful full step or fit proof.
Activations/kernels/allocator/OS remain additional. No-master recovery weights and
moments alone are 82,506,004,800 bytes; four retained/new versions are about 330.02
GB. Preflight uses the configured module inventory and selected precision policy.

Acquire assets only when explicitly authorized, then run the bounded six-stage
pilot commands from the root README. Teacher extraction and target preparation
run in separate processes from generator fine tuning. Full-width 10M teacher
states alone are 81.92 GB; fine-code storage is 7.68 GB before metadata. Checkpoint
retention can require roughly four recovery-state versions at publication peak;
the current development disk cannot validate a full 8B recovery cycle.

The exact full-training feasibility command is:

```sh
ncm profile train --config configs/validation/spark_8b_profile.yaml
```

Its finite cap is 3,600 seconds and three full optimizer steps, microbatch one,
the actual 36-block/32-update/1024-canvas configuration, BF16 parameters/gradients,
FP32 moments, explicit `fp32_moments_bf16_stochastic_v1`, activation checkpointing,
and separate small training-CE chunks. It saves
after two steps, releases model/optimizer/loss references (the completed gradient
report was already released just after its optimizer step), restores one destination
from restricted shards, and takes the next bounded step. Warmup is disabled.

BF16/FP32 CPU weak-reference tests verify that old parameters, gradients and moments
are dead before replacement allocation. This fixes extra live ownership; it does not
prove that a caching allocator returns reserved memory or that the full model fits.

Record allocated/reserved device peaks, restoration peak, host/process memory,
first versus later step times, source/answer throughput, actual sequence/canvas
positions, loss components, representative weight changes, moment sizes, and
checkpoint timings/disk. First-step initialization and validation/I/O remain
charged. A reduced-width test or warmup-only update cannot satisfy this milestone.
No execution time estimate should use advertised FP4 inference FLOPs.

Real-data codec training, real-8B full fine tuning/recovery, compatible kernel
behavior, memory feasibility, downstream quality, and speed are **not executed or
not measured** in this environment. If preflight or runtime fails, retain the
failed run and report the actual blocker; do not freeze the backbone to claim fit.

Read-only TensorBoard monitoring, optional installation, canonical logs, samples,
resume attempts and SSH viewing are documented in [monitoring](monitoring.md).
