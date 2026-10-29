# Spark/CUDA feasibility and continuation

The intended Spark profile has 128 GB shared physical memory. It is not the current
development machine. Keep an already validated CUDA PyTorch installation; install
the package without requesting the CPU extra:

```sh
pip install -e '.[upstream,prepare]'
ncm doctor --profile spark
ncm prepare --config configs/validation/real_pilot.yaml --plan-only
```

Tested CPU dependencies are pinned in pyproject/uv.lock. ARM64/Blackwell SDPA and
the tiled optimizer remain unvalidated here. The stage configs identify Spark;
ordinary CUDA hosts can select their CUDA profile. Host offload is not an additional
independent memory pool on Spark. Available unified/device capacity must not be added.

For 8.2B parameters, BF16 weights and gradients plus two FP32 Adam moments are
about 98.4 decimal GB before activations/new modules/OS overhead. No FP32 master
copy is used; adding one would add about 32.8 GB. These are stated calculations,
not a successful optimizer step or proof of fit. Preflight estimates the configured
model and checks lower-bound memory plus retained/new checkpoint disk demand.

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
FP32 moments, activation checkpointing, and chunked exact vocabulary CE. It saves
after two steps, releases model/optimizer/loss references, restores one destination
from restricted shards, and takes the next bounded step. Warmup is disabled.

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
