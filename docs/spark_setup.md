# Single DGX Spark setup and compatibility

The intended primary machine is one Spark. No Spark was connected during setup.
NVIDIA documents an ARM CPU and unified CPU/GPU memory; inspect the actual machine
instead of using advertised capacity or FP4 peak throughput for BF16 planning.
See https://docs.nvidia.com/dgx/dgx-spark/hardware.html (checked 2026-10-25).

Use a separate environment or vendor-supported container. Keep its existing CUDA
PyTorch build: **do not run the CPU-extra sync inside that environment**. Install
the project with `pip install -e '.[upstream,scoring,prepare]'` only after checking that the
pinned Transformers/scoring requirements are compatible with the existing torch.
This extra does not request or replace PyTorch. `requirements/gpu-tooling.txt`
pins application tooling; `configs/hardware/spark.yaml` records unresolved runtime
details. Record the actual image digest and Python/driver/CUDA/torch/kernel versions.
There is no validated Spark container digest yet; guessing one would be misleading.

Run `ncm doctor --profile spark`. Check ARM wheels, usable unified memory, BF16,
SDPA, compilation support, model cache, disk and isolation. The doctor cannot certify
an engine just because a CUDA device or Docker executable is present. It reports
unknown measurements explicitly. Add a generic second device using the CUDA profile.

`ncm prepare --profile spark` now retrieves pinned pilot weights/data, verifies
integrity, prepares train/dev prompts and produces launch configurations. Give it
an asset directory with enough disk space. See [prepare.md](prepare.md) for its
finite acquisition budgets. It does not alter the working CUDA/PyTorch environment.

After explicit model/download budgets and data preparation, run SHR-002 parity using
the exact target/draft/tokenizer, precision and backend. Cover rejection/EOS/length
cases before timing. The eager reference supports correctness, not deployment speed.
DCore's pinned SGLang stack and custom hierarchy engine integration remain unverified.
Do not compare a published multi-GPU LongSpark number with a single Spark measurement.

The first training phase caps each matched method at 100 steps, 1800 seconds and
1M source-token passes; its aggregate caps cover three methods. Runtime duration,
throughput and memory estimates remain unknown until calibration is measured.

## Revision-2 implementation update — 2026-10-27

The implemented efficient reference path uses Transformers incremental KV/features and SDPA; no SGLang/vLLM-specific hierarchy adapter is claimed. Run exact cache-only SHR-002 verification with the generated verification config and full effective EOS policy before research timing. Qwen3-8B/DFlash and Spark/CUDA validation remain unexecuted on this Windows CPU PyTorch host. Preserve a separate CUDA environment.
