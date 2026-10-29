# Setup

Use Python 3.12 with the checked-in `uv.lock`:

```sh
uv sync --locked --python 3.12 --extra cpu --extra prepare --group dev
uv run --no-sync ncm --help
uv run --no-sync ncm doctor --profile cpu
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv run --no-sync ncm validate --config configs/validation/cpu_native.yaml
```

The package is `ncm-lab` 0.2.0 and CLI is `ncm`. Windows can use
`.venv/Scripts/ncm.exe`; POSIX uses `.venv/bin/ncm`. `scripts/tasks.ps1` and Makefile
expose install, test, lint, smoke, and preparation planning. Help/install/planning
do not acquire models or start initial 10M/100M training.

The neutral `.ncm-lab` marker locates repository configurations. Set NCM_REPO_ROOT
when running stage commands elsewhere. Standalone exported generation can run
without a checkout; it uses a built-in finite generation budget and a local runs
directory when no project marker exists. It still requires the installed package
and pinned Torch/Transformers dependencies.

Run `ncm prepare --config configs/validation/real_pilot.yaml --plan-only` for the
16.437 GB pinned-asset preview. Explicit preparation uses finite byte/disk/time
limits, verified checksums, and resumable HTTP acquisition. No DFlash model,
benchmark evaluator, finance dataset, or paper registry is installed.

For GPU hosts preserve their compatible CUDA Torch build instead of installing
the CPU extra over it. See [Spark setup](spark_setup.md). Environment detection
does not establish training, optimizer, recovery, kernel, quality, or speed readiness.
