# Portable setup

Use Python 3.12; Python 3.15 on this host is outside the pinned project range.
`uv sync --locked --extra cpu --extra scoring --extra prepare --group dev` creates an isolated
`.venv`, with CPU torch from the explicit CPU index and hash-locked dependencies.
Windows x86-64 Python 3.12.14, torch 2.13.0+cpu and transformers 5.15.0 were exercised.
Other supported platforms still need their own verification run. Dependency
installation downloads packages only; it does not start model/dataset acquisition.

Run the README's doctor, pytest, lint, data-check and smoke commands. `ncm --help`
and each subcommand's help are authoritative. With an activated environment, `ncm`
works directly; otherwise use `uv run --no-sync ncm`. PowerShell convenience targets
are provided in `scripts/tasks.ps1`; POSIX targets are in the Makefile.

`NCM_REPO_ROOT` resolves configuration if working outside a source checkout.
Run configuration paths are explicit arguments; paths inside tracked configs resolve
against that checkout. Outputs default to structured ignored run directories.
Set `artifact_root` to relocate heavy artifacts while retaining track/experiment/run
layout. Never commit weights, target features, private tokens, or full data releases.

The smoke workflow needs no network, weights, paid APIs, or generated-code execution.
It uses tiny random model/tensor fixtures and small original licensed training records.
Those are excluded from publication evidence. See `third_party/README.md` for source
snapshots and licenses, `docs/implementation_status.md` for validation limits, and
`docs/spark_setup.md` for a separate GPU environment.

Use `ncm prepare --profile spark` for budgeted model/data retrieval and prompt
preparation, or add `--dry-run` for an offline preview. See [prepare.md](prepare.md).
