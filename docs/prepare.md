# One-command preparation for the first pilot

With the project CLI installed, run:

```sh
ncm prepare --profile spark
```

This command retrieves the exact pinned Qwen3-8B target and DFlash drafter, tokenizer
and configuration files, original OASST **training** parquet, and original GSM8K
**training** data. It verifies every file against a locked SHA256 or Git blob identity,
prepares up to 2,048 approved English training/development prompts with whole-tree
separation, and writes source receipts, a readiness report and launch configurations.
Model code is not downloaded/executed; the existing local adapter loads safetensors.

The pilot selection is **18,538,430,392 bytes (18.54 GB)**. From an empty asset
directory, the default disk check requires about **20.82 GB free**, including processed
data allowance and a 2 GiB reserve. Actual training/checkpoint space is additional.
The default preparation budget permits at most 32 GiB of charged payload downloads,
48 GiB stored assets, two hours per invocation and four hours across recovery attempts.
Inspect/change `configs/preparation/pilot.yaml` before intentionally choosing another
resource envelope. Reusing a phase ID with different limits/asset selection is rejected.

Choose a larger disk using an absolute path:

```sh
ncm prepare --profile spark --assets-root /mnt/research/ncm-assets
```

On Windows with the existing environment:

```powershell
.venv\Scripts\ncm.exe prepare --profile cuda --assets-root D:\ncm-assets
```

The path is your chosen location; no specific mount/drive is assumed. The default is
`.cache/research-assets` under the checkout. The current development drive has only
about 5 GB free, so its full preparation preflight stops before downloading weights.
Acquisition may run on a machine without the final GPU, but its report keeps hardware
readiness false. This command does not replace PyTorch, install CUDA, claim GPU parity,
generate teacher continuations, train a drafter, run benchmarks or provision resources.

For an offline preview that neither downloads nor creates an asset directory:

```sh
ncm prepare --profile spark --dry-run
```

An interrupted transfer retains its `.partial` file. Repeat the same command to
verify/reuse completed files and resume compatible HTTP ranges. If a server ignores
Range, that file restarts; prior transfer bytes remain charged. A failed/short read
conservatively consumes the requested read reservation. The limit counts payload
body bytes, not TLS/header overhead. An invalid checksum or corrupt completed cache
is reported for inspection rather than silently accepted. A local `.prepare.lock`
prevents simultaneous writes; inspect an interrupted attempt before removing a stale
lock. Old attempts and their reports are retained.

## Dependencies and GPU environments

The `prepare` extra supplies the pinned Parquet reader. For a CPU development install:

```sh
uv sync --locked --extra cpu --extra scoring --extra prepare --group dev
```

In a separately configured Spark/CUDA environment, preserve its working PyTorch build:

```sh
pip install -e '.[upstream,scoring,prepare]'
```

These extras do not request PyTorch. They do not establish CUDA compatibility. Use
`ncm doctor --profile spark` and the separate hardware-validation procedure. An
alternative single invocation from an existing checkout, if `uv` is installed, is:

```sh
uv run --locked --inexact --extra upstream --extra scoring --extra prepare ncm prepare --profile spark
```

`--inexact` preserves existing packages, including the installed GPU PyTorch. On a
machine without torch, assets can still be fetched; hardware readiness remains false.
This dependency path has not been validated on an actual Spark during this session.

## What happens next

The final JSON prints absolute paths to `teacher.yaml`, `teacher_budget.yaml`,
`pilot.yaml`, `pilot_budget.yaml`, and a verification configuration, plus exact
`next_commands`. They live under the immutable preparation attempt's `launch/`.
No model/data path editing is needed. The teacher configuration writes a content-hashed
reference to its canonical run artifact; the pilot follows that reference rather than
copying the generated corpus or relying on a manually selected latest checkpoint.
Teacher generation and pilot training use separate finite compute phase IDs.
The generated teacher config selects at most 100 prompts, so normal corpus completion
fits its 100-step cap; wall/token limits can still stop the run explicitly.

Asset preparation alone is not scientific readiness: real-model parity/memory checks
and a complete ACL-suite contamination audit still need evidence. Prompt filtering
here checks GSM8K train overlap and exact OASST duplicates; it does not certify all
IFEval/EvalPlus/LongBench or pretrained-target exposure. Test labels never enter the
prepared prompts, and no publication claim becomes eligible through preparation.

Optional financial acquisition adds the original train/development releases, including
full DocFinQA filings, under the same explicit byte/storage limits:

```sh
ncm prepare --profile spark --include-financial
```

Test releases require the explicit `--with-test-data` option; they are stored without
feeding their gold into training prompts. The added dataset selection has its own
named acquisition phase. Financial scoring/serving integration and code isolation
remain separate work; downloading releases does not implement those components.

## Validation performed

Tests cover HTTP resume/restart, checksum mismatch, cached reuse, byte/time limits,
disk preflight, prompt/gold separation, tree splitting and generated-config wiring.
The full CPU suite completed with 72 passed and one optional CUDA test skipped;
lint and Git whitespace checks passed.
The real HTTPS path verified the pinned 728-byte target config. A separate 64 MiB
bounded check retrieved 39,516,251 bytes of official OASST train parquet and
4,166,206 bytes of GSM8K train data; preprocessing produced 108 train and 20 dev
prompts with source/checksum receipts. These are acquisition/software checks, not
model results. No multi-GB model weights were downloaded for this implementation.

## Revision-2 implementation update — 2026-10-27

Preparation now emits separate train and dev teacher launch configurations with whole-tree prompt receipts, response boundaries/policies, teacher-output references and external run roots. It retrieves assets only when explicitly invoked without --dry-run. The bounded CPU acceptance command retrieves no research assets. Codec fitting and transfer are separate bounded stage commands; successful acquisition does not assert construction, correctness, GPU or performance readiness.

## Continued implementation — 2026-10-28

Preparation also emits `codec_pilot_matrix.yaml`, linking initial released-flat parity, train/dev teacher generation, codec selection, latent transfer and adapted parity. `--include-external-baseline` adds the pinned DCore assets and a distinct `external_verification.yaml`; it does not relabel ordinary DFlash. All downloads still use finite acquisition budgets. No research weights were fetched during remediation.

The generated calibration pilot consumes both teacher references and has no obsolete processed-data paths or arbitrary readiness-file gate. Full-suite overlap auditing is required for the locked study; the bounded calibration pilot alone cannot support a publication claim.

Use the latest full validation counts and artifact receipts in [remediation_report.md](remediation_report.md); the earlier 72-test result above describes the original setup.
