# Experiment lifecycle

The registry maps canonical IDs to explicit directories. Every core experiment has
a protocol, configuration, analysis entry point and result index. Planning uses only
local files. Use `ncm experiment list` and `ncm experiment plan --id ACL-001 --profile spark`.

`ncm prepare --profile spark` retrieves the pinned pilot assets and prepares prompt
files plus machine-specific launch configurations. It prints exact teacher/pilot
commands and wires their artifact references. See [prepare.md](prepare.md) for finite
acquisition budgets, disk requirements and resumable transfers.

Supply a resolved config and finite budget to `ncm experiment run`. A run gets a unique
ID, immutable launch/config/protocol/environment copies, append-only events and a final
manifest with artifact digests. Failed, blocked and cancelled runs are retained. Heavy
artifacts are ignored by Git; derived indexes contain pointers, not duplicated predictions.

For a portable first run:

```sh
ncm smoke
ncm experiment run --id ACL-001 --config configs/cpu_profile.yaml --budget configs/budgets/cpu_profile.yaml
ncm results analyze --experiment ACL-001
ncm report build --study acl
```

The first command trains two fixture steps including save/resume and writes a validated
prediction/scoring/report chain. The second measures tiny artificial tensor shapes.
Both are excluded from research findings. `ncm results validate --run RUN_ID` verifies
artifact integrity, configuration identity and fixture exclusion.

Real pilot configuration is at
`experiments/acl/ACL-001_component_profile_and_pilot/configs/run.yaml`. It requires
locked cached model snapshots, target-generated train-only token IDs in the configured
JSONL, a readiness artifact and the actual CUDA stack. The command records exact blockers
until those exist. No download is triggered. Training records contain input_ids,
group_id, origin=target_distillation, split=train and generated_tokens; checks reject
other splits. Resume requires a trusted checkpoint path and expected SHA256.

To generate capped target continuations after preparing allowlisted train prompts,
use `ncm experiment run --id SHR-001 --config configs/teacher.yaml --budget configs/budgets/pilot.yaml`.
It uses only cached locked target/tokenizer files and writes teacher.jsonl inside the
immutable run directory; copy/reference that artifact in the training configuration.
No evaluator gold enters the teacher prompts. The budget also accounts for generated
tokens; this data-preparation phase needs its own deliberately selected phase ID.

Checkpoint selection is deterministic on development loss with step/artifact-ID tie
breaks. The last budgeted checkpoint has an explicit content artifact ID but is not
automatically publication-selected. Record the dev selection evidence and full backend
parity before real timing. A research reference evaluation remains ineligible for
optimized deployment-speed claims.

Analyze validates raw run hashes before deriving summaries. Reports read selected
eligible run manifests and never promote fixtures. Review exports select one study,
exclude other experiment directories, preserve licenses and third-party provenance,
write an archive digest/internal receipt, and request human anonymity review. Nothing
is uploaded or submitted. Preserve the ACL code/config/checkpoint/result snapshot
before journal-specific research changes alter shared results.

## Revision-2 implementation update — 2026-10-27

Use ncm acceptance for the full bounded CPU codec-to-transfer workflow; ncm matrix plan/run freezes and executes a DAG. Codec/transfer configs are under ACL-005. The train handler evaluates actual held-out records and writes selection.json; the codec handler writes codec-selection.json. ncm checkpoint validate-selection checks retained weights and optional dev identity. Recovery final.pt is transient latest within an active run; selected.pt is deployment state, not an optimizer-resume artifact.
