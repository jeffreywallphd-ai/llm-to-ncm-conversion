# Configurable canvas experiments

The registry in `src/ncm_lab/config/canvas.py` defines xsmall/8, small/32,
medium/64, large/256 and xlarge/1024. Fresh named pilots default to medium.
An explicit configuration controls geometry unless `--canvas-size` intentionally
overrides its canvas and schedule. Other architecture fields are retained and
validated. Explicit legacy files and inference metadata keep their saved geometry;
the Architecture value object's historical default remains 1024 for compatibility.
Typed stage configs without an architecture now resolve to medium consistently.

All production presets use the same pinned Qwen3-8B, width 384, four queries,
8/16/12 blocks and 32 updates. The 1024 schedule remains byte-for-byte equivalent
in serialized architecture fields; a cosmetic label is never added to its identity.
Other schedules use `round(4 * (C / 4) ** (i / 31))` for i=0..31. Repeated sizes
retain distinct level embeddings, gates and refinement operations. Software fixtures
explicitly use tiny hidden/latent widths and cannot establish real-model behavior.

See the root README's **Configurable canvas sizes and 100-step pilots** for the
independently copyable commands. Scripts and `ncm pilot train` share one resolver.
`--plan-only` reports schedules, objectives, stage command templates, namespaced
paths, known reuse decisions and clocks without mutation/allocation/acquisition.
Capacity remains unknown until local tokenization and real reference qualification.

`--prepare` executes data -> teacher -> codec -> targets -> generator in separate
owned processes, without acquisition or an outer training deadline. Original
teacher/codec/training implementations are reused. The codec's existing two-step
objective is not upgraded to 100. Its adequacy requires separate measurement.
Default reference admission includes EOS and excludes overlength answers; an
8-position canvas fits seven content positions plus EOS. Prompt capacity stays 512.
Qualification records inspected source rows, eligible references, exclusion reasons,
split counts, distinct answer positions and admitted length statistics. Missing
train/dev capacity blocks before teacher allocation. Explicit `--data-config PATH`
with `--prepare` selects a native data config and approved local source/bounds.

`runtime/canvas_pipeline.py` centralizes namespace construction. Full experiment
fingerprints include architecture, source/configuration/numerical contracts and
implementation. Per-stage ledgers retain all charges within that experiment;
2/100 generator targets and validation/checkpoint cadences share one phase.
Immutable run artifacts are separate from atomic convenience pointers under
`runs/handoffs/canvas/<preset>/<fingerprint>/`. Whole-pipeline and stage writer locks
reject concurrent conflicting writers. A stale lock is retained after a killed
process: inspect its owner before manual recovery; never reset a ledger to evade caps.
No legacy global handoff is rewritten. Hash-bound producer receipts require sealed
completed runs and verified files, not merely a zero child exit status.

Resume infers architecture and immutable input manifests from the hash-verified
checkpoint and its launch record. It does not resolve mutable latest pointers.
Explicit size/config mismatches fail before model allocation. Checkpoints must
remain in their original run layout with resolved config/launch records. The
complete normal optimizer, RNG, runtime, source, phase and implementation checks
still apply. Canvas changes require fresh training; shapes alone do not authorize
transfer. Historical clock/chat compatibility receipts remain immutable and do not
approve this scientific source change. No legacy recovery transition was added.
Existing independent inference exports remain loadable with their saved geometry.

`--no-chat` does not import/check the optional server or probe a port. Successful
completion prints the exact export and standalone chat command. Otherwise chat
loads only the completed run's export after the training process exits. Canonical
summary/attempt/stream metadata includes canvas and origin; TensorBoard's attempt
text includes these fields. Export and summary label the selected optimizer step.
All C positions compute before EOS trimming, independently of projection chunk size.

Testing uses bounded CPU fixtures with the production block/update structure,
meta-device parameter-shape checks and a local 2-to-100 continuation. Full 8B
learning, all-size Spark feasibility, real-data short-answer availability, recovery
memory peaks, quality and speed remain separate hardware/data work. No curriculum,
cross-size transfer or new benchmark suite is implemented.
