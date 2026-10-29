# Native decisions — 2026-10-29

The user explicitly replaced the unchanged-target drafter goal with a native
embedded Qwen hierarchy. Preserve the old work at a verified archive branch before
removing active drafter/study code. Do not reconstruct an academic study on main.

Use the specified 8/16/12 block allocation, 32 sequential updates, 1024-position
canvas, shared width-384 codec, sixteen cumulative losses, and full fine tuning.
Use fixed interval-overlap geometry and integer-center positions; no gold length
or target code enters the student. Keep longer skips disabled. Initial gates are
nonzero and group projections are shared only within successive pairs.

Teacher states use P-1+t, including the first answer token and genuine EOS. Cache
only valid states/fine codes with content/geometry/precision identity. Larger target
preparation is a separate streaming process, never a generator-side teacher fallback.
Normalize using training-only statistics in raw encoder coordinates.

Use exact checkpointed/chunked vocabulary losses and explicit FP32-moment tiled
AdamW, with parameter-dtype gradients and no master copy. These choices trade
recomputation for bounded memory and require actual Spark measurement. Publish
bounded shards behind atomic manifests and retain prior/selected versions.

Report software fixtures, small-batch learning, real-data codec fitting, and actual
8B optimizer/recovery as distinct milestones. Preserve negative/failed attempts and
cumulative budget charges. No initial large run is launched to finish implementation.
