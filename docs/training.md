# Native training, recovery, and budgets

All code is shared under `src/ncm_lab`; the six workflow READMEs provide exact stage
commands. `ncm prepare --plan-only` resolves finite asset budgets without retrieval
or training. Explicit asset preparation downloads only Qwen3-8B and OASST training
assets. Named hash-bound handoffs connect data, features, codec, targets, and generator.

Codec training combines 0.5 normalized full-resolution MSE, 0.5 randomly selected
coarse-cutoff MSE, and beta=0.1 frozen-head KL. At most 32 valid positions in chunks
of eight reach the full vocabulary head. Checkpointed chunk graphs bound retained
softmax activations; chunking alone is not claimed as proof of backward memory use.
Encoder/decoder gradients through that frozen head are independently tested.
Selection minimizes a fixed held-out objective; final-normalized state and latent
scales use only the admitted training set. Codec weights/scales are then frozen.

Generator loss is correctly aligned parallel text CE plus 0.1 times the mean of
sixteen normalized cumulative-code losses. Each active group is normalized by
valid positions/channels; empty bins are excluded and all-empty supervision fails.
The fixed scalar weights are initial choices, not tuned optima. Component losses
and gradient norms remain visible in streamed metrics.

An optional processed-token warmup freezes original parameters while preserving
autograd through their operations. Full fine tuning then includes every original
block, exercised embeddings, final norm/head, and all new modules. Warmup tokens
count inside the generator's 100M limit. The full-training recovery probe disallows
warmup and performs two full steps, release/reload, then another step.

The supported optimizer is tiled AdamW with FP32 first/second moments, parameter-
dtype gradients, and no FP32 master weights. CPU diagnostics use FP32 parameters;
the proposed Spark path uses BF16 parameters. Original and new generator parameters
have explicit separate rates; codec parameters use the codec learning rate. Global
gradient clipping defaults to 1. No compressed-state kernel or LoRA is assumed.
Actual parameter counts, moment bytes/dtypes, gradients, memory and step timings are
recorded; the numerical BF16 update policy is not a guarantee of convergence.

Recovery schema `native_recovery_v1` saves the entire model (including embeddings,
head, norms), optimizer/group mapping, constant-rate scheduler/scaler policy, phase,
RNGs, sample order/cursor, completed steps/tokens, source/codec identities and runtime.
It commits only at optimizer boundaries. Failed partial accumulation is not exact
mid-step recovery and its attempted work remains charged.

Serialization copies at most one configured CPU shard buffer (64 MiB default),
clones slices to avoid writing an entire backing storage, and memory-maps restricted
tensor shards on restore. Restore allocates one empty model destination from meta,
rebuilds nonpersistent RoPE buffers, then copies shards directly into parameters
and FP32 optimizer moments. It never builds a full model/optimizer Python bytes blob.
The probe drops the completed loss graph and old model/optimizer before reloading.

An atomically committed small manifest/pointer references completed checksummed
shards. All shards validate before weights are mutated. Interrupted writes cannot
mix versions or silently fall back to arbitrary older artifacts. Retention keeps the
two newest recovery versions plus selected weights; preflight includes the new
version before retirement (up to four versions). Pending/failed writes survive for
inspection. The separate inference export cannot resume training.

```sh
ncm generator train --config configs/validation/generator_pilot.yaml --resume /ABS/RUN/checkpoints/current.json
ncm codec train --config configs/validation/codec_pilot.yaml --resume /ABS/RUN/checkpoints/current.json
```

Caps may explicitly extend a resumed invocation, but source, loss, optimizer,
architecture, phase identity, and software contracts must match. The persistent
phase ledger is independent of checkpoints. It charges attempted work before
execution; new run IDs and older checkpoints cannot reset it. Full per-run wall
reservations survive abrupt crashes; normal closure replaces the reservation with
observed execution time. Source/validation/warmup/I/O costs remain charged to their
stage/phase. Per-run and retained aggregate disk caps include pending writes.
Budget checks occur at primitive boundaries; an executing kernel/file operation
may finish before the stop is observed. No budget is silently expanded.

The larger sequence is explicitly separate from smoke validation:

```sh
ncm data prepare --config configs/data/initial_100m_source_pool.yaml
ncm teacher extract --config configs/teacher/initial_10m_features.yaml
ncm codec train --config configs/codec/initial_10m_answer_tokens.yaml
ncm targets build --config configs/targets/initial_100m_targets.yaml
ncm generator train --config configs/generator/initial_100m_processed_tokens.yaml
```

The codec requires at least 10M distinct admitted training answer positions and
initially runs about three passes. The generator cap is 100M processed training
prompt-plus-answer tokens, including optional warmup. Actual charged work also
reports held-out processing separately through the phase ledger. Configurations
are finite future envelopes, not estimates of sufficiency or permission to launch.
Current pinned source may fall short; such a run blocks rather than repeats to fill
the unique quota. Longer-run time estimates require a compatible measured GPU probe.
