# Stage 3 operator workflow and contracts

Stage 3 is a proposed research-candidate recipe, not a quality or hardware claim.
Use a separate checkout/device from the running Stage 2 experiment. Shared model
assets are read-only; experiment manifests, targets, codecs, ledgers and locks are
independent. Preserve GPL-3.0 and all source/component notices.

## Launch and planning

From the repository root in a working Spark CUDA environment:

```bash
python -u scripts/run_stage3.py --execute
```

This performs all eight stages and opens local chat. `--execute --no-chat` exits
after final evaluation and reporting. No action, or `--plan-only`, is offline:
no full corpus scan, weight hashing, model allocation, downloads or server.
`--qualify-only` scans only approved local source/tokenizer assets, prepares
candidate split membership and reports final target capacity as pending
distillation. It cannot predict teacher acceptance rates.

Install project dependencies separately, preserving the working CUDA Torch:
`python -m pip install -e '.[upstream,prepare,monitoring,chat]'`.
Never use the CPU extra on Spark. Missing assets have separate acquisition commands:

```bash
ncm prepare --config configs/validation/real_pilot.yaml
ncm prepare --config configs/preparation/tulu3_sources.yaml
```

The launcher does not invoke either command or retrieve missing assets.

## Recipe and variants

`configs/stage3/experiment.yaml` identifies independently configurable source,
distillation, teacher, codec, targets, generator and evaluation stages. Scheduler
contracts are saved with each codec/generator configuration. Defaults:

| Setting | Codec | Generator |
|---|---:|---:|
| Canvas positions including genuine EOS | 256 | 256 |
| Completed updates | 10,000 | 20,000 |
| Physical batch / accumulation | 4 / 8 | 1 / 16 |
| Nominal presentations | 320,000 | 320,000 |
| Parameters / moments | FP32 / FP32 | BF16 / FP32 |
| Peak LR | 1e-4 | backbone 1e-5; added modules 1e-4 |
| LR warmup | 500 | 1,000 |
| Terminal LR / peak | 0.1 | 0.1 |
| Validation / routine dev examples | 250 / 256 | 250 / 256 |
| Full recovery cadence | 500 | 1,000 |

Both use weight decay 0.01, clip norm 1, the existing tiled AdamW betas/epsilon,
and no generator FP32 master copy. Codec loss remains equal full/cutoff normalized
MSE plus 0.1 head KL; generator loss remains text CE plus 0.1 auxiliary latent
loss. The frozen vocabulary head is loaded separately for FP32 codec training;
teacher extraction is BF16. Encoding/scales use selected FP32 codec computation,
FP32 fine-code storage, and an explicit cast at the generator loss boundary.

Use `--config PATH`, `--data-config PATH`, `--source tulu`, `--canvas-size`,
`--seed`, `--experiment-dir` and `--target-policy` only for a new experiment.
All five shared canvas names and numeric aliases remain available. Alternate
geometry creates fresh artifacts; smaller canvas does not shrink backbone/Adam
state. `--target-policy reference-only` is a fingerprinted ablation; distillation
failure never silently falls back to it. Reduced fixture quotas are software-test
settings. Other scientific changes are labeled explicit variants.

The LR schedule is `linear_warmup_cosine_v1`. For next update t=completed+1,
t<=W uses t/W; otherwise r+(1-r)*(1+cos(pi*(t-W)/(T-W)))/2. With zero warmup,
t=1 uses peak and t=T uses r (T=1 uses peak). Peaks and horizon are immutable.
Failed updates, validation, microbatches and elapsed time never advance it.
The module-only/backbone-freezing warmup remains zero. Losses average actual
microbatch objectives after dividing by the actual accumulation count. This is
not a global-token average for unequal-length accumulated examples.

## Source, distillation and qualification

The approved Tülu revision/component-license policy is inherited from the existing
adapter. Metadata must fit the declared complete-pass row cap; no dataset revision
upgrade or extra corpus is implicit. The adapter uses bounded Parquet batches and
a disk-backed SQLite group/dedup index. Full supported context and the recorded
non-thinking template are retained; overlength prompts are excluded, never
silently truncated. Duplicate prompts connect groups before splits; shared generic
answer text alone does not connect unrelated prompts.

The fixed split seed is 17; validation/selection seed is 20261029. Optional
`previous_locked_manifests` accepts explicit hash-bound locked artifacts. Only
that supplied scope is checked; absent prior manifests do not establish broader
cross-experiment independence.

Free-form train/dev targets use one greedy, dropout-free original-Qwen candidate
per prompt. The teacher receives only the recorded prompt IDs/mask. There are at
most 250,000 eligible train/dev candidates, with at most three attempts for an
interrupted record, and no resampling of completed rejected candidates. Resolved
generation settings disable forced EOS, inherited minimum lengths, and max_time.
An observed EOS at position 256 is valid; reaching capacity without EOS is not.
Empty decoded answers, interior special tokens, invalid tokens and long sequences
with repeated-token fraction >=0.9 are rejected. Repetition is an operational
filter, not a semantic correctness test. No generated Python is executed.

Default approved mixed-source metadata does not certify authoritative references;
free-form teacher correctness remains unknown. A reference is retained in teacher-consistent
mode only when source component metadata explicitly marks it authoritative, its
component is in `distillation.authoritative_components`, and its declared rule is
the supported `normalized_exact` rule. No component is allowlisted by default.
Explicit reference-only mode retains
original complete references and still records their mixed origins. Teacher outputs
are recorded as teacher_generated, with original reference text preserved separately;
fixture lineage always remains software_fixture.

Per-record committed SQLite transactions retain raw token IDs, input/policy hashes,
origin, stop reason and admission. Resume verifies these records and skips completed
outputs. Attempt counts and budget charges precede generation. An unfinished source
index or feature/target attempt can be restarted as its owned unit; earlier runs
and charges remain. No latest-run search or Stage 2 handoff is used.

Final qualification seals actual counts and origins, groups, token totals and length
bins. Train target/minimum are 100,000/90,000; dev/test require 2,000 each. Below
minimum stops before codec/generator allocation. Under target warns and reports
actual counts. Training bins 1–64/65–128/129–256 aim for 25/35/40 percent of
accepted targets. Short bins redistribute deterministically among remaining
accepted examples without duplication or fabricated lengths. Locked test uses
complete original references selected independently of teacher output/student quality.
Its style distribution may differ from distilled training/dev.

## Source index storage and the November 7 failure

The failed source run `20261107T150600Z-3ab41bf8` retained 387,813 rows and
1,115,126 union-find nodes in a 4,000,276,480-byte SQLite index. Its configured
4,000,000,000-byte limit was incorrectly shared with the candidate-cache limit.
The 250,000 generation-candidate cap applies after complete source grouping,
splitting and ranking; it cannot limit the earlier full-scan index safely.
No teacher distillation or training had started. The failed experiment is retained.

Stage 3 now uses separate finite allowances:

| Quantity | Standard allowance |
|---|---:|
| Temporary full-scan SQLite index (`qualification.max_source_index_bytes`) | 32,000,000,000 bytes |
| Published candidate and locked-reference caches combined (`max_candidate_bytes`) | 4,000,000,000 bytes |
| Uncompressed bytes per indexed row | 16,777,216 bytes |
| Source cumulative logical payload/SQLite writes per attempt | 128,000,000,000 bytes |
| Source cumulative logical writes across three allowed attempts | 384,000,000,000 bytes |

The temporary Stage 3 payload format is versioned `zlib_json_v1` inside the
`stage3_source_index_v1` policy. Compression preserves complete references,
prompt/answer token IDs, licenses, origins and grouping/ranking inputs. All
eligible rows still participate in grouping before selection. Stage 1/2 keep
their existing TEXT indexes and policies. Codec/generator settings are unchanged.

SQLite `max_page_count` enforces the main-index byte allowance, including its
indexes. Native per-run/phase retained-storage checks remain active. Selected
candidate-cache size and cumulative logical writes have separate checks; pruning
or failed attempts never reset accounting. Source admission checks an allowance
of two index capacities plus the candidate cache and free-disk reserve: 88 GB
free for standard settings (68 GB source workspace plus 20 GB reserve). This is
a conservative admission allowance, not measured full-source usage or a hardware
fit claim. Other source files remain subject to the native retained-storage cap.
The offline plan exposes these numbers without scanning data or loading a model.

`source_storage.json` records the policy, observed index/cache bytes, last checked
scan counters and capacity category. Native work charges and `artifacts/scan.json`
retain later attempted work if a transaction fails. SQLite/disk/cache exhaustion
stops dependent stages and prints the report path and relevant configuration
fields. Compression and the 32 GB default have only fixture validation here;
production full-scan completion still requires an operator run. The
[dated validation receipt](stage3_source_storage_validation_2026-11-09.json) records
514 passing tests, one optional CUDA skip, lint and local wheel packaging checks.

Use a **new experiment after installing this fix**, since implementation and
storage-policy identities have changed. Existing source assets are verified and
reused read-only. Do not edit the old experiment's immutable settings or ledgers,
delete its failed index, or bypass the implementation check. Normal launch:

```bash
python -u scripts/run_stage3.py --execute
```

For a data/tokenizer-only check first, use `--qualify-only`, retain its new
printed manifest, then resume that same new manifest with `--execute` to continue.
This correction does not lower quotas, sample fewer source rows, truncate
references, change canvas size, freeze parameters, or introduce clock deadlines.

## Recovery, storage and process ownership

Keep the printed absolute manifest. Resume exactly that experiment:

```bash
python -u scripts/run_stage3.py --resume /ABS/EXPERIMENT/manifest.json --execute
```

Scientific overrides on resume are rejected before allocation. UI settings may
change. Completed receipts/configs/artifacts are verified and skipped; an at-target
training recovery performs finalization without another update. A changed schedule
horizon is a new experiment, not an implicit extension.

Validation/selection and full recovery have different cadences. Selected snapshots
contain sharded weights, step and scientific metadata; recovery snapshots also
contain FP32 moments, private rounding RNG, sampler/cursor, scheduler and exposure
counters. Weights-only snapshots cannot be passed as training recovery. Resume
uses the last complete owned recovery, and later uncheckpointed progress is replayed
under retained attempted-work charges. Publication is atomic; pruning never resets
write ledgers. Initialization is measured separately and cannot become the trained
selection. Finite codec non-improvement warns under
`validity_required_quality_advisory_v1`; nonfinite/structurally invalid artifacts,
normalization and scales remain blockers. Stage 1/2 gates are unchanged.

Training/evaluation/model-loading clocks are disabled, including offline teacher
AR generation. Elapsed observations persist. There are no timer-only heartbeats.
Children are owned foreground process groups; Ctrl+C stops only the owned tree.
Cleanup waits after cancellation are separate from execution deadlines. Experiment
and chat locks prevent competing writers and are never removed for another job.

Resource plans derive from admitted counts, actual lengths and update/accumulation
settings, including validation, normalization/scales, sampling, rejected teacher
candidates, final evaluation and finite retries. Peak retained bytes, cumulative
writes and shared-memory residency are separate quantities. Training alone has
81,920,000 maximum answer positions per stage; generator prompt+answer positions
can reach 245,760,000. These exclude validation and offline AR decode work.
The 8.25B native generator still needs roughly 99GB for BF16 weights/gradients
plus FP32 moments before activations/scratch/OS. This is an arithmetic baseline,
not a Spark backward/recovery fit measurement. MemAvailable, visible cgroup
constraints and device capacity bound one shared pool; never add host and GPU RAM.
No OOM-driven freezing, quantization, batching or geometry changes are automatic.

## Evaluation and chat

After selection freezes, full dev diagnostics and locked original-reference test
report student CE/outputs and the original AR Qwen reference on identical test
prompts and matched output allowance. Models are loaded serially. Outputs, lengths,
EOS/empty/repetition diagnostics, task rules where supported, length bins and
synchronized end-to-end generation repeats are saved. Loading time is separate;
whole-answer return does not provide time-to-first-token. Exact match on free-form
text is diagnostic. Human review is still required for relevance/coherence.
No teacher repairs student output and no external judge/API is used.

Reports are `reports/stage3_summary.md` and `.json`, including rate histories,
initial/selected/terminal diagnostics, exposures, I/O, retries, selected export and
exact resume/manual chat commands. They distinguish operational completion from
useful quality. Stage 2 is historical context, not a matched ablation.

After successful reports, the primary launcher serves the selected export on
loopback only. The default chat session has 7,200 seconds and 100 requests;
`--chat-seconds`/`--chat-requests` are independent of training clocks. No browser
is opened. `--no-chat` requires no free port or running server. UI failure is
recorded separately in `chat_status.json` without invalidating training/evaluation.
Retry with the same manifest and `--chat-port 0`. Completed resume does not repeat
locked evaluation or training. Ctrl+C in final chat leaves the experiment completed.

The launcher prints the actual selected export and exact command. Manual form:

```bash
python -m ncm_lab chat serve --checkpoint /ABS/SELECTED/EXPORT/manifest.json --device cuda --profile spark --port 7860
```

Troubleshooting wrappers require sealed prerequisites and never invoke upstream work:

```bash
python scripts/train_stage3_codec.py --experiment /ABS/EXPERIMENT/manifest.json
python scripts/train_stage3_generator.py --experiment /ABS/EXPERIMENT/manifest.json
python scripts/evaluate_stage3.py --experiment /ABS/EXPERIMENT/manifest.json
```

All implementation tests use tiny production classes/local fixtures. No real Stage 3
training, distillation, full-8B feasibility, convergence or speedup is established
by those tests. Measured hardware and output-review evidence remain required.
