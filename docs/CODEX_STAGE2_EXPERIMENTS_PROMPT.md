# Codex implementation prompt: unattended Stage 2 experiment

Implement and verify a second single-pass native experiment to test whether more
short-answer data and full-generator updates improve relevance and reduce
repetition at the same 64-position canvas. Reuse the Stage 1 pipeline and preserve
its existing entry points. Follow repository AGENTS instructions. Do not
download research assets, launch the real run, or infer research results from
software fixtures while implementing this work.

## Recipe

- Use the pinned Qwen3-8B model, tokenizer and no-thinking template, medium
  64-position geometry, seed 17, and the same Stage 1 loss, selection, optimizer
  and precision boundaries. Start from the pretrained backbone and seeded added
  modules. Do not initialize from a pilot or Stage 1 export.
- Select 16,000 distinct training pairs, 200 dev pairs and 200 locked test pairs.
  Admit complete answers of up to 63 content tokens plus EOS and rendered
  prompts of up to 512 tokens. Exclude overlength answers without truncation.
  Retain group isolation and stable deduplication. Freeze IDs, lengths, hashes
  and source attribution before training. Keep the locked artifact out of
  normalization, training, validation and selection.
- Default to the existing pinned OASST source. Keep its Stage 1 split so no
  groups from the standard medium Stage 1 held-out partition enter training.
  Exclude the standard Stage 1 locked test's selected groups from Stage 2 test
  selection. This applies to the standard pinned recipe; it does not establish
  freshness against custom experiments.
- Offer explicit `--source tulu` as the larger-source alternative. Scan up to
  400,000 rows in deterministic pinned shard order. Reuse Tülu's component
  policy, complete conversation renderer, language filter, terms, attribution,
  exact OASST prompt exclusion and deduplication. Keep connected conversation
  and exact-prompt groups together. Select the same fixed train/dev/test counts
  with length stratification and component balance within length bands. This
  is a bounded short-answer slice, separate from the legacy 1,024-position
  Tülu extension. Exact OASST prompt exclusion does not establish semantic or
  benchmark decontamination.
- Extract frozen BF16 teacher features on train/dev only. Train the FP32 codec
  for 1,000 optimizer updates (batch 4, accumulation 4; 16,000 presentations).
  Check the selected codec against the Stage 1 component gate and cache frozen
  FP32 targets only on success. Train the full BF16 generator for 4,000 optimizer
  updates (batch 1, accumulation 8; 32,000 presentations, two nominal passes).
  Keep Stage 1 validation/checkpoint cadence, including generator validation
  every 50 and scheduled recovery saves every 100 updates. Select on the fixed
  dev subset's mean example text CE, independently of the terminal checkpoint.

## Execution and evaluation

Provide one launcher, `python -u scripts/run_stage2.py --execute`, for
qualification → teacher → codec gate → targets → generator → evaluation → chat.
Do not add an intermediate review, scheduled stop, extension or manual handoff.
Stop automatically on insufficient capacity, an invalid/unqualified codec,
artifact mismatch or a resource failure. Preserve attempts and print the
experiment manifest. Use Stage 1's elapsed observations without run/phase clock
enforcement for this workflow, with finite per-run and aggregate work, retained
storage and cumulative I/O budgets. The 24–48 hour window is for planning.

Planning must be offline and load no model; acquisition is a separate explicit
command. Source selection is immutable on resume. The same launcher must resume
from its exact manifest and verified owned checkpoints without repeating sealed
stages. A source failure can be followed by a separately identified new
experiment using the explicit alternate source. No fallback or quota change is
automatic.

After generator selection is frozen, run the Stage 1 locked evaluation and
retain raw prompt/reference/prediction rows and metrics for text CE, EOS,
repetition, returned and computed lengths, and runtime. Assess relevance and
coherence on a fixed set of prompts with the reference and raw output side by
side. For a Stage 1 comparison, use matched prompts and disclose differences
in training data. A single 64-position experiment cannot rank canvas sizes.

After the evaluator exits successfully, serve the verified selected standalone
export through the existing Werkzeug chat interface. Use loopback and bounded
chat sessions. Print the URL and keep the server in the foreground. If chat
cannot start, keep the completed training and evaluation intact and allow
relaunch from the same manifest without more training.

## Acceptance

Add the run, source choice, offline plan, resume and chat instructions next to
Stage 1 in the root README. Verify both plans without research dependencies or
network use. Exercise real tiny CPU stages through locked evaluation, plus data
isolation, failure gates, selected-export binding and resume without retraining.
Run `python -m pytest` and `ruff check .`; report any checks that could not run.
Report real Spark data capacity, resource fit and quality as unverified until
measured on the intended experiment.
