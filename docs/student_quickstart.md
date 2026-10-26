# Researcher quickstart

1. Install and run the CPU quickstart in the README. Read the architecture's indexed
   anchor example before changing masks, positions, taps, labels or cache logic.
2. Run `ncm experiment list`, then `ncm experiment plan --id ACL-001 --profile spark`.
   Plans do not initialize models or launch training. Read that experiment's README
   and protocol; the human team assigns an owner and independent reviewer.
3. Inspect data using `ncm data inspect --dataset finqa` and validate the tracked
   training samples with `ncm data check`. Gold records are evaluator-only. Do not
   open test splits to tune prompts, repair rules, architecture or checkpoints.
4. Run the bounded CPU shape diagnostic from the README. Inspect the returned run's
   config, events, component timings and final manifest. Validate its returned ID.
   It is a software microbenchmark, not real-model evidence.
5. On the actual Spark, use its doctor report and a separate CUDA environment. Obtain
   explicitly budgeted, locked weights and prepare teacher data. Complete target/
   draft parity before timing. Run only the pilot caps, analyze failures, and stop
   if the measured opportunity is unsupported. Do not escalate to rescue a premise.

Record work in the contribution ledger as it happens. Reproducing an import is not
reproducing a benchmark; a passing citation-ID check is not a factuality assessment;
a completed report is not a manuscript. Bring uncertain scientific choices to the
human reviewer with the exact configuration, source reference and evidence.
