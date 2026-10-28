# Implementation and evidence status — 2026-10-28

Starting commit: c60a8a1849410e7266dcdca494362bf108eecceb (main). Implementation branch: codex/multiscale-latent-remediation. The validation receipt records the pre-commit snapshot; Git history records the current commit. The auxiliary codec and causal transfer are part of the first technical study.

| Capability | Implementation | Validation | Research evidence |
|---|---|---|---|
| Shared factory, actual query hierarchy/skip, homogeneous frozen target/head | complete | CPU FP32/BF16 forward/backward/step | none |
| Common multi-EOS greedy policy, incremental target KV/features and rollback | complete Transformers SDPA path | CPU differential/production tests | none |
| Response-only training, deterministic sampling/resume, development selection | complete | Student and codec recovery tested | none |
| Nested/linear/independent/single/channel/pooling codecs and causal bridges | complete | CPU fit/select/freeze/reload/transfer | none |
| Teacher, codec, cache, transfer, diagnostics, development, parity and evaluation stages | complete | Saved-artifact production DAG | none |
| Gold-free overlap audit and whole-tree/source split receipts | complete | CPU content/split/leakage checks | none |
| Bounded methods/matrices, cost selection and failure handling | complete | CPU execution and 779-job offline plan | none |
| Per-seed official metrics, paired analysis, mechanism/cost tables and figures | complete reference analysis | Fixture observations only | none |
| Evidence assessment, selection and portable dependency export | complete | Nonempty engineering archive validated outside checkout | none |
| Pinned DCore external drafter adapter | complete pinned-source adapter | CPU direct-forward parity | none |
| Released Qwen/DFlash/DCore on CUDA or Spark | implemented path awaiting execution | Not executed | none |
| Official IFEval/LongBench APIs | implemented; source versions/hashes pinned | Official runtime/data not provisioned | none |
| HumanEval+ execution | blocked pending verified external isolation | No host generated-Python execution | none |
| IS-specific FIX-13/14 financial grouping/bundle/workflow changes | explicitly deferred | Existing tests retained; reviewed defects remain | none |

The complete first-paper software flow works on tiny CPU models. GPU/kernel compatibility, training memory, convergence, benchmark reproduction and speed need separate measurements. Selected measured-research export replay also awaits genuine eligible runs. DFlare remains close prior art, without an adapter; the selected implemented external control is DCore. Native target modification and a standalone NCM generator remain future work.

See [per-requirement checklist](remediation_checklist.json), [handoff](remediation_report.md), [validation receipt](remediation_validation.json), and [exact architecture](autoencoder_architecture.md).
