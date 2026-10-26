"""Regenerate checked-in protocol/catalog files. Does not run experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, indent=2) + "\n" if not isinstance(value, str) else value, encoding="utf-8")


EXPERIMENTS = [
    ("SHR-001", "environment_and_integrity", "shared", "Establish environment and input/result integrity.",
     "All declared software contracts pass; any leakage/hash failure blocks downstream claims.",
     ["AR"], ["finqa_train_sample", "tatqa_train_sample"], ["integrity_failures"], [], "fixture"),
    ("SHR-002", "target_and_verification", "shared", "Does verification preserve target behavior?",
     "Token outputs match target greedy outputs across rejection, EOS and rollback cases; one mismatch disconfirms parity.",
     ["AR", "FLAT_RELEASED", "HIER_SKIP"], ["finite_model_fixture", "general_dev"], ["exact_token_agreement"], ["SHR-001"], "evaluate"),
    ("SHR-003", "flat_drafter_baseline", "shared", "Can released and strong external baselines be reproduced in one envelope?",
     "Pinned compatible baselines meet parity and have measured cost; unavailable baselines remain excluded, never zero-throughput.",
     ["AR", "FLAT_RELEASED", "STRONG_EXTERNAL"], ["gsm8k", "ifeval", "humaneval_plus", "longbench"], ["request_seconds", "committed_tokens_per_second"], ["SHR-002"], "evaluate"),
    ("ACL-001", "component_profile_and_pilot", "acl", "Do physical query reductions leave a credible learning/runtime opportunity?",
     "Cycle time / expected committed tokens improves on the best matched flat development control; no runtime opportunity and no quality benefit stops escalation.",
     ["FLAT_ADAPTED", "HIER_SKIP", "HIER_NO_SKIP"], ["oasst1", "general_dev"], ["cycle_seconds_per_expected_committed_token"], ["SHR-003"], "train"),
    ("ACL-002", "main_quality_speed", "acl", "Does progressive drafting improve measured efficiency with preserved output and task quality?",
     "Matched trained comparisons improve complete-request latency with parity; slower or inconclusive results are retained.",
     ["AR", "FLAT_RELEASED", "FLAT_ADAPTED", "HIER_SKIP", "FLAT_COST", "STRONG_EXTERNAL"], ["gsm8k", "ifeval", "humaneval_plus", "longbench"],
     ["request_seconds", "committed_tokens_per_second", "exact_token_agreement", "official_task_quality", "memory_bytes"], ["ACL-001"], "evaluate"),
    ("ACL-003", "hierarchy_and_skip_ablations", "acl", "Which effects are due to resolution, learned expansion, and the default skip?",
     "Predeclared paired contrasts isolate useful contributions; a flat/cost-matched win or null contrast limits the mechanism claim.",
     ["FLAT_ADAPTED", "HIER_SKIP", "HIER_NO_SKIP", "FIXED_BROADCAST", "EARLIER", "LATER", "LATE_NEAR_BUDGET"], ["oasst1", "general_dev", "gsm8k", "ifeval"],
     ["cycle_seconds_per_expected_committed_token", "accepted_prefix_survival"], ["ACL-001"], "train"),
    ("ACL-004", "generalization_and_robustness", "acl", "Where does the technical result hold or fail?",
     "A bounded replication across lengths/load/compatible target sizes retains the effect; missing hardware limits the claim.",
     ["AR", "FLAT_RELEASED", "HIER_SKIP", "FLAT_COST"], ["gsm8k", "longbench"], ["request_seconds", "task_quality", "deadline_completion"], ["ACL-002"], "evaluate"),
    ("IS-001", "financial_answer_accuracy", "is_journal", "What are financial answer accuracy and latency before workflow changes?",
     "Verified decoding preserves original target outputs; throughput can change while intrinsic answer accuracy remains equal.",
     ["AR", "FLAT_RELEASED", "HIER_SKIP"], ["finqa", "tatqa"], ["official_accuracy", "request_seconds", "correct_requests_per_hour", "scale_errors"], ["SHR-003"], "evaluate"),
    ("IS-002", "full_filing_workflow", "is_journal", "How do evidence preparation and retrieval affect the full DocFinQA workflow?",
     "A table-preserving condition improves evidence/answer outcomes within cost constraints; citation validity alone cannot establish support.",
     ["AR", "FLAT_RELEASED", "HIER_SKIP"], ["docfinqa"], ["numerical_accuracy", "retrieval_recall", "complete_request_seconds"], ["IS-001"], "workflow"),
    ("IS-003", "workpapers_and_validation", "is_journal", "Do output contracts, checking, and one repair improve auditable workpapers?",
     "Checks/repair improve correct returned coverage net of introduced errors and latency; no gain or harmful repairs disconfirm benefit.",
     ["AR", "FLAT_RELEASED", "HIER_SKIP"], ["finqa", "tatqa", "docfinqa"], ["fully_correct_workpapers", "coverage", "repair_errors", "request_seconds"], ["IS-002"], "workflow"),
    ("IS-004", "operating_frontier", "is_journal", "When does a development-trained decoder policy help bounded local workloads?",
     "Pre-generation routing improves correct completions per elapsed hour against the best development-selected fixed method; test hindsight is an unavailable oracle.",
     ["AR", "FLAT_RELEASED", "HIER_SKIP"], ["finqa", "tatqa", "docfinqa"], ["correct_requests_per_hour", "p50_seconds", "p95_seconds", "deadline_completion"], ["IS-003"], "workload"),
]


def main():
    sources = json.loads((ROOT / "third_party/sources.lock.yaml").read_text())['sources']
    model = {"target_id": "Qwen/Qwen3-8B", "target_revision": sources["target"]["revision"],
             "tokenizer_revision": sources["target"]["revision"], "draft_id": "z-lab/Qwen3-8B-DFlash-b16",
             "draft_revision": sources["drafter"]["revision"], "thinking": False,
             "scope": "hierarchical speculative drafting; unchanged AR target; not native NCM"}
    put("configs/models/qwen3_8b_dflash.yaml", model)
    methods = {
        "AR": {"description": "ordinary autoregressive target", "adaptation": False},
        "FLAT_RELEASED": {"description": "locked released five-layer DFlash, 15 future proposals plus anchor", "adaptation": False},
        "FLAT_ADAPTED": {"schedule": [15]*5, "skip": False, "adaptation": True},
        "HIER_SKIP": {"schedule": [4,4,8,15,15], "skip": True, "adaptation": True},
        "HIER_NO_SKIP": {"schedule": [4,4,8,15,15], "skip": False, "adaptation": True},
        "FLAT_COST": {"description": "select active flat depth/cost tolerance on development profiling", "selection_status": "pending"},
        "STRONG_EXTERNAL": {"model_id": "ernie-research/DCore-DFlash-Qwen3-8B-Budget16", "revision": sources["dcore_drafter"]["revision"], "backend": "DCore pinned SGLang; parity unverified"},
    }
    put("configs/methods.yaml", {"schema_version": 1, "methods": methods})
    for name, architecture, precision in [("cpu", "portable", "float32"), ("spark", "aarch64", "bfloat16"), ("cuda", "detect", "bfloat16")]:
        put(f"configs/hardware/{name}.yaml", {"profile": name, "devices": 0 if name == "cpu" else 1,
            "architecture": architecture, "precision": precision, "usable_memory_bytes": None,
            "compatibility": "detect with doctor; GPU parity and throughput not validated", "remote_resources_required": False,
            "unified_memory": "detect, never sum host and GPU totals", "precision_must_be_probed": True})
    budget = {"phase_budget_id": "cpu-verification-v2", "max_steps": 8, "max_wall_seconds": 120,
              "max_training_source_token_passes": 1000, "max_supervised_future_positions": 120,
              "max_storage_bytes": 20000000, "max_download_bytes": 0,
              "max_phase_steps": 10000, "max_phase_source_token_passes": 1000000}
    budget.update(max_phase_wall_seconds=150000,max_phase_storage_bytes=25000000000,max_phase_download_bytes=0)
    put("configs/budgets/cpu_smoke.yaml", budget)
    put("configs/budgets/cpu_profile.yaml", {**budget, "phase_budget_id": "cpu-shape-profile-v2", "max_steps": 60,
                                            "max_wall_seconds": 180})
    put("configs/budgets/pilot.yaml", {**budget, "phase_budget_id": "acl-pilot-phase-1-v2", "max_steps": 100,
        "max_wall_seconds": 1800, "max_training_source_token_passes": 1000000,
        "max_supervised_future_positions": 1500000, "max_storage_bytes": 16000000000,
        "max_phase_steps": 300, "max_phase_source_token_passes": 3000000,
        "max_phase_wall_seconds":5400,"max_phase_storage_bytes":48000000000})
    put("configs/smoke.yaml", {"experiment_id": "SHR-001", "executor": "fixture", "profile": "cpu", "seed": 0,
                              "result_origin": "test_fixture", "precision": "float32"})
    put("configs/cpu_profile.yaml", {"experiment_id": "ACL-001", "executor": "profile", "profile": "cpu", "seed": 0,
                                    "device": "cpu", "result_origin": "test_fixture", "concurrency": [1,4],
                                    "prefix_lengths": [8,32], "repetitions": 3, "precision": "float32"})
    registry = []
    for eid, slug, track, question, hypothesis, method_ids, datasets, metrics, dependencies, executor in EXPERIMENTS:
        directory = f"experiments/{track}/{eid}_{slug}"
        entry = {"id": eid, "slug": slug, "directory": directory, "study": track}
        registry.append(entry)
        protocol = {**entry, "schema_version": 1, "question": question, "hypothesis": hypothesis,
            "methods": method_ids, "datasets": datasets, "primary_metrics": metrics,
            "dependencies": dependencies, "seeds": [0,1,2] if eid in {"ACL-002","ACL-003"} else [0],
            "timing_repetitions": 5, "owner": None, "independent_reviewer": None,
            "selection": "Lock architecture, checkpoint, prompts, and cost tolerance using development data only; never test outcomes.",
            "stopping": "Stop at first finite step, time, source-token, supervised-position, storage or aggregate phase cap. Preserve failures.",
            "grouping_unit": "report/context for financial records; message_tree_id for OASST; original task IDs otherwise",
            "splits": "official train/dev/test retained; full declared test for headline results; smoke subsets labeled",
            "uncertainty": "paired report/tree-group bootstrap per training seed; timing repetitions separately recorded",
            "result_status": "Not run; no empirical findings yet", "implementation_status": "CPU reference and protocol implemented; real-model/optimized runs gated",
            "controls": {"thinking": False, "target_unchanged": True, "native_ncm": False,
                "sample_from_anchor": False, "proposal_horizon": 15, "prefix_cutoff": "strictly before each anchor",
                "same_templates_precision_stop_policy": True, "eos_not_suppressed": True},
            "exclusions": ["test_fixture results", "failed parity configurations", "unsupported sandbox/engine", "missing gold/evaluator"],
            "additional_training_match": "same warm start, data, source-token exposure, anchor sampling and valid-token CE for paired controls",
            "claim_ids": [f"{eid}-C1"]}
        if eid == "ACL-001":
            protocol["stages"] = {"A": "physically smaller query tensors; expansion/head/context/verify costs; concurrency 1 and 4",
                                  "B": "calibrate <=100 steps AND <=1800 seconds; pilot <=1M source-token passes per trained method",
                                  "continuation_target": "10% development cycle-cost improvement is provisional, not a promised result"}
        if eid == "ACL-003":
            protocol["schedules"] = {"main": [4,4,8,15,15], "earlier": [4,8,15,15,15],
                                    "later": [4,4,4,15,15], "flat": [15]*5, "late_near_budget": [8,8,8,8,15]}
            protocol["contrasts"] = ["main vs flat: resolution", "main vs no skip: skip",
                "learned vs fixed broadcast: expansion", "51 vs 52 work-proxy updates: schedule, not equal measured latency"]
        if eid == "IS-003":
            protocol["factorial_controls"] = {"bundle_sizes": [1,2,4,8], "validation": ["contract", "checks", "checks_one_repair"],
                "cache_policy": "explicitly match per-request/joint workpaper prefix reuse", "gold_visible_to_validator": False}
        put(f"{directory}/protocol.yaml", protocol)
        cfg = {"experiment_id": eid, "executor": executor, "profile": "spark", "device": "cuda", "precision": "bfloat16",
            "result_origin": "measured", "model": model, "method": "HIER_SKIP", "schedule": [4,4,8,15,15], "skip": True,
            "seed": 0, "require_gpu_parity": executor not in {"fixture", "train"},
            "required_files": ["results/generated/real_model_readiness.json"],
            "training_data": "data/processed/oasst1_train_teacher.jsonl"} if executor == "train" else {
            "experiment_id": eid, "executor": executor, "profile": "spark", "device": "cuda", "precision": "bfloat16",
            "result_origin": "measured", "model": model, "method": "FLAT_RELEASED", "schedule": [15]*5, "skip": False,
            "seed": 0, "require_gpu_parity": eid != "SHR-002", "dataset": "finqa" if track == "is_journal" else "gsm8k",
            "data_path": "data/processed/locked_evaluation.jsonl", "split": "dev", "max_requests": 8, "max_new_tokens": 256,
            "output_instruction": "Return the required benchmark answer/program only. For GSM8K terminate with #### followed by the number."}
        if executor == "fixture":
            cfg = {"experiment_id": eid, "executor": "fixture", "profile": "cpu", "precision": "float32", "result_origin": "test_fixture", "seed": 0}
        if eid == "SHR-002":
            cfg["executor"] = "verification"
        put(f"{directory}/configs/run.yaml", cfg)
        put(f"{directory}/analysis.py", f'"""Analyze immutable runs for {eid}; no model implementation here."""\nfrom ncm_lab.reporting.reports import analyze\nimport json\n\nif __name__ == "__main__":\n    print(json.dumps(analyze("{eid}"), indent=2))\n')
        readme = f'''# {eid}: {slug.replace('_', ' ')}

{question}

## Protocol

Hypothesis: {hypothesis}

Treatments and controls: {', '.join(method_ids)}. Match target, tokenizer, thinking
mode (off), precision, prompt/stop policies, resource envelope and optimizations.
Released comparisons and equal-additional-adaptation comparisons are separate.

Inputs: {', '.join(datasets)}; pinned revisions and terms are in
[`data/manifests`](../../../data/manifests). Preserve official splits and group by
report/context or conversation tree. Freeze development selection before opening test.
Do not expose gold, future target features, or oracle routing information to deployment.

Primary outcomes: {', '.join(metrics)}. Training seeds: {protocol['seeds']}; five randomized
timing trials. The independent unit is a request group, with training seed uncertainty
reported separately. Retain failed and timed-out requests in denominators.

Dependencies: {', '.join(dependencies) or 'none'}. CPU checks are portable. Real-model work
requires cached locked weights, measured hardware readiness, a compatible GPU stack,
correctness gates, data/evaluator access, and finite run AND aggregate phase budgets.
Human owner and independent reviewer are unassigned; the actual team must assign them.

## Commands

Run from any directory after installation with `NCM_REPO_ROOT` pointing at the checkout:

```sh
ncm experiment plan --id {eid} --profile spark
ncm experiment run --id {eid} --config {directory}/configs/run.yaml --budget configs/budgets/pilot.yaml
ncm results analyze --experiment {eid}
```

The run command records precise unmet prerequisites instead of downloading weights.
Select checkpoints/architecture on development data only. Stop at the first configured
budget cap; no automatic escalation. See [protocol.yaml](protocol.yaml) for contrasts,
selection, exclusions and stopping rules, and [workflow](../../../docs/experiment_workflow.md).

## Results and interpretation

Not run; no empirical findings yet.

CPU software validation is separate from research results. Real-model/optimized engine
and GPU parity remain unverified. See [result index](results/README.md). An eager reference
cannot establish deployment speed. A null result or unsupported hypothesis remains in
the run history. No native NCM, semantic-concept, or universal hardware claim is implied.
Claim reference: `{eid}-C1`; eligibility is selected in the study manifest, never inferred
from file existence or a completed fixture run.
'''
        put(f"{directory}/README.md", readme)
        put(f"{directory}/results/README.md", f"# {eid} results\n\nNot run; no empirical findings yet.\n\nRaw runs live in `../runs/<immutable-run-id>/` (ignored). Use\n`ncm results analyze --experiment {eid}` to validate and derive the result index.\nRetain failed/cancelled/blocked runs. Test fixtures never support publication claims.\n")
    put("experiments/registry.yaml", {"schema_version": 1, "experiments": registry})
    for study in ("acl", "is_journal"):
        ids = [e["id"] for e in registry if e["study"] == study]
        shared = ["SHR-001", "SHR-002", "SHR-003"]
        put(f"publications/{study}/study.yaml", {"schema_version": 1, "study": study,
            "scope": "progressive future-query architecture" if study == "acl" else "financial workflow system design",
            "native_ncm": False, "required_experiments": ids, "shared_controls": shared,
            "selected_runs": [], "eligible_methods": list(methods),
            "eligible_datasets": ["gsm8k", "ifeval", "humaneval_plus", "longbench"] if study == "acl" else ["finqa", "tatqa", "docfinqa"],
            "fixed_selection_rules": "development-only selection; full official test for headline results; exact backend correctness gate",
            "tables": ["eligible runs and provenance", "per-task outcomes with group confidence intervals", "cost/failure boundary conditions"],
            "missing_evidence": ["human experiment owners and independent reviewers", "real-model parity", "calibration and learning curves",
                                 "optimized matched-baseline measurements", "all decisive repeated-seed runs", "independent human verification"],
            "shared_evidence_class": "shared code and baseline controls are not independent new results"})
        put(f"publications/{study}/claims.yaml", {"schema_version": 1, "claims": [
            {"id": eid + "-C1", "claim": next(e[4] for e in EXPERIMENTS if e[0] == eid),
             "experiment_ids": [eid], "eligible_run_ids": [], "analysis_outputs": [],
             "uncertainty": "unmeasured", "evidence_class": "unsupported", "new_vs_shared": "new study evidence required"}
            for eid in ids]})
        put(f"publications/{study}/README.md", f"# {study} study\n\nPlans and experiment reports only; no manuscript or empirical findings yet.\n\n"
            f"Use `ncm report build --study {study}`. Selection requires immutable run IDs and manifest\nSHA256 values in `study.yaml`. Fixture and failed runs cannot populate evidence tables.\n"
            "See [publication relationship](../publication_relationship.md), the claim map, and study manifest.\n")
    put("results/index.json", {"schema_version": 1, "experiments": []})
    put(".python-version", "3.12\n")


if __name__ == "__main__":
    main()
