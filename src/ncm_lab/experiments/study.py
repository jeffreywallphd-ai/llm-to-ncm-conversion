"""One compact first-paper specification, concrete bounded jobs and locked contrasts."""

import copy
from pathlib import Path
from ncm_lab.runtime.io import root, read, canonical_hash

MAIN = {
    "flat": "FLAT_ADAPTED",
    "flat-ae": "FLAT_MULTI_AE",
    "hier-no-skip": "HIER_NO_SKIP",
    "hier-ae-no-skip": "HIER_MULTI_AE_NO_SKIP",
    "hier": "HIER_SKIP",
    "hier-ae": "HIER_MULTI_AE",
}
CONTROLS = {
    "capacity": "HIER_LATENT_CE_ONLY",
    "codec-ce": "HIER_CODEC_CE_ONLY",
    "aux": "HIER_MULTI_AE_AUX_ONLY",
    "linear": "HIER_LINEAR_CODEC",
    "independent": "HIER_INDEPENDENT_AE",
    "single": "HIER_SINGLE_AE",
    "reconstruction": "HIER_MULTI_AE",
}
EXPLORATORY = {
    "fixed": "FIXED_BROADCAST",
    "earlier": "EARLIER",
    "later": "LATER",
    "near-budget": "LATE_NEAR_BUDGET",
    "pooled": "HIER_POOLED_CONTEXT",
}


def materialize(path, tiers=None):
    path = Path(path).resolve()
    spec = read(path)
    if spec.get("parent_spec"):
        parent = (root() / spec["parent_spec"]).resolve()
        full, deps = materialize(parent, tiers)
        selected = {
            j["id"] for j in full["jobs"] if j["config"]["experiment_id"] == spec["select_experiment"]
        }
        # Owned fits retain their declared consumers even when evaluation belongs
        # to another ACL folder. Shared prerequisites alone do not expand a study.
        frontier = {
            j["id"]
            for j in full["jobs"]
            if j["id"] in selected and j["config"]["executor"] in {"train", "codec"}
        }
        while frontier:
            children = {
                j["id"]
                for j in full["jobs"]
                if j["id"] not in selected
                and j["config"]["executor"] != "robustness"
                and set(j.get("depends_on", [])) & frontier
            }
            selected.update(children)
            frontier = children
        known = {j["id"]: j for j in full["jobs"]}
        pending = list(selected)
        while pending:
            for dependency in known[pending.pop()]["depends_on"]:
                if dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        full = {
            **full,
            "jobs": [j for j in full["jobs"] if j["id"] in selected],
            "contrast_instances": [
                c for c in full.get("contrast_instances", []) if set(sum(c["roles"].values(), [])) <= selected
            ],
        }
        return full, [path, *deps]
    if spec.get("kind") != "first_paper_v3":
        return spec, [path]
    ladder_path = root() / spec["data_ladder"]
    ladder = read(ladder_path)
    chosen = tiers or spec["selected_tiers"]
    if (
        not chosen
        or sorted(set(chosen)) != list(chosen)
        or any(t not in ladder["candidate_positions"] for t in chosen)
    ):
        raise ValueError("select an increasing finite subset of declared data-ladder tiers")
    contrasts_path = root() / spec["contrasts"]
    contrasts = read(contrasts_path)["contrasts"]
    jobs = []
    instances = []
    common = spec["common"]
    stage = spec["budgets"]["stage"]
    evaluation = spec["budgets"]["evaluation"]

    def add(identity, config, dependencies=(), budget=None):
        jobs.append(
            {
                "id": identity,
                "config": copy.deepcopy(config),
                "budget": budget or stage,
                "depends_on": list(dependencies),
            }
        )
        return identity

    def cfg(experiment, executor, **values):
        return {**common, "experiment_id": experiment, "executor": executor, **values}

    audit = add(
        "adaptation-overlap-audit",
        {
            "experiment_id": "SHR-001",
            "executor": "overlap_audit",
            "result_origin": "measured",
            "profile": "cpu",
            "device": "cpu",
            "training_prompt_receipt": spec["data"]["training_prompt_receipt"],
            "benchmark_receipts": [v["receipt"] for v in spec["tasks"].values()]
            + [spec["development_task"]["receipt"]],
            "threshold": 0.85,
        },
        budget=spec["budgets"]["audit"],
    )
    cost_profiles = []
    for depth, layers in [(1, [2]), (3, [0, 2, 4]), (4, [0, 1, 3, 4])]:
        cost_profiles.append(
            add(
                f"cost-profile-{depth}",
                cfg(
                    "ACL-001",
                    "profile",
                    seed=17,
                    data_seed=17,
                    methods=["FLAT_COST", "HIER_SKIP"],
                    method_overrides={"FLAT_COST": {"layer_indices": layers}},
                    cost_candidate=True,
                    prefix_lengths=[128, 512, 2048],
                    concurrency=[1],
                    repetitions=3,
                ),
            )
        )
    cost_choice = add(
        "cost-depth-selection",
        {
            "experiment_id": "ACL-001",
            "executor": "cost_selection",
            "result_origin": "measured",
            "profile": "cpu",
            "device": "cpu",
            "profile_runs": ["@" + p for p in cost_profiles],
            "tolerance": 0.1,
        },
        cost_profiles,
    )
    cost_options = {
        "cost_selection": "@cost-depth-selection/cost-selection.json",
        "cost_selection_sha256": "@cost-depth-selection/cost-selection.json#sha256",
        "layer_indices": "@cost-depth-selection/cost-selection.json#json:layer_indices",
    }

    def task_values(task):
        return {
            "dataset": task["dataset"],
            "split": task["split"],
            "data_path": task["path"],
            "input_receipts": [task["receipt"]],
            "output_instruction": task["output_instruction"],
            **({"official_evaluator": task["official_evaluator"]} if task.get("official_evaluator") else {}),
        }

    dev = task_values(spec["development_task"])
    references = {}
    for label, method in [("ar", "AR"), ("released", "FLAT_RELEASED"), ("external", "STRONG_EXTERNAL")]:
        model = copy.deepcopy(common["model"])
        if method == "STRONG_EXTERNAL":
            model.update(spec["external_model"])
        parity = add(
            "reference-parity-" + label,
            cfg(
                "SHR-002",
                "verification",
                method=method,
                model=model,
                seed=0,
                max_requests=8,
                max_new_tokens=spec["max_new_tokens"],
                **dev,
            ),
            budget=evaluation,
        )
        for task_name, task in spec["tasks"].items():
            refs = []
            for repetition in range(spec["timing_repetitions"]):
                refs.append(
                    add(
                        f"reference-{task_name}-{label}-r{repetition}",
                        cfg(
                            "SHR-003",
                            "evaluate",
                            method=method,
                            model=model,
                            seed=0,
                            max_requests=task["max_requests"],
                            max_new_tokens=spec["max_new_tokens"],
                            full_official_split=True,
                            require_gpu_parity=True,
                            parity_run="@" + parity,
                            warmup_requests=spec["warmup_requests"],
                            timing_repetition=repetition,
                            comparison="released_reference",
                            **task_values(task),
                        ),
                        [parity],
                        evaluation,
                    )
                )
            references[(task_name, label)] = refs
    for tier in chosen:
        for pair in spec["seed_pairs"]:
            seed = pair["student"]
            codec_seed = pair["codec"]
            suffix = f"t{tier}-s{seed}"
            data = {
                **spec["data"]["teacher_inputs"],
                "data_tier": tier,
                "data_ladder_version": 3,
                "data_seed": seed,
                "codec_seed": codec_seed,
                "overlap_audit": "@adaptation-overlap-audit/overlap_audit.json",
                "research_scope": "locked_study",
                "anchor_policy": "first_eligible_for_dev_seeded_train",
                "comparison": "equal_data",
            }
            codecs = {}
            for label, kind, levels, width, objective in [
                ("prediction", "nested", [4, 8, 15], 256, "prediction_aware"),
                ("reconstruction", "nested", [4, 8, 15], 256, "reconstruction_only"),
                ("linear", "linear", [4, 8, 15], 256, "prediction_aware"),
                ("independent", "independent", [4, 8, 15], 256, "prediction_aware"),
                ("single", "independent", [8], 864, "prediction_aware"),
                ("channel", "channel", [15], 256, "prediction_aware"),
                ("pooling", "pooling", [4, 8, 15], 256, "reconstruction_only"),
            ]:
                weights = [0.0] if objective == "reconstruction_only" else [0.1]
                identifier = add(
                    f"codec-{label}-{suffix}",
                    cfg(
                        "ACL-005",
                        "codec",
                        **data,
                        seed=seed,
                        codec_kind=kind,
                        levels=levels,
                        latent_width=width,
                        codec_objective=objective,
                        prediction_weights=weights,
                        steps=spec["codec_steps"],
                        max_development_examples=spec["max_development_examples"],
                    ),
                    [audit],
                )
                codecs[label] = identifier
                for split, diagnostic_data in [
                    ("dev", spec["data"]["teacher_inputs"]["development_data"]),
                    ("test", spec["data"]["diagnostic_test_data"]),
                ]:
                    add(
                        f"codec-diagnostic-{label}-{suffix}-{split}",
                        cfg(
                            "ACL-005",
                            "codec_evaluate",
                            seed=seed,
                            codec_seed=codec_seed,
                            data_seed=seed,
                            data_tier=tier,
                            codec_kind=kind,
                            levels=levels,
                            latent_width=width,
                            codec_objective=objective,
                            prediction_weights=weights,
                            dataset="gsm8k" if split == "test" else "oasst1",
                            split=split,
                            evaluation_data=diagnostic_data,
                            input_receipts=[
                                spec["data"]["diagnostic_test_receipt"]
                                if split == "test"
                                else spec["data"]["teacher_inputs"]["input_receipts"][1]
                            ],
                            full_official_split=split == "test",
                            codec_selection="@" + identifier + "/codec-selection.json",
                            codec_sha256="@" + identifier + "/codec-selected.pt#sha256",
                        ),
                        [identifier],
                    )
            fitted = {}
            for regime in spec["comparison_regimes"]:
                for label, method in {
                    **MAIN,
                    **CONTROLS,
                    "cost-flat": "FLAT_COST",
                    **(EXPLORATORY if regime == "equal_data" else {}),
                }.items():
                    if regime == "equal_total_training_wall" and label not in spec["equal_budget_treatments"]:
                        continue
                    codec_label = (
                        label
                        if label in {"linear", "independent", "single", "reconstruction"}
                        else "prediction"
                    )
                    uses_codec = method in {
                        "FLAT_MULTI_AE",
                        "HIER_MULTI_AE_NO_SKIP",
                        "HIER_MULTI_AE",
                        "HIER_CODEC_CE_ONLY",
                        "HIER_MULTI_AE_AUX_ONLY",
                        "HIER_LINEAR_CODEC",
                        "HIER_INDEPENDENT_AE",
                        "HIER_SINGLE_AE",
                    }
                    options = {
                        "method": method,
                        "treatment_id": label,
                        "seed": seed,
                        "init_seed": seed,
                        "comparison": regime,
                        "steps": spec["student_steps"],
                        "evaluation_interval": spec["selection_interval"],
                        "checkpoint_interval": spec["selection_interval"],
                        "max_development_examples": spec["max_development_examples"],
                    }
                    dependencies = [audit]
                    if uses_codec:
                        parent = codecs[codec_label]
                        options.update(
                            codec_selection="@" + parent + "/codec-selection.json",
                            codec_sha256="@" + parent + "/codec-selected.pt#sha256",
                            codec_objective="reconstruction_only"
                            if codec_label == "reconstruction"
                            else "prediction_aware",
                        )
                        dependencies.append(parent)
                    if method == "FLAT_COST":
                        options.update(cost_options)
                        dependencies.append(cost_choice)
                    if regime == "equal_total_training_wall":
                        options.update(
                            spec["cost_policy"], budget_endpoint_policy="best_completed_dev_before_budget"
                        )
                    train_id = add(
                        f"train-{label}-{suffix}-{regime}",
                        cfg("ACL-003" if label in EXPLORATORY else "ACL-005", "train", **{**data, **options}),
                        dependencies,
                    )
                    evaluation_options = {
                        key: value
                        for key, value in options.items()
                        if key
                        in {
                            "method",
                            "treatment_id",
                            "seed",
                            "init_seed",
                            "codec_selection",
                            "codec_sha256",
                            "codec_objective",
                            "comparison",
                            "cost_selection",
                            "cost_selection_sha256",
                            "layer_indices",
                        }
                    }
                    evaluation_options.update(
                        data_seed=seed,
                        codec_seed=codec_seed,
                        data_tier=tier,
                        checkpoint="@" + train_id + "/selected.pt",
                        checkpoint_sha256="@" + train_id + "/selected.pt#sha256",
                        checkpoint_selection="@" + train_id + "/selection.json",
                        overlap_audit=data["overlap_audit"],
                    )
                    if regime == "equal_total_training_wall":
                        evaluation_options["total_new_training_wall_seconds"] = spec["cost_policy"][
                            "total_new_training_wall_seconds"
                        ]
                    parity = add(
                        f"parity-{label}-{suffix}-{regime}",
                        cfg(
                            "SHR-002",
                            "verification",
                            **evaluation_options,
                            **dev,
                            max_requests=8,
                            max_new_tokens=spec["max_new_tokens"],
                        ),
                        [train_id],
                        evaluation,
                    )
                    add(
                        f"dev-{label}-{suffix}-{regime}",
                        cfg(
                            "ACL-003",
                            "evaluate",
                            **evaluation_options,
                            **dev,
                            max_requests=32,
                            max_new_tokens=spec["max_new_tokens"],
                            warmup_requests=spec["warmup_requests"],
                            require_gpu_parity=True,
                            parity_run="@" + parity,
                        ),
                        [parity],
                        evaluation,
                    )
                    if uses_codec:
                        add(
                            f"latent-diagnostic-{label}-{suffix}-{regime}",
                            cfg(
                                "ACL-005",
                                "latent_diagnostics",
                                **evaluation_options,
                                split="dev",
                                dataset="oasst1",
                                evaluation_data=data["development_data"],
                                input_receipts=[data["input_receipts"][1]],
                                max_requests=4,
                            ),
                            [train_id],
                        )
                        if label not in EXPLORATORY:
                            add(
                                f"latent-diagnostic-test-{label}-{suffix}-{regime}",
                                cfg(
                                    "ACL-005",
                                    "latent_diagnostics",
                                    **evaluation_options,
                                    split="test",
                                    dataset="gsm8k",
                                    evaluation_data=spec["data"]["diagnostic_test_data"],
                                    input_receipts=[spec["data"]["diagnostic_test_receipt"]],
                                    max_requests=32,
                                    diagnostic_scope="predeclared held-out test subset; not full benchmark evaluation",
                                ),
                                [train_id],
                            )
                    tasks = (
                        spec["tasks"]
                        if label in MAIN or label == "cost-flat"
                        else {"gsm8k": spec["tasks"]["gsm8k"]}
                    )
                    if label in EXPLORATORY:
                        tasks = {}
                    for task_name, task in tasks.items():
                        ids = []
                        for repetition in range(spec["timing_repetitions"]):
                            ids.append(
                                add(
                                    f"final-{task_name}-{label}-{suffix}-{regime}-r{repetition}",
                                    cfg(
                                        "ACL-002" if label in MAIN or label == "cost-flat" else "ACL-003",
                                        "evaluate",
                                        **evaluation_options,
                                        **task_values(task),
                                        max_requests=task["max_requests"],
                                        max_new_tokens=spec["max_new_tokens"],
                                        full_official_split=True,
                                        require_gpu_parity=True,
                                        parity_run="@" + parity,
                                        warmup_requests=spec["warmup_requests"],
                                        timing_repetition=repetition,
                                    ),
                                    [parity],
                                    evaluation,
                                )
                            )
                        fitted[(regime, label, task_name)] = ids
            for contrast in contrasts:
                for regime in contrast.get("regimes", ["equal_data"]):
                    for task_name in contrast.get("tasks", ["gsm8k"]):
                        roles = {
                            role: (
                                references[(task_name, label.split(":", 1)[1])]
                                if label.startswith("shared:")
                                else fitted.get((regime, label, task_name), [])
                            )
                            for role, label in contrast["roles"].items()
                        }
                        if any(not refs for refs in roles.values()):
                            raise ValueError(
                                "missing decisive final contrast path: "
                                + contrast["id"]
                                + "/"
                                + suffix
                                + "/"
                                + regime
                            )
                        instances.append(
                            {
                                "id": contrast["id"] + "-" + suffix + "-" + regime + "-" + task_name,
                                "claim_ids": contrast["claim_ids"],
                                "experiment_ids": contrast["experiment_ids"],
                                "roles": roles,
                                "seed_pair": pair,
                                "condition": {"data_tier": tier, "regime": regime, "task": task_name},
                                "held_constant": [
                                    "target",
                                    "dataset",
                                    "data_sha256",
                                    "split",
                                    "request_set_sha256",
                                    "policy_hash",
                                    "prompt_policy",
                                    "runtime",
                                    "scope",
                                    "result_origin",
                                ]
                                + (
                                    []
                                    if any(
                                        label.startswith("shared:") for label in contrast["roles"].values()
                                    )
                                    else [
                                        "data_tier",
                                        "comparison_regime",
                                        "data_ladder_version",
                                        "training_subset_sha256",
                                    ]
                                ),
                                "intentionally_varied": contrast["intentionally_varied"],
                                "shared_reference_roles": [
                                    role
                                    for role, label in contrast["roles"].items()
                                    if label.startswith("shared:")
                                ],
                                "retry_selection": "first eligible endpoint for the job in recorded launch order; never select on final outcomes",
                                "coverage": "full designated official split; source receipts required",
                            }
                        )
    shared_longbench = [
        job["id"]
        for job in jobs
        if job["config"]["experiment_id"] == "ACL-002"
        and job["config"]["executor"] == "evaluate"
        and job["config"].get("dataset") == "longbench"
    ]
    add(
        "robustness-shared-context-strata",
        {
            "experiment_id": "ACL-004",
            "executor": "robustness",
            "result_origin": "measured",
            "profile": "cpu",
            "device": "cpu",
            "context_length_strata": spec["context_length_strata"],
            "shared_evaluation_runs": ["@" + identifier for identifier in shared_longbench],
        },
        shared_longbench,
        spec["budgets"]["analysis"],
    )
    result = {
        **spec,
        "jobs": jobs,
        "contrast_instances": instances,
        "selected_tiers": chosen,
        "robustness": {
            "experiment_id": "ACL-004",
            "factor": "input token length",
            "strata": spec["context_length_strata"],
            "source": "shared ACL-002 LongBench final observations",
            "new_independent_runs": False,
        },
        "coverage": {
            "decisive_final_contrasts": len(instances),
            "training_jobs_with_consumers": True,
            "evidence_complete": False,
            "reason": "software paths only; real assets/calibration/measurements absent",
        },
    }
    result["declaration_sha256"] = canonical_hash({k: v for k, v in result.items() if k != "jobs"})
    return result, [path, ladder_path, contrasts_path]


def validate_coverage(jobs, contrasts):
    known = {j["id"]: j for j in jobs}
    used = {parent for job in jobs for parent in job.get("depends_on", [])}
    terminal_fits = [j["id"] for j in jobs if j["config"]["executor"] == "train" and j["id"] not in used]
    if terminal_fits:
        raise ValueError("training jobs without declared consumers: " + str(terminal_fits))
    report = []
    for contrast in contrasts:
        configs = [known[job]["config"] for refs in contrast["roles"].values() for job in refs]
        if not configs:
            continue
        for role, ids in contrast["roles"].items():
            if role in contrast["shared_reference_roles"]:
                continue
            for identifier in ids:
                config = known[identifier]["config"]
                if (
                    config.get("data_tier") != contrast["condition"]["data_tier"]
                    or config.get("comparison", "equal_data") != contrast["condition"]["regime"]
                ):
                    raise ValueError(contrast["id"] + ": tier/comparison regime mismatch")
        for field in (
            "dataset",
            "split",
            "data_path",
            "output_instruction",
            "precision",
            "device",
            "max_new_tokens",
            "warmup_requests",
        ):
            if len({canonical_hash(c.get(field)) for c in configs}) != 1:
                raise ValueError(contrast["id"] + ": held-constant final field differs: " + field)
        if any(
            c["executor"] != "evaluate" or c["split"] != "test" or not c.get("full_official_split")
            for c in configs
        ):
            raise ValueError(contrast["id"] + ": decisive contrast lacks full final-test path")
        for c in configs:
            if c["method"] not in {"AR", "FLAT_RELEASED", "STRONG_EXTERNAL"} and not all(
                c.get(k) for k in ("checkpoint", "checkpoint_sha256", "checkpoint_selection", "parity_run")
            ):
                raise ValueError(contrast["id"] + ": missing selection/parity artifacts")
        report.append(
            {
                "contrast_id": contrast["id"],
                "claim_ids": contrast["claim_ids"],
                "paths_complete": True,
                "timing_repetitions_per_role": {role: len(refs) for role, refs in contrast["roles"].items()},
                "research_evidence_complete": False,
                "unmet_prerequisites": [
                    "actual source/official split receipts",
                    "released-model calibration/parity",
                    "eligible measured runs",
                ],
            }
        )
    return {
        "schema_version": 3,
        "contrasts": report,
        "terminal_training_jobs": terminal_fits,
        "scientific_success": False,
        "software_paths_complete": True,
    }
