from ncm_lab.runtime.io import root, read


def registry():
    spec = read(root() / "experiments/registry.yaml")
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported experiment registry schema")
    out, paths = {}, set()
    required = {"id", "slug", "directory", "study", "question", "hypothesis", "methods", "datasets",
                "primary_metrics", "dependencies", "seeds", "timing_repetitions", "selection", "stopping"}
    for entry in spec["experiments"]:
        if entry["id"] in out or entry["directory"] in paths:
            raise ValueError("duplicate experiment ID/directory")
        path = (root() / entry["directory"]).resolve()
        if not path.is_relative_to(root()) or not path.is_dir():
            raise ValueError("experiment directory is missing or outside repository")
        protocol = read(path / "protocol.yaml")
        if not required <= protocol.keys() or any(protocol[k] != entry[k] for k in ("id", "slug", "directory", "study")):
            raise ValueError(f"invalid protocol contract: {entry['id']}")
        for name in ("README.md", "analysis.py", "results/README.md", "configs/run.yaml"):
            if not (path / name).is_file():
                raise ValueError(f"missing experiment artifact: {entry['id']}/{name}")
        paths.add(entry["directory"])
        out[entry["id"]] = protocol
    for protocol in out.values():
        if not set(protocol["dependencies"]) <= out.keys():
            raise ValueError("unresolved experiment dependency")
    return out


def get(experiment_id):
    if experiment_id not in registry():
        raise ValueError(f"unknown experiment ID: {experiment_id}; use ncm experiment list")
    return registry()[experiment_id]


def plan(experiment_id, profile="cpu"):
    p = get(experiment_id)
    spec_path = root()/"experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml"
    spec = read(spec_path)
    selected = [job for job in spec["jobs"] if job["config"]["experiment_id"] == experiment_id
                and (not job["config"].get("method") or job["config"]["method"] in p["methods"])]
    known = {job["id"]:job for job in spec["jobs"]}
    closure = {job["id"] for job in selected}
    pending = list(closure)
    while pending:
        for dependency in known[pending.pop()].get("depends_on",[]):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    trained = [m for m in p["methods"] if m not in {"AR", "FLAT_RELEASED", "STRONG_EXTERNAL"}]
    count = len(trained) * len(p["seeds"])
    per_run = 1_000_000 if experiment_id == "ACL-001" else 30_000_000
    return {"id": p["id"], "profile": profile, "offline_plan_only": True, "methods": p["methods"],
            "implementation": "later_journal_workflow_pending" if experiment_id.startswith("IS-") else "shared_staged_handlers",
            "research_spec":str(spec_path), "direct_job_ids":[j["id"] for j in selected],
            "dependency_closure_ids":sorted(closure), "default_config":str(root()/p["directory"]/"configs/run.yaml"),
            "bounded_matrix_command":"ncm matrix plan --spec <research_spec> --output <new-resolved-plan.json>",
            "unsupported_conditions":["generated Python without external isolation", "unvalidated GPU/backend",
                "second target or concurrent batched decoding without a compatible implementation"],
            "datasets": p["datasets"], "training_seeds": p["seeds"], "trained_method_seed_runs": count,
            "timing_repetitions": p["timing_repetitions"], "per_run_provisional_source_token_cap": per_run,
            "selected_training_matrix_upper_token_passes": count * per_run,
            "whole_phase_budget": "required finite reservation; pilot, ablation, evaluation and replication add to this matrix",
            "estimated_examples": None, "estimated_anchors": None, "supervised_future_positions": None,
            "unknown_costs": ["calibrate measured steps/second including frozen target", "dataset sizes after deduplication",
                              "teacher generation", "evaluation repetitions", "second target/hardware", "checkpoint storage"],
            "storage": "online target features; finite artifact/cache caps required",
            "prerequisites": p["dependencies"] + ["locked source/data revisions", "cached compatible weights",
                            "correctness gate on exact backend/precision", "finite per-run and aggregate phase budget"],
            "selection": p["selection"], "stopping": p["stopping"], "research_run_started": False}
