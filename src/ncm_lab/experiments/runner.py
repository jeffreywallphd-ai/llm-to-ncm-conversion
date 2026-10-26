from pathlib import Path
from ncm_lab.experiments.registry import get
from ncm_lab.runtime.budget import Budget, Meter, BudgetExceeded
from ncm_lab.runtime.doctor import doctor
from ncm_lab.runtime.io import root, read
from ncm_lab.runtime.provenance import Run, validate_run


def prerequisites(config):
    if config["executor"] in {"fixture", "profile"} and config.get("result_origin") == "test_fixture":
        return []
    missing = []
    for requirement in config.get("required_files", []):
        if not (root() / requirement).exists():
            missing.append(f"required artifact missing: {requirement}")
    for key in ("training_data", "data_path", "prompts_path"):
        if key in config and not (root() / config[key]).is_file():
            missing.append(f"{key} missing: {config[key]}")
    if config.get("training_data_reference"):
        try:
            from ncm_lab.training.loop import resolve_training_data
            resolve_training_data(config)
        except (OSError,ValueError,KeyError) as exc:
            missing.append(f"teacher data not prepared: {exc}; run the generated teacher configuration first")
    if config.get("executor") in {"workflow", "workload", "external_engine"}:
        missing.append("configure and audit the external workflow/serving harness before running this research protocol")
    if config.get("require_gpu_parity") and not config.get("parity_run"):
        missing.append("supply a validated parity_run for the exact model/backend/precision before a research timing run")
    if config.get("parity_run"):
        try:
            parity = validate_run(Path(config["parity_run"]), require_current_code=True)
            if parity["status"] != "completed" or parity["result_origin"] != "measured":
                missing.append("parity_run must be a completed measured real-model verification run")
            metrics = read(Path(config["parity_run"])/"metrics.json")
            parity_config = read(Path(config["parity_run"])/"config.json")
            if parity["experiment_id"] != "SHR-002" or not metrics.get("all_equal") or metrics.get("requests",0) < 1:
                missing.append("parity_run must contain successful SHR-002 token-agreement observations")
            for field in ("model","precision","device","method","schedule","skip","checkpoint_sha256"):
                if parity_config.get(field) != config.get(field):
                    missing.append(f"parity configuration differs for {field}")
        except (OSError, ValueError) as exc:
            missing.append(f"parity prerequisite invalid: {exc}")
    if config.get("result_origin") != "test_fixture":
        cache = Path(config.get("model_cache", Path.home() / ".cache/huggingface/hub"))
        for kind in (("target",) if config["executor"]=="teacher" else ("target","draft")):
            model = config.get("model", {}).get(f"{kind}_id")
            rev = config.get("model", {}).get(f"{kind}_revision")
            local = config.get("model",{}).get(kind+"_path")
            snapshot = Path(local) if local else cache / ("models--" + str(model).replace("/", "--")) / "snapshots" / str(rev)
            if not snapshot.is_dir():
                missing.append(f"cached {kind} snapshot absent: {model}@{rev}; explicit model download budget required")
    return missing


def run(experiment_id, config_path, budget_path):
    config_path,budget_path = Path(config_path),Path(budget_path)
    if not config_path.is_absolute():
        config_path = root()/config_path
    if not budget_path.is_absolute():
        budget_path = root()/budget_path
    protocol, config = get(experiment_id), read(config_path)
    if config.get("experiment_id") != experiment_id:
        raise ValueError("configuration experiment ID mismatch")
    budget = Budget(**read(budget_path))
    environment = doctor(config.get("profile", "cpu"))
    record = Run(protocol, config, budget, environment, config.get("parent_run_ids", []))
    missing = prerequisites(config)
    if missing:
        final = record.close("blocked", {"unmet_prerequisites": missing}, "; ".join(missing))
        return {"run_id": record.id, "path": str(record.path), "status": final["status"], "blockers": missing}
    try:
        from ncm_lab.experiments.executors import fixture_workflow, profile, train_real, evaluate, teacher, verification
        executor = {"fixture": fixture_workflow, "profile": profile, "train": train_real, "evaluate": evaluate,
                    "teacher":teacher,"verification":verification}[config["executor"]]
        metrics = executor(record, config, Meter(budget, record.path))
        final = record.close(metrics.get("status", "completed"), metrics)
    except (BudgetExceeded, KeyboardInterrupt) as exc:
        final = record.close("cancelled", {}, str(exc))
    except Exception as exc:
        final = record.close("failed", {}, f"{type(exc).__name__}: {exc}")
    return {"run_id": record.id, "path": str(record.path), "status": final["status"], "error": final["error"]}
