from pathlib import Path
import time
from ncm_lab.experiments.registry import get
from ncm_lab.runtime.budget import Budget, Meter, BudgetExceeded
from ncm_lab.runtime.doctor import doctor
from ncm_lab.runtime.io import root, read
from ncm_lab.runtime.provenance import Run, validate_run


def prerequisites(config):
    if config["executor"] in {"fixture", "profile", "acceptance"} and config.get("result_origin") == "test_fixture":
        return []
    missing = []
    receipts=list(config.get('input_receipts',[]))+list(config.get('benchmark_receipts',[]))
    if config.get('training_prompt_receipt'):
        receipts.append(config['training_prompt_receipt'])
    if config.get('overlap_audit'):
        receipts.append(config['overlap_audit'])
    if config.get('latent_cache'):
        receipts.append(config['latent_cache'])
    for receipt in receipts:
        if not (root()/receipt).is_file():
            missing.append('required input/audit receipt missing: '+receipt)
    if str(config.get("device", "cpu")).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            missing.append("requested CUDA execution is unavailable in this Python environment")
    if config.get("executor") == "evaluate" and config.get("dataset") in {"ifeval", "longbench"}:
        evaluator = config.get("official_evaluator",{})
        if not evaluator.get("root") or not Path(evaluator["root"]).is_dir() or not evaluator.get("files_sha256"):
            missing.append("provision the pinned official evaluator checkout and verified file hashes")
    if config.get("dataset") == "humaneval_plus":
        missing.append("generated-Python scoring requires a verified external isolation boundary")
    for requirement in config.get("required_files", []):
        if not (root() / requirement).exists():
            missing.append(f"required artifact missing: {requirement}")
    for key in ("training_data", "data_path", "prompts_path","evaluation_data"):
        if key in config and not (root() / config[key]).is_file():
            missing.append(f"{key} missing: {config[key]}")
    if config.get("development_data_reference") and not (root()/config["development_data_reference"]).is_file():
        missing.append("held-out development teacher artifact not prepared")
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
            for field in ("model","precision","device","method","schedule","skip","checkpoint_sha256",
                          "generation_policy","max_new_tokens","codec_sha256"):
                if parity_config.get(field) != config.get(field):
                    missing.append(f"parity configuration differs for {field}")
        except (OSError, ValueError) as exc:
            missing.append(f"parity prerequisite invalid: {exc}")
    if config.get('executor') in {'overlap_audit','cost_selection'}:
        return missing
    if config.get("result_origin") != "test_fixture":
        cache = Path(config.get("model_cache", Path.home() / ".cache/huggingface/hub"))
        for kind in (("target",) if config["executor"] in {"teacher", "codec", "codec_evaluate", "latent_cache"}
                     or config.get("method") == "AR" else ("target","draft")):
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
    config.setdefault('cpu_threads',1)
    if isinstance(config['cpu_threads'],bool) or not isinstance(config['cpu_threads'],int) or not 1<=config['cpu_threads']<=256:
        raise ValueError('cpu_threads must be an integer in 1..256')
    import torch
    torch.set_num_threads(config['cpu_threads'])
    if config.get('executor') in {'codec','codec_evaluate'}:
        kind=config.get('codec_kind','nested')
        weights=config.get('prediction_weights',[0.0,.1])
        expected='CODEC_'+kind.upper()+('_RECONSTRUCTION' if kind=='nested' and weights==[0] else '')
        if config.get('method',expected)!=expected:
            raise ValueError('codec identity contradicts its computation')
        config={**config,'method':expected,'dataset':config.get('dataset','oasst1')}
    if config.get("experiment_id") != experiment_id:
        raise ValueError("configuration experiment ID mismatch")
    budget = Budget(**read(budget_path))
    environment = doctor(config.get("profile", "cpu"))
    record = Run(protocol, config, budget, environment, config.get("parent_run_ids", []))
    missing = prerequisites(config)
    if missing:
        final = record.close("blocked", {"unmet_prerequisites": missing}, "; ".join(missing))
        return {"run_id": record.id, "path": str(record.path), "status": final["status"], "blockers": missing}
    meter = Meter(budget, record.path)
    started = time.monotonic()
    def incomplete_resources(reason):
        return {'executor_wall_seconds':time.monotonic()-started,'target_processed_positions':None,
            'optimizer_steps':None,'teacher_generated_tokens':None,
            'storage_bytes':sum(p.stat().st_size for p in record.path.rglob('*') if p.is_file()),
            'missing_reason':reason,'charged_source_positions':meter.source_tokens,'charged_operations':meter.steps}
    try:
        from ncm_lab.experiments.executors import (fixture_workflow,profile,train_real,evaluate,teacher,
            verification,development,latent_diagnostics,cost_selection_stage)
        from ncm_lab.experiments.codec_stages import codec_stage, codec_evaluate_stage, latent_cache_stage
        from ncm_lab.experiments.acceptance import full_acceptance
        from ncm_lab.data.overlap import audit
        executor = {"fixture": fixture_workflow, "profile": profile, "train": train_real, "evaluate": evaluate,
                    "teacher":teacher,"verification":verification, "codec": codec_stage,
                    "codec_evaluate": codec_evaluate_stage, "latent_cache": latent_cache_stage,
                    "acceptance": full_acceptance, "development": development,'overlap_audit':audit,
                    'latent_diagnostics':latent_diagnostics,'cost_selection':cost_selection_stage}[config["executor"]]
        from ncm_lab.data.artifacts import verify_input
        inputs = []
        receipts = list(config.get("input_receipts", []))+list(config.get('benchmark_receipts',[]))
        if config.get('training_prompt_receipt'):
            receipts.append(config['training_prompt_receipt'])
        for field in ("training_data_reference", "development_data_reference"):
            if config.get(field):
                reference = read(root()/config[field])
                if reference.get("input_receipt"):
                    receipts.append(reference["input_receipt"])
        for receipt in dict.fromkeys(receipts):
            inputs.append({"receipt": receipt, "identity": verify_input(receipt)})
        if config.get("result_origin") != "test_fixture":
            if any(item['identity'].get('fixture') for item in inputs):
                raise ValueError('fixture input lineage cannot enter a measured research run')
            consumed = {}
            for key in ("training_data", "development_data", "data_path", "prompts_path","evaluation_data"):
                if config.get(key):
                    consumed[key] = (root()/config[key]).resolve()
            if config.get("training_data_reference"):
                from ncm_lab.training.loop import resolve_training_data
                consumed["training_data"] = resolve_training_data(config).resolve()
            if config.get("development_data_reference"):
                reference = read(root()/config["development_data_reference"])
                consumed["development_data"] = Path(reference["path"]).resolve()
            for key, path in consumed.items():
                matched = [i["identity"] for i in inputs if Path(i["identity"]["path"]).resolve() == path]
                if len(matched) != 1:
                    raise ValueError(f"{key} requires exactly one verified consumed-file receipt")
                expected = "train" if key == "training_data" else "dev" if key == "development_data" else config.get("purpose", "train") if key == "prompts_path" else config["split"]
                if expected=='test_evaluation':
                    expected='test'
                if matched[0]["split"] != expected or matched[0].get("fixture"):
                    raise ValueError("research input split/origin receipt mismatch")
            if config.get("training_data") and config.get("development_data"):
                from ncm_lab.training.records import load_records, disjoint
                disjoint(load_records(consumed["training_data"], "train"), load_records(consumed["development_data"], "dev"))
        from ncm_lab.runtime.io import write_json
        write_json(record.path/"inputs.json", inputs, exclusive=True)
        if config.get('preparation_reference'):
            from ncm_lab.runtime.io import digest
            reference=read(config['preparation_reference'])
            if digest(reference['manifest'])!=reference['sha256']:
                raise ValueError('preparation evidence changed')
            preparation=read(reference['manifest'])
            if preparation.get('status')!='completed' or preparation.get('training_started') or preparation.get('teacher_generation_started'):
                raise ValueError('invalid asset/data preparation evidence')
            write_json(record.path/'preparation.json',{'reference':reference,'evidence':preparation},exclusive=True)
        dependencies = {}
        candidates = [Path(item["identity"]["path"]).parent for item in inputs]
        candidates += [Path(config[key]).parent for key in ("checkpoint","codec_selection","resume_checkpoint","overlap_audit","latent_cache") if config.get(key)]
        if config.get("parity_run"):
            candidates.append(Path(config["parity_run"]))
        for candidate in candidates:
            for base in (candidate,*candidate.parents):
                if (base/"manifest.json").is_file():
                    candidate_manifest = read(base/"manifest.json")
                    if not isinstance(candidate_manifest,dict) or candidate_manifest.get("run_id") != base.name:
                        continue
                    parent = validate_run(base)
                    if parent["run_id"] != record.id:
                        from ncm_lab.runtime.io import digest
                        dependencies[parent["run_id"]] = {"run_id":parent["run_id"],"path":str(base),
                            "manifest_sha256":digest(base/"manifest.json")}
                    break
        write_json(record.path/"dependencies.json",list(dependencies.values()),exclusive=True)
        if config.get("result_origin") != "test_fixture" and config['executor'] not in {'overlap_audit','cost_selection'}:
            from ncm_lab.models.assets import verify_model_assets
            roles = ('target',) if config['executor'] in {'teacher','codec','codec_evaluate','latent_cache'} or config.get('method')=='AR' else ('target','draft')
            write_json(record.path/'model_assets.json',verify_model_assets(config['model'],roles,meter),exclusive=True)
        metrics = executor(record, config, meter)
        codec = metrics.get("codec",{}).get("search_cost",{})
        metrics["resources"] = {"executor_wall_seconds":time.monotonic()-started,
            "target_processed_positions":metrics.get("new_invocation_target_positions",metrics.get("target_processed_positions",
                metrics.get("source_token_passes",codec.get("new_invocation_target_positions",meter.new_source_tokens)))),
            "optimizer_steps":metrics.get("new_invocation_optimizer_steps",metrics.get("optimizer_steps",codec.get("new_invocation_optimizer_steps",codec.get('optimizer_steps',0)))),
            "teacher_generated_tokens":metrics.get("teacher_generated_tokens",metrics.get("generated_teacher_tokens",0) if config["executor"]=="teacher" else 0),
            "storage_bytes":sum(p.stat().st_size for p in record.path.rglob("*") if p.is_file())}
        final = record.close(metrics.get("status", "completed"), metrics)
    except (BudgetExceeded, KeyboardInterrupt) as exc:
        final = record.close("cancelled", {'resources':incomplete_resources('interrupted or budget-limited operation; only elapsed time/charged inputs are known'),
            'cancel_reason':'user_interrupt' if isinstance(exc,KeyboardInterrupt) else 'budget'}, str(exc))
    except Exception as exc:
        final = record.close("failed", {'resources':incomplete_resources('failed operation; completed model work is unknown')}, f"{type(exc).__name__}: {exc}")
    return {"run_id": record.id, "path": str(record.path), "status": final["status"], "error": final["error"]}
