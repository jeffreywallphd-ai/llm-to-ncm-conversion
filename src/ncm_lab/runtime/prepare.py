"""One-command asset preparation; generation and training remain explicit runs."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shutil
import uuid
from ncm_lab.data.download import PreparationBudget, AcquisitionMeter, asset_path, download_asset, atomic_json
from ncm_lab.runtime.io import root, read, digest, canonical_hash, write_json


def selected_assets(include_financial=False,with_test_data=False):
    lock = read(root()/"third_party/preparation_assets.lock.json")
    if lock["schema_version"] != 1:
        raise ValueError("unsupported preparation asset lock")
    groups = {"pilot"}
    if include_financial:
        groups |= {"financial-notices",*(f"{dataset}-{split}" for dataset in ("finqa","tatqa","docfinqa") for split in ("train","dev"))}
    if with_test_data:
        groups.add("gsm8k-test")
        if include_financial:
            groups |= {f"{dataset}-test" for dataset in ("finqa","tatqa","docfinqa")}
    selected = [asset for asset in lock["assets"] if groups & set(asset["groups"])]
    ids = set()
    for asset in selected:
        if asset["id"] in ids or not isinstance(asset["size_bytes"],int) or isinstance(asset["size_bytes"],bool) or asset["size_bytes"] < 0:
            raise ValueError("duplicate or invalid preparation asset")
        if not re.fullmatch(r"[0-9a-f]{40}",asset["revision"]) or not asset["url"].startswith("https://"):
            raise ValueError("preparation assets require immutable revisions and HTTPS")
        if not (asset.get("sha256") or asset.get("git_blob_sha1")):
            raise ValueError("preparation asset has no pinned content identity")
        ids.add(asset["id"])
    return selected


def resolve_inputs(assets_root=None,budget_path=None):
    base = Path(assets_root).expanduser() if assets_root else root()/".cache/research-assets"
    if not base.is_absolute():
        base = root()/base
    path = Path(budget_path) if budget_path else root()/"configs/preparation/pilot.yaml"
    if not path.is_absolute():
        path = root()/path
    return base.resolve(),PreparationBudget(**read(path))


def disk_free(base):
    existing = base
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def plan(base,budget,assets):
    missing,partial_bytes = 0,0
    for asset in assets:
        path = asset_path(base,asset)
        if path.is_file() and path.stat().st_size == asset["size_bytes"]:
            continue  # Candidate only: the executable command re-hashes every cached file.
        partial = path.with_suffix(path.suffix+".partial")
        partial_size = partial.stat().st_size if partial.is_file() else 0
        if partial_size > asset["size_bytes"]:
            raise ValueError(f"oversized partial download: {partial}")
        partial_bytes += partial_size
        missing += asset["size_bytes"]-partial_size
    total = sum(a["size_bytes"] for a in assets)
    free = disk_free(base)
    required_free = missing+budget.max_processed_bytes+budget.reserve_free_bytes
    existing_storage = sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) if base.exists() else 0
    blockers = []
    if free < required_free:
        blockers.append(f"insufficient free disk: {free:,} bytes available, {required_free:,} required; use --assets-root on a larger disk")
    if existing_storage+missing+budget.max_processed_bytes > budget.max_storage_bytes:
        blockers.append("selected assets exceed the configured storage budget")
    ledger = base/"preparations"/f"{budget.phase_budget_id}.ledger.json"
    charged = read(ledger)["charged_download_bytes"] if ledger.exists() else 0
    if charged+missing > budget.max_download_bytes:
        blockers.append("remaining aggregate download budget cannot cover the missing asset bytes")
    return {"assets_root":str(base),"asset_count":len(assets),"total_asset_bytes":total,
        "estimated_remaining_download_bytes":missing,"partial_bytes":partial_bytes,"charged_download_bytes":charged,
        "free_disk_bytes":free,"required_free_bytes":required_free,"budget":asdict(budget),"blockers":blockers,
        "existing_asset_directory_bytes":existing_storage,
        "cache_note":"sizes are candidates only; full locked checksum verification occurs during preparation",
        "network_started":False,"training_started":False,"teacher_generation_started":False,
        "includes_test_data":any(a.get("split")=="test" for a in assets)}


def prepare(assets_root=None,budget_path=None,profile="spark",dry_run=False,include_financial=False,
            with_test_data=False,progress=None):
    base,budget = resolve_inputs(assets_root,budget_path)
    if include_financial or with_test_data:
        budget = replace(budget,phase_budget_id=budget.phase_budget_id+("-financial" if include_financial else "")+("-tests" if with_test_data else ""))
    assets = selected_assets(include_financial,with_test_data)
    overview = plan(base,budget,assets)
    if dry_run:
        return {"status":"planned",**overview}
    if overview["blockers"]:
        return {"status":"blocked",**overview}
    if importlib.util.find_spec("pyarrow") is None:
        return {"status":"blocked",**overview,"blockers":["install the project prepare extra (pyarrow) before downloading; see docs/prepare.md"]}
    base.mkdir(parents=True,exist_ok=True)
    lock = base/".prepare.lock"
    try:
        with lock.open("x",encoding="utf-8") as stream:
            stream.write("Preparation in progress; inspect the previous attempt before removing a stale lock.\n")
    except FileExistsError:
        return {"status":"blocked",**overview,"blockers":[f"preparation already locked: {lock}; inspect the active/interrupted attempt"]}
    attempt = base/"preparations"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:10])
    meter = None
    try:
        attempt.mkdir(parents=True,exist_ok=False)
        write_json(attempt/"config.json",{"profile":profile,"budget":asdict(budget),"assets":assets,
                   "asset_lock_sha256":digest(root()/"third_party/preparation_assets.lock.json")},exclusive=True)
        meter = AcquisitionMeter(base,budget,canonical_hash(assets))
        if progress:
            progress(f"Preparing {len(assets)} pinned files ({overview['total_asset_bytes']/1e9:.2f} GB), with checksums and recovery.")
        receipts = []
        for index,asset in enumerate(assets,1):
            meter.check()
            if progress:
                progress(f"[{index}/{len(assets)}] {asset['id']} ({asset['size_bytes']/1e6:.1f} MB)")
            receipts.append(download_asset(base,asset,meter,progress=progress))
            with (attempt/"downloads.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(receipts[-1])+"\n")
        from ncm_lab.data.prepare_prompts import prepare_prompts
        oasst = next(a for a in assets if a["role"]=="oasst1" and a["filename"].endswith(".parquet"))
        gsm = next(a for a in assets if a["role"]=="gsm8k" and a["split"]=="train")
        prompts = prepare_prompts(asset_path(base,oasst),asset_path(base,gsm),attempt/"processed",budget)
        meter.check()
        from ncm_lab.runtime.doctor import doctor
        environment = doctor(profile)
        configs = write_launch_configs(attempt,base,assets,prompts,profile,budget)
        cuda = environment["torch"]["cuda_available"]
        hardware_ok = environment["spark_ready"] if profile=="spark" else cuda if profile=="cuda" else False
        blockers = [] if hardware_ok else ["assets are prepared; a verified CUDA/Spark runtime and sufficient model/training memory are still required"]
        result = {"schema_version":1,"status":"completed","attempt":str(attempt),"assets_root":str(base),
            "assets_verified":True,"environment":environment,"hardware_matches_profile":hardware_ok,
            "gpu_parity_validated":False,"ready_for_training":False,"remaining_prerequisites":blockers+[
                "run the bounded teacher-generation step", "real-model correctness/parity and training-memory validation remain unverified"],
            "prompts":prompts,"configs":configs,"download_receipts":receipts,"training_started":False,
            "teacher_generation_started":False,"research_findings":[]}
        write_json(attempt/"manifest.json",result,exclusive=True)
        atomic_json(base/"preparations/latest.json",{"manifest":str(attempt/"manifest.json"),"sha256":digest(attempt/"manifest.json")})
        return result
    except (Exception,KeyboardInterrupt) as exc:
        result = {"status":"cancelled" if isinstance(exc,KeyboardInterrupt) else "failed","attempt":str(attempt),
                  "error":f"{type(exc).__name__}: {exc}","recovery":"repeat the same prepare command; verified files are reused and compatible partial downloads resume"}
        if attempt.exists() and not (attempt/"manifest.json").exists():
            write_json(attempt/"manifest.json",result,exclusive=True)
        return result
    finally:
        if meter:
            meter.save()
        lock.unlink(missing_ok=True)


def write_launch_configs(attempt,base,assets,prompts,profile,budget):
    model = read(root()/"configs/models/qwen3_8b_dflash.yaml")
    for role in ("target","draft"):
        asset = next(a for a in assets if a["role"]==role and a["filename"]=="config.json")
        model[role+"_path"] = str(asset_path(base,asset).parent)
    device,precision = ("cpu","float32") if profile=="cpu" else ("cuda","bfloat16")
    teacher = {"experiment_id":"SHR-001","executor":"teacher","profile":profile,"device":device,
        "precision":precision,"result_origin":"measured","model":model,"prompts_path":prompts["outputs"]["train"]["path"],
        "max_new_tokens":128,"max_examples":100,"teacher_output_reference":str(attempt/"teacher_output.json"),
        "artifact_root":str(base/"runs")}
    train = read(root()/"experiments/acl/ACL-001_component_profile_and_pilot/configs/run.yaml")
    train.update(model=model,profile=profile,device=device,precision=precision,
        required_files=[str(attempt/"manifest.json")],artifact_root=str(base/"runs"),
        training_data_reference=str(attempt/"teacher_output.json"))
    train.pop("training_data",None)
    launch = attempt/"launch"
    write_json(launch/"teacher.yaml",teacher,exclusive=True)
    write_json(launch/"pilot.yaml",train,exclusive=True)
    pilot_budget = read(root()/"configs/budgets/pilot.yaml")
    pilot_budget["phase_budget_id"] = "pilot-"+attempt.name
    teacher_budget = {**pilot_budget,"phase_budget_id":"teacher-"+attempt.name}
    write_json(launch/"pilot_budget.yaml",pilot_budget,exclusive=True)
    write_json(launch/"teacher_budget.yaml",teacher_budget,exclusive=True)
    gsm = next(a for a in assets if a["role"]=="gsm8k" and a["split"]=="train")
    parity = read(root()/"experiments/shared/SHR-002_target_and_verification/configs/run.yaml")
    parity.update(model=model,profile=profile,device=device,precision=precision,
                  data_path=str(asset_path(base,gsm)),split="train",artifact_root=str(base/"runs"))
    write_json(launch/"verification.yaml",parity,exclusive=True)
    return {"teacher_config":str(launch/"teacher.yaml"),"teacher_budget":str(launch/"teacher_budget.yaml"),
        "pilot_config":str(launch/"pilot.yaml"),"pilot_budget":str(launch/"pilot_budget.yaml"),
        "verification_config":str(launch/"verification.yaml"),
        "next_commands":[f'ncm experiment run --id SHR-001 --config "{launch / "teacher.yaml"}" --budget "{launch / "teacher_budget.yaml"}"',
                         f'ncm experiment run --id ACL-001 --config "{launch / "pilot.yaml"}" --budget "{launch / "pilot_budget.yaml"}"']}
