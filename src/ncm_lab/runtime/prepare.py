"""Explicit bounded asset acquisition; planning never retrieves models or starts stages."""

from dataclasses import asdict
import shutil
from ncm_lab.runtime.io import root, read, write_json, digest, canonical_hash
from ncm_lab.data.download import (
    PreparationBudget,
    AcquisitionMeter,
    download_asset,
    asset_path,
)


def prepare(value, plan_only=True):
    settings = value["preparation"]
    assets = read(root() / settings["asset_lock"])["assets"]
    base = root() / settings["assets_root"]
    budget = PreparationBudget(**settings["budget"])
    needed = []
    # This plan checks existence/size only. Full content validation occurs on explicit acquisition.
    for a in assets:
        path = asset_path(base, a)
        if not path.is_file() or path.stat().st_size != a["size_bytes"]:
            needed.append(a)
    missing = sum(a["size_bytes"] for a in needed)
    storage = sum(a["size_bytes"] for a in assets)
    free = shutil.disk_usage(root()).free
    result = {
        "schema": "native_preparation_plan_v1",
        "plan_only": plan_only,
        "model_downloads_started": False,
        "training_started": False,
        "assets": len(assets),
        "missing_payload_bytes": missing,
        "asset_storage_bytes": storage,
        "free_disk_bytes": free,
        "budget": asdict(budget),
        "stages": settings["stages"],
        "cache_estimates_decimal_bytes": {
            "10m_teacher_states_bf16": 10000000 * 4096 * 2,
            "10m_fine_codes_bf16": 10000000 * 384 * 2,
        },
        "blockers": [],
    }
    if free < missing + budget.reserve_free_bytes:
        result["blockers"].append("insufficient free disk for missing assets plus reserve")
    if missing > budget.max_download_bytes or storage > budget.max_storage_bytes:
        result["blockers"].append("acquisition budget cannot cover pinned selected assets")
    if plan_only:
        return result
    if result["blockers"]:
        return {**result, "status": "blocked"}
    base.mkdir(parents=True, exist_ok=True)
    meter = AcquisitionMeter(base, budget, canonical_hash(assets))
    try:
        receipts = []
        for asset in assets:
            output = download_asset(base, asset, meter)
            path = asset_path(base, asset)
            receipts.append(
                {
                    "id": asset["id"],
                    "path": str(path),
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "license": asset["license"],
                }
            )
        output = base / "native-preparation.json"
        write_json(
            output,
            {
                "schema": "native_asset_preparation_v1",
                "assets": receipts,
                "budget": asdict(budget),
                "ledger": str(meter.path),
                "training_started": False,
            },
        )
        return {
            **result,
            "plan_only": False,
            "model_downloads_started": bool(needed),
            "status": "completed",
            "artifact": str(output),
            "next_commands": [
                "ncm " + command + " --config " + path for command, path in settings["stages"].items()
            ],
        }
    finally:
        meter.save()
