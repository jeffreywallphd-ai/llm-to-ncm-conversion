"""Bounded acquisition/preprocessing check on official training data, no weights.

Downloads only the locked OASST train parquet and GSM8K train file (<=64 MiB total).
Produces preparation receipts, not model metrics or publication findings.
"""
from dataclasses import replace
import json
from ncm_lab.data.download import AcquisitionMeter, asset_path, download_asset
from ncm_lab.data.prepare_prompts import prepare_prompts
from ncm_lab.runtime.io import root
from ncm_lab.runtime.prepare import selected_assets, resolve_inputs


def main():
    base = root()/".cache/prepare-real-data-smoke"
    budget = replace(resolve_inputs()[1],phase_budget_id="real-training-data-smoke",max_download_bytes=67108864,
                     max_wall_seconds=300,max_phase_wall_seconds=600,max_prompt_examples=128)
    assets = [a for a in selected_assets() if (a["role"]=="oasst1" and a["filename"].endswith(".parquet"))
              or (a["role"]=="gsm8k" and a["split"]=="train")]
    meter = AcquisitionMeter(base,budget,"oasst-train-and-gsm8k-train-only")
    for asset in assets:
        print(json.dumps(download_asset(base,asset,meter)),flush=True)
    output = base/"processed"
    if not (output/"prompt_provenance.json").exists():
        source = next(a for a in assets if a["role"]=="oasst1")
        gsm = next(a for a in assets if a["role"]=="gsm8k")
        result = prepare_prompts(asset_path(base,source),asset_path(base,gsm),output,budget)
        print(json.dumps(result,indent=2))
    meter.save()


if __name__ == "__main__":
    main()
