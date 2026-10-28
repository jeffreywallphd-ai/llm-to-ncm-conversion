"""Generate acquisition manifests from the inspected source lock, without downloads."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / "third_party/sources.lock.yaml").read_text())["sources"]
    specs = {
        "finqa": ("czyssrs/FinQA", "MIT", "finqa_family", {
            "train": ("dataset/train.json", 78216616, 6251), "dev": ("dataset/dev.json", 10954658, 883),
            "test": ("dataset/test.json", 14395143, 1147)}),
        "tatqa": ("NExTplusplus/TAT-QA", "MIT", "tatqa", {
            "train": ("dataset_raw/tatqa_dataset_train.json", 12845647, 13215),
            "dev": ("dataset_raw/tatqa_dataset_dev.json", 1637431, 1668),
            "test": ("dataset_raw/tatqa_dataset_test_gold.json", 2167546, 1669)}),
        "gsm8k": ("openai/grade-school-math", "MIT", "gsm8k", {
            "train": ("grade_school_math/data/train.jsonl", None, 7473),
            "test": ("grade_school_math/data/test.jsonl", None, 1319)}),
    }
    for dataset,(repo,license_name,family,splits) in specs.items():
        rev = sources[dataset]["revision"]
        spec = {"schema_version": 1, "dataset": dataset, "source_url": f"https://github.com/{repo}",
            "revision": rev, "license": license_name, "family": family, "setup_download_budget_bytes": 0,
            "grouping": "report/context content hash; official source IDs preserved",
            "gold": "released original answers/programs/evidence; evaluator only",
            "count_policy": "expected counts are assertions to audit after a full release fetch, not forced values",
            "transformations": ["allowlist ModelInput fields", "retain raw source and receipt SHA256", "no label edits"],
            "splits": {name: {"filename": Path(path).name, "url": f"https://raw.githubusercontent.com/{repo}/{rev}/{path}",
                              "size_bytes": size, "expected_questions": count, "sha256": None}
                       for name,(path,size,count) in splits.items()}}
        put(dataset,spec)
    put("docfinqa", {"schema_version": 1, "dataset": "docfinqa", "source_url": "https://huggingface.co/datasets/kensho/DocFinQA",
        "revision": sources["docfinqa"]["revision"], "license": "MIT (dataset card); verify underlying filing terms",
        "family": "finqa_family", "setup_download_budget_bytes": 0,
        "structure": "released full filing text; original geometry/headers may be unavailable",
        "scorer_status": "authors evaluator not yet located; exact official reproduction blocked; no generated Python on host",
        "normalization": "must predeclare a development-only policy and sensitivity analysis before evaluation",
        "splits": {name: {"filename": name+".json", "expected_questions": count,
            "url": f"https://huggingface.co/datasets/kensho/DocFinQA/resolve/{sources['docfinqa']['revision']}/{name}.json",
            "sha256": None, "blocker": "multi-GB full filings; explicit download and storage budget required"}
            for name,count in [("train",5735),("dev",780),("test",922)]}})
    for dataset,url,revision,license_name,blocker in [
        ("oasst1","https://huggingface.co/datasets/OpenAssistant/oasst1",sources["oasst1"]["revision"],"Apache-2.0 (card; verify at acquisition)","use English undeleted approved training prompts; split entire message_tree_id and deduplicate evaluation overlap"),
        ("ifeval","https://github.com/google-research/google-research/tree/master/instruction_following_eval",sources.get('ifeval',{}).get('revision'),"Apache-2.0","use pinned evaluator source/data metadata; provision official environment before scoring"),
        ("humaneval_plus","https://github.com/evalplus/evalplus",sources["evalplus"]["revision"],"Apache-2.0","EvalPlus official task and isolated execution environment required; no host Python execution"),
        ("longbench","https://github.com/THUDM/LongBench",sources["longbench"]["revision"],"MIT code; inspect original GovReport dataset terms","pin the original GovReport release and official scoring dependencies; LongBench v2 is not a substitute")]:
        put(dataset,{"schema_version":1,"dataset":dataset,"source_url":url,"revision":revision,"license":license_name,
            "setup_download_budget_bytes":0,"gold_separation":"explicit loader allowlist","splits":{"train" if dataset=="oasst1" else "test":
                {"url":None,"blocker":blocker}},"reproduction_status":"not benchmark reproduced"})


def put(name,spec):
    path = ROOT / "data/manifests" / f"{name}.yaml"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and name in {'ifeval','longbench','humaneval_plus'}:
        existing=json.loads(path.read_text(encoding='utf-8'))
        # Preserve separately inspected immutable payload/scorer metadata.
        for key in ('data_release','data_version','version_source_sha256'):
            if key in existing:
                spec[key]=existing[key]
        if existing.get('revision') and not spec.get('revision'):
            spec['revision']=existing['revision']
        if existing['splits']['test'].get('git_blob_sha1'):
            spec['splits']['test']=existing['splits']['test']
    path.write_text(json.dumps(spec,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
