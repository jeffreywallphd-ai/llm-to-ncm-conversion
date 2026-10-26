"""Refresh pinned acquisition metadata only (no model/data payload downloads).

Requires the existing source revisions. Inspect changes to the resulting lock before
using them: runtime preparation never follows a moving branch or executes model code.
"""
import fnmatch
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MAX_METADATA_BYTES = 4_000_000


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ncm-lab-asset-lock"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(MAX_METADATA_BYTES + 1)
    if len(body) > MAX_METADATA_BYTES:
        raise ValueError("metadata response exceeds audit cap")
    return json.loads(body)


def main():
    sources = json.loads((ROOT / "third_party/sources.lock.yaml").read_text())["sources"]
    assets = []
    for role, source_key, repo, repo_type, names in [
        ("target", "target", "Qwen/Qwen3-8B", "models", None),
        ("draft", "drafter", "z-lab/Qwen3-8B-DFlash-b16", "models", None),
        ("oasst1", "oasst1", "OpenAssistant/oasst1", "datasets", [
            "data/train-00000-of-00001-b42a775f407cee45.parquet", "README.md", "LICENSE"]),
        ("docfinqa", "docfinqa", "kensho/DocFinQA", "datasets", ["train.json", "dev.json", "test.json", "README.md"]),
    ]:
        revision = sources[source_key]["revision"]
        info = get_json(f"https://huggingface.co/api/{repo_type}/{repo}/revision/{revision}?blobs=true")
        if info["sha"] != revision:
            raise ValueError("metadata revision mismatch")
        selected = []
        for item in info["siblings"]:
            name = item["rfilename"]
            patterns = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
                        "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                        "chat_template.jinja", "README.md", "LICENSE", "LICENSE.txt"]
            if names is not None and name not in names:
                continue
            if names is None and not any(fnmatch.fnmatch(name, p) for p in patterns):
                continue
            lfs = item.get("lfs") or {}
            if not isinstance(item.get("size"), int):
                raise ValueError(f"asset size missing: {repo}/{name}")
            split = Path(name).stem if name in {"train.json", "dev.json", "test.json"} else None
            groups = ["pilot"] if role != "docfinqa" else [f"docfinqa-{split}"] if split else ["financial-notices"]
            source = f"https://huggingface.co/{'datasets/' if repo_type == 'datasets' else ''}{repo}"
            asset = {"id": f"{role}:{name}", "role": role, "repo_id": repo, "revision": revision,
                     "filename": name, "split": "train" if role == "oasst1" and name.endswith(".parquet") else split,
                     "path": f"{role}/{revision}/{name}", "url": f"{source}/resolve/{revision}/{name}",
                     "size_bytes": item["size"], "sha256": lfs.get("sha256"),
                     "git_blob_sha1": None if lfs else item["blobId"], "groups": groups,
                     "license": info.get("cardData", {}).get("license", "see included source notice")}
            assets.append(asset)
            selected.append(name)
        if names is None and ("config.json" not in selected or not any(n.endswith(".safetensors") for n in selected)):
            raise ValueError(f"incomplete safetensors model release: {repo}")
    for dataset, repo, paths in [
        ("gsm8k", "openai/grade-school-math", {"train": "grade_school_math/data/train.jsonl", "test": "grade_school_math/data/test.jsonl"}),
        ("finqa", "czyssrs/FinQA", {s: f"dataset/{s}.json" for s in ("train", "dev", "test")}),
        ("tatqa", "NExTplusplus/TAT-QA", {"train": "dataset_raw/tatqa_dataset_train.json", "dev": "dataset_raw/tatqa_dataset_dev.json", "test": "dataset_raw/tatqa_dataset_test_gold.json"}),
    ]:
        revision = sources[dataset]["revision"]
        tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1")
        files = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
        for split, name in {**paths, "notice": "LICENSE"}.items():
            item = files[name]
            groups = ["pilot"] if dataset == "gsm8k" and split in {"train", "notice"} else [f"{dataset}-{split}"]
            if split == "notice" and dataset != "gsm8k":
                groups = ["financial-notices"]
            assets.append({"id": f"{dataset}:{split}", "role": dataset, "repo_id": repo, "revision": revision,
                "filename": Path(name).name, "split": split, "path": f"{dataset}/{revision}/{Path(name).name}",
                "url": f"https://raw.githubusercontent.com/{repo}/{revision}/{name}", "size_bytes": item["size"],
                "sha256": None, "git_blob_sha1": item["sha"], "groups": groups, "license": "MIT"})
    value = {"schema_version": 1, "audit_date": "2026-10-25", "assets": assets,
             "notes": "Pinned file hashes/sizes; safetensors only; test payloads excluded from pilot preparation."}
    dest = ROOT / "third_party/preparation_assets.lock.json"
    dest.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(dest), "assets": len(assets), "pilot_bytes": sum(a["size_bytes"] for a in assets if "pilot" in a["groups"])}))


if __name__ == "__main__":
    main()
