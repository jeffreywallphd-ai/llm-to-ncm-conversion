"""Fetch <=512 KiB of released TRAIN data, plus <=2 MB of source metadata/code."""
import hashlib
import json
import urllib.request
from ncm_lab.data.acquire import training_sample
from ncm_lab.runtime.io import root, read


def main():
    base = root()
    lock = read(base / "third_party/sources.lock.yaml")
    for dataset, repo, path in [("finqa", "czyssrs/FinQA", "dataset/train.json"),
                                ("tatqa", "NExTplusplus/TAT-QA", "dataset_raw/tatqa_dataset_train.json")]:
        rev = lock["sources"][dataset]["revision"]
        url = f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"
        print(training_sample(url, base / f"tests/fixtures/{dataset}_train_sample.json", count=2), flush=True)
    revision = lock["sources"]["dcore"]["revision"]
    paths = ["specforge/core/dflash.py", "examples/colocated/qwen3_8b/dflash/5l_bs16_5hs_ce_exp.sh",
             "tests/test_utils/test_dflash_losses.py", "tests/test_utils/test_dflash_mask.py",
             "specforge/modeling/draft/dflash.py"]
    for path in paths:
        url = f"https://raw.githubusercontent.com/ernie-research/DCore/{revision}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read(400001)
        if len(body) > 400000:
            raise ValueError("source inspection cap exceeded")
        dest = base / ".cache/audit" / ("dcore--" + path.replace("/", "__"))
        dest.write_bytes(body)
        lock["sources"]["dcore"]["files"][path] = hashlib.sha256(body).hexdigest()
    (base / "third_party/sources.lock.yaml").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
