from pathlib import Path
import hashlib
import json
import os
import yaml


def root() -> Path:
    configured = os.environ.get("NCM_REPO_ROOT")
    candidates = [Path(configured)] if configured else [*Path.cwd().resolve().parents, Path.cwd().resolve()]
    candidates += list(Path(__file__).resolve().parents)
    for p in candidates:
        if (p / "experiments/registry.yaml").is_file() and (p / "pyproject.toml").is_file():
            return p
    raise ValueError("repository configuration not found; set NCM_REPO_ROOT to the checkout")


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
