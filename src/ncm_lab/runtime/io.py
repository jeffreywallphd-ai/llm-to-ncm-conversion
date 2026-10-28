from pathlib import Path
import hashlib
import json
import os
import yaml


def root() -> Path:
    configured = os.environ.get("NCM_REPO_ROOT")
    candidates = [Path(configured)] if configured else [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    candidates += list(Path(__file__).resolve().parents)
    for p in candidates:
        if (p / "experiments/registry.yaml").is_file() and (p / "pyproject.toml").is_file():
            return p
    raise ValueError("repository configuration not found; set NCM_REPO_ROOT to the checkout")


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        text = f.read()
    return json.loads(text) if text.lstrip().startswith(("{", "[")) else yaml.safe_load(text)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
