"""Content identities and small atomic metadata; no academic-registry dependency."""

import hashlib
import json
import os
from pathlib import Path
import yaml


def root():
    configured = os.environ.get("NCM_REPO_ROOT")
    candidates = [Path(configured)] if configured else [Path.cwd(), *Path.cwd().parents]
    candidates += list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / ".ncm-lab").is_file():
            return candidate.resolve()
    raise ValueError("native project marker missing; set NCM_REPO_ROOT")


def read(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.lstrip().startswith(("{", "[")) else yaml.safe_load(text)


def digest(path, check=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            if check:
                check()
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    write_json(temporary, value)
    os.replace(temporary, path)


def reference(path):
    path = Path(path).resolve()
    return {"schema": "native_reference_v1", "manifest": str(path), "sha256": digest(path)}


def resolve_reference(path):
    path = Path(path)
    if not path.is_absolute():
        path = root() / path
    value = read(path)
    if value.get("schema") == "native_reference_v1":
        target = Path(value["manifest"])
        if digest(target) != value["sha256"]:
            raise ValueError("handoff manifest changed")
        return target
    return path


def implementation_hash():
    package = Path(__file__).resolve().parents[1]
    return canonical_hash(
        [(p.relative_to(package).as_posix(), digest(p)) for p in sorted(package.rglob("*.py"))]
    )
