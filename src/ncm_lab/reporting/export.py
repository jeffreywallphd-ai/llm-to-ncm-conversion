import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from ncm_lab.reporting.reports import selected_runs, build
from ncm_lab.runtime.io import root, digest


def review_export(study, output, anonymize=False):
    spec, selected = selected_runs(study, require_current=True)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("export destination already exists; use a new path")
    candidates = [root() / "LICENSE", root() / "pyproject.toml", root() / "uv.lock"]
    for directory in ("src", "third_party", "configs", f"publications/{study}"):
        candidates += [p for p in (root() / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for eid in spec["required_experiments"] + spec["shared_controls"]:
        from ncm_lab.experiments.registry import get
        base = root() / get(eid)["directory"]
        candidates += [base / "README.md", base / "protocol.yaml", base / "analysis.py", *list((base / "configs").glob("*"))]
    candidates += [root() / "experiments/registry.yaml"]
    for dataset in spec["eligible_datasets"]:
        candidates += [root() / "data/manifests" / f"{dataset}.yaml"]
    candidates += [root() / "docs" / n for n in ("architecture.md", "data_provenance.md", "evaluation_protocol.md", "references.bib")]
    for path, run in selected:
        candidates += [path / name for name in ("manifest.json","metrics.json","config.json","protocol.json","environment.json")]
    report = Path(build(study)["path"])
    members, redactions = {}, []
    identity_tokens = []
    if anonymize:
        try:
            remote = subprocess.run(["git","remote","get-url","origin"],cwd=root(),capture_output=True,
                                    text=True,timeout=10,check=True).stdout.strip().removesuffix(".git")
            match = re.search(r"github\.com[:/]([^/]+)/([^/]+)$",remote)
            identity_tokens.append((remote,"[review repository]"))
            if match:
                identity_tokens.append((match[1],"review-owner"))
        except (OSError,subprocess.SubprocessError):
            pass
    for path in candidates:
        if not path.is_file() or path.suffix in {".pyc", ".pt", ".safetensors"}:
            continue
        relative = path.relative_to(root()).as_posix()
        if study == "acl" and relative.startswith("src/ncm_lab/business/"):
            continue
        if relative == "experiments/registry.yaml":
            from ncm_lab.runtime.io import read
            registry = read(path)
            allowed = set(spec["required_experiments"] + spec["shared_controls"])
            registry["experiments"] = [e for e in registry["experiments"] if e["id"] in allowed]
            text = json.dumps(registry, indent=2)
        else:
            text = path.read_text(encoding="utf-8")
        if anonymize and "/vendor/" not in relative and not relative.endswith("LICENSE"):
            original = text
            for identity, replacement in identity_tokens:
                text = text.replace(identity,replacement)
            text = text.replace(str(root()), "[workspace]").replace(root().as_posix(), "[workspace]")
            text = re.sub(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\s\"']*", "[local path]", text)
            if text != original:
                redactions.append(relative)
        members[relative] = text.encode()
    members["EXPERIMENT_REPORT.md"] = report.read_bytes()
    members["README.md"] = (f"# {study} review artifact\n\n"
        "This is a study-specific source/protocol export, not a finished paper or benchmark result.\n"
        "Install with `uv sync --locked --extra cpu --extra scoring --group dev`, then use\n"
        "`uv run --no-sync ncm experiment list` and the included experiment READMEs.\n"
        "No empirical findings yet. GPU, official task harness and research evidence remain pending.\n"
        "See EXPERIMENT_REPORT.md and REVIEW_REPORT.json. Third-party licenses are preserved.\n").encode()
    members["REVIEW_REPORT.json"] = json.dumps({"study": study, "anonymized": anonymize,
        "selected_eligible_runs": len(selected), "redacted_files": redactions,
        "third_party_attribution_preserved": True, "human_anonymity_review_required": True,
        "submission_ready": False, "source_tree_digest": "see internal receipt"}, indent=2).encode()
    hashes = {name: hashlib.sha256(body).hexdigest() for name, body in sorted(members.items())}
    members["ARCHIVE_CONTENTS.json"] = json.dumps(hashes, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            archive.writestr(name, body)
    internal = output.with_suffix(output.suffix + ".internal.json")
    from ncm_lab.runtime.provenance import source_tree_hash
    internal.write_text(json.dumps({"archive_sha256": digest(output), "source_tree_hash": source_tree_hash(),
                                    "study": study, "anonymized": anonymize}, indent=2), encoding="utf-8")
    return {"path": str(output), "sha256": digest(output), "internal_receipt": str(internal),
            "selected_eligible_runs": len(selected), "submission_ready": False}
