import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from ncm_lab.reporting.reports import selected_runs, build
from ncm_lab.runtime.io import root, digest


def review_export(study, output, anonymize=False):
    spec, selected = selected_runs(study, require_current=False)
    evidence=dict((run['run_id'],(path,run)) for path,run in selected)
    pending=list(evidence.values())
    from ncm_lab.runtime.provenance import validate_run
    from ncm_lab.runtime.io import read
    while pending:
        path,run=pending.pop()
        if (path/'dependencies.json').exists():
            for dependency in read(path/'dependencies.json'):
                parent=Path(dependency['path'])
                manifest=validate_run(parent)
                if digest(parent/'manifest.json') != dependency['manifest_sha256']:
                    raise ValueError('export evidence dependency changed')
                if manifest['run_id'] not in evidence:
                    evidence[manifest['run_id']]=(parent,manifest)
                    pending.append((parent,manifest))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("export destination already exists; use a new path")
    candidates = [root() / "LICENSE", root() / "pyproject.toml", root() / "uv.lock"]
    from ncm_lab.runtime.provenance import source_file
    for directory in ("src", "third_party", "configs", f"publications/{study}"):
        candidates += [p for p in (root() / directory).rglob("*") if source_file(p)]
    for eid in spec["required_experiments"] + spec["shared_controls"]:
        from ncm_lab.experiments.registry import get
        base = root() / get(eid)["directory"]
        candidates += [base / "README.md", base / "protocol.yaml", base / "analysis.py", *list((base / "configs").rglob("*"))]
    candidates += [root() / "experiments/registry.yaml"]
    for dataset in spec["eligible_datasets"]:
        candidates += [root() / "data/manifests" / f"{dataset}.yaml"]
    candidates += [root() / "docs" / n for n in ("architecture.md", "data_provenance.md", "evaluation_protocol.md", "references.bib")]
    external, original_manifests = [], {}
    extras={}
    def include_analysis(path):
        path=Path(path).resolve()
        if path in extras:
            return
        record=read(path)
        extras[path]='assessments/'+digest(path)+'.json'
        for name,sha in record.get('files',{}).items():
            source=(path.parent/name).resolve()
            if not source.is_relative_to(path.parent) or digest(source)!=sha:
                raise ValueError('analysis dependency missing/changed')
            extras[source]='analysis/'+sha+source.suffix
        derived=record.get('derived_tables_manifest')
        if derived:
            if digest(derived['path'])!=derived['sha256']:
                raise ValueError('derived analysis receipt changed')
            include_analysis(derived['path'])
    for entry in spec['selected_runs']:
        if entry.get('attestation'):
            path=Path(entry['attestation']).resolve()
            extras[path]='assessments/'+digest(path)+'.json'
            assessment=read(path)
            if assessment.get('analysis'):
                if digest(assessment['analysis']['path'])!=assessment['analysis']['sha256']:
                    raise ValueError('selected analysis attestation changed')
                include_analysis(assessment['analysis']['path'])
    candidates+=list(extras)
    for path, run in evidence.values():
        from ncm_lab.runtime.provenance import implementation_hash
        snapshot = path/"source_snapshot.zip"
        if run["code"].get("implementation_hash") != implementation_hash():
            if not snapshot.is_file() or digest(snapshot) != run["code"].get("source_snapshot_sha256"):
                raise ValueError("older evidence requires its preserved study source snapshot")
        original_manifests[run["run_id"]] = run
        candidates += [p for p in path.rglob("*") if p.is_file()]
        if (path/'model_assets.json').exists():
            for asset in read(path/'model_assets.json'):
                external.append({'artifact':'model/'+asset['role']+'/'+asset['filename'],
                    'sha256':asset['sha256'],'bytes':asset['bytes'],'resolver':'file named by sha256 in reviewer cache'})
        if (path/'inputs.json').exists():
            for item in read(path/'inputs.json'):
                data=item['identity']
                external.append({'artifact':'input/'+data['schema']+'/'+data['split'],'sha256':data['sha256'],
                    'bytes':Path(data['path']).stat().st_size,'resolver':'file named by sha256 in reviewer cache'})
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
        if not path.is_file() or path.suffix == ".pyc":
            continue
        owner = next(((base,run) for base,run in evidence.values() if path.is_relative_to(base)),None)
        relative = extras.get(path.resolve()) or (("evidence/"+owner[1]["run_id"]+"/"+path.relative_to(owner[0]).as_posix()) if owner else path.relative_to(root()).as_posix())
        if path.suffix in {".pt", ".safetensors"} or path.stat().st_size > 16*1024*1024:
            external.append({"artifact": relative, "sha256": digest(path), "bytes": path.stat().st_size,
                             "resolver": "file named by sha256 in explicitly provided reviewer cache"})
            continue
        if path.suffix == ".zip":
            members[relative] = path.read_bytes()
            continue
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
        "Inspect EXPERIMENT_REPORT.md for selected observations and their outstanding evidence gates.\n"
        "See EXPERIMENT_REPORT.md and REVIEW_REPORT.json. Third-party licenses are preserved.\n").encode()
    members["REVIEW_REPORT.json"] = json.dumps({"study": study, "anonymized": anonymize,
        "selected_eligible_runs": len(selected), "redacted_files": redactions,
        "third_party_attribution_preserved": True, "human_anonymity_review_required": True,
        "submission_ready": False, "source_tree_digest": "see internal receipt"}, indent=2).encode()
    from ncm_lab.reporting.portable import validator_source
    members['validate_export.py']=validator_source()
    hashes = {name: hashlib.sha256(body).hexdigest() for name, body in sorted(members.items())}
    members["ARCHIVE_CONTENTS.json"] = json.dumps(hashes, indent=2).encode()
    hashes["ARCHIVE_CONTENTS.json"] = hashlib.sha256(members["ARCHIVE_CONTENTS.json"]).hexdigest()
    members["EXPORT_MANIFEST.json"] = json.dumps({"schema_version": 1, "files": hashes,
        "external_artifacts": external, "research_eligible": bool(selected),
        "integrity_contract": "export hashes bind transformed bytes; original manifests remain source references"},indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            archive.writestr(name, body)
    internal = output.with_suffix(output.suffix + ".internal.json")
    from ncm_lab.runtime.provenance import source_tree_hash
    internal.write_text(json.dumps({"archive_sha256": digest(output), "source_tree_hash": source_tree_hash(),
                                    "study": study, "anonymized": anonymize,
                                    "source_manifests": original_manifests}, indent=2), encoding="utf-8")
    return {"path": str(output), "sha256": digest(output), "internal_receipt": str(internal),
            "selected_eligible_runs": len(selected), "submission_ready": False}
