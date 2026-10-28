"""Self-contained engineering evidence package with transformed-byte integrity."""
import hashlib
import json
from pathlib import Path
import re
import zipfile
from ncm_lab.runtime.io import root, digest, write_json
from ncm_lab.runtime.provenance import validate_run


def validator_source():
    return (
        "import hashlib,json,pathlib,sys\n"
        "base=pathlib.Path(sys.argv[1]).resolve(); spec=json.loads((base/'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))\n"
        "def sha(path):\n"
        " h=hashlib.sha256()\n"
        " with path.open('rb') as f:\n"
        "  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)\n"
        " return h.hexdigest()\n"
        "for name,expected in spec['files'].items():\n"
        " p=(base/name).resolve()\n"
        " assert p.is_relative_to(base) and p.is_file(), name\n"
        " assert sha(p)==expected, name\n"
        "for ref in spec.get('external_artifacts',[]):\n"
        " assert len(sys.argv)>2,'Supply content-addressed resolver directory'\n"
        " p=pathlib.Path(sys.argv[2])/ref['sha256']\n"
        " assert p.is_file() and sha(p)==ref['sha256'],ref['artifact']\n"
        "sys.path.insert(0,str(base/'src'))\n"
        "from ncm_lab.evaluation.tables import analyze_records\n"
        "rows=[]\n"
        "for p in (base/'evidence').rglob('predictions.jsonl'):\n"
        " rows.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines())\n"
        "analysis=analyze_records(rows)\n"
        "print(json.dumps({'valid':True,'evidence_class':spec.get('evidence_class','study_review'),'research_eligible':spec.get('research_eligible',False),'analysis_groups':len(analysis['per_seed'])}))\n"
    ).encode()


def engineering_export(run_path, output, anonymize=True):
    run_path, output = Path(run_path).resolve(), Path(output).resolve()
    run = validate_run(run_path)
    if run["result_origin"] != "test_fixture":
        raise ValueError("engineering export requires an explicitly labeled fixture run")
    members, transformations = {}, []
    candidates = [root()/"LICENSE", root()/"pyproject.toml", root()/"uv.lock"]
    from ncm_lab.runtime.provenance import source_file
    candidates += [p for p in (root()/"src").rglob("*") if source_file(p)]
    for path in candidates:
        members[path.relative_to(root()).as_posix()] = path.read_bytes()
    snapshot = run_path/"source_snapshot.zip"
    if snapshot.exists():
        with zipfile.ZipFile(snapshot) as archive:
            for name in archive.namelist():
                if ".." in Path(name).parts or name.startswith(("/", "\\")):
                    raise ValueError("unsafe source snapshot")
                members[name] = archive.read(name)
    for path in run_path.rglob("*"):
        if not path.is_file():
            continue
        original = path.read_bytes()
        transformed = original
        if anonymize and path.suffix in {".json", ".jsonl", ".yaml", ".md"}:
            text = original.decode("utf-8")
            text = text.replace(str(root()), "[workspace]").replace(root().as_posix(), "[workspace]")
            text = re.sub(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\s\"']*", "[private-path]", text)
            transformed = text.encode()
        name = "evidence/" + path.relative_to(run_path).as_posix()
        members[name] = transformed
        transformations.append({"name": name, "source_sha256": hashlib.sha256(original).hexdigest(),
                                "export_sha256": hashlib.sha256(transformed).hexdigest(),
                                "transformed": transformed != original})
    members["README.md"] = (b"Engineering fixture evidence only. From a trusted installed ncm-lab, run "
        b"ncm publication validate-export --path ARCHIVE.zip --reanalyze. "
        b"The optional packaged validator requires a verified external isolation boundary; do not execute archive code on the host.\n")
    members["validate_export.py"] = validator_source()
    spec = {"schema_version": 1, "evidence_class": "software_fixture", "research_eligible": False,
        "source_manifest_sha256": digest(run_path/"manifest.json"),
        "integrity_contract": "hashes below bind transformed export bytes; original run hashes are source references",
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in sorted(members.items())}}
    members["EXPORT_MANIFEST.json"] = json.dumps(spec, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            archive.writestr(name, body)
    write_json(output.with_suffix(".private.json"), {"archive_sha256": digest(output),
        "source_run": str(run_path), "source_manifest": json.loads((run_path/"manifest.json").read_text(encoding='utf-8')),
        "transformations": transformations}, exclusive=True)
    return {"path": str(output), "sha256": digest(output), "research_eligible": False}


def validate_export(path, resolver=None,reanalyze=False):
    with zipfile.ZipFile(path) as archive:
        name = "EXPORT_MANIFEST.json" if "EXPORT_MANIFEST.json" in archive.namelist() else "ARCHIVE_CONTENTS.json"
        spec = json.loads(archive.read(name))
        files = spec["files"] if name == "EXPORT_MANIFEST.json" else spec
        for member, sha in files.items():
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise ValueError("unsafe export member")
            if hashlib.sha256(archive.read(member)).hexdigest() != sha:
                raise ValueError(f"corrupt exported artifact: {member}")
        for reference in spec.get("external_artifacts", []):
            if resolver is None:
                raise ValueError("required external artifacts: supply a portable content-addressed resolver directory")
            resolved = Path(resolver)/reference["sha256"]
            if not resolved.is_file() or digest(resolved) != reference["sha256"]:
                raise ValueError(f"missing/corrupt external artifact {reference['sha256']}")
        analysis=None
        if reanalyze:
            from ncm_lab.evaluation.tables import analyze_records
            rows=[]
            for name in files:
                if name.startswith('evidence/') and name.endswith('predictions.jsonl'):
                    rows.extend(json.loads(line) for line in archive.read(name).decode('utf-8').splitlines() if line.strip())
            analysis=analyze_records(rows)
    return {"valid": True, "files": len(files), "research_eligible": spec.get("research_eligible", False),
            'analysis':analysis,'archive_code_executed':False}
