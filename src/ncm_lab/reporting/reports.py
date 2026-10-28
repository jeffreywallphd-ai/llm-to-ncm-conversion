import json
from pathlib import Path
from ncm_lab.experiments.registry import registry
from ncm_lab.runtime.io import root, read, write_json, digest
from ncm_lab.runtime.provenance import validate_run


def run_paths(include_engineering=False):
    paths = set((root() / "experiments").glob("*/*/runs/*/manifest.json"))
    entries = list((root()/"results/generated/run_locations").glob("*.json"))
    if include_engineering:
        entries += list((root()/"results/generated/engineering_run_locations").glob("*.json"))
    for entry in entries:
        location = read(entry)
        path = Path(location["path"])/"manifest.json"
        if path.parent.name != location["run_id"]:
            raise ValueError("invalid run location registry")
        if path.is_file():
            manifest=read(path)
            external_fixture=(manifest.get("result_origin")=="test_fixture"
                and not path.is_relative_to((root()/"experiments").resolve()))
            if external_fixture and not include_engineering:
                continue
            paths.add(path)
    return sorted(paths)


def find_run(run_id):
    supplied = Path(run_id)
    if supplied.is_dir():
        return supplied
    matches = [p.parent for p in run_paths(include_engineering=True) if p.parent.name == run_id]
    if len(matches) != 1:
        raise ValueError(f"run ID must resolve uniquely: {run_id}")
    return matches[0]


def selected_runs(study, require_current=False):
    manifest = read(root() / "publications" / study / "study.yaml")
    selected = []
    allowed = set(manifest["required_experiments"]) | set(manifest["shared_controls"])
    for entry in manifest["selected_runs"]:
        run_path = find_run(entry["run_id"])
        if digest(run_path / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("study cites a missing or changed run manifest")
        run = validate_run(run_path, require_current_code=require_current)
        if entry.get("attestation"):
            from ncm_lab.reporting.evidence import validate_selection
            validate_selection(entry, run_path)
        if (run["experiment_id"] not in allowed or run["status"] != "completed"
                or run["result_origin"] != "measured" or not (entry.get("attestation") or run["publication_eligible"])):
            raise ValueError(f"ineligible study evidence: {run['run_id']}")
        selected.append((run_path, run))
    return manifest, selected


def analyze(experiment_id):
    if experiment_id not in registry():
        raise ValueError("unregistered experiment")
    records = []
    for path in run_paths():
        run = validate_run(path.parent)
        if run["experiment_id"] == experiment_id:
            records.append({"run_id": run["run_id"], "status": run["status"], "origin": run["result_origin"],
                            "publication_eligible": run["publication_eligible"],
                            "manifest_sha256": digest(path), "metrics": read(path.parent / "metrics.json")})
    output = root() / "results/generated" / f"{experiment_id}.json"
    from ncm_lab.evaluation.tables import analyze_records
    observations = []
    for path in run_paths():
        if read(path)["experiment_id"] == experiment_id and (path.parent/"predictions.jsonl").exists():
            origin=read(path)['result_origin']
            observations.extend({**json.loads(line),'result_origin':origin} for line in (path.parent/"predictions.jsonl").read_text().splitlines())
    analysis = analyze_records(observations)
    from ncm_lab.evaluation.codec_analysis import analyze_codec_runs
    codec = analyze_codec_runs([p.parent for p in run_paths() if read(p)["experiment_id"] == experiment_id],
                               root()/"results/generated"/(experiment_id+"-tables"))
    findings=[]
    for study in ('acl','is_journal'):
        _,selected=selected_runs(study)
        for _,manifest in selected:
            if manifest['experiment_id']==experiment_id:
                findings.append({'run_id':manifest['run_id'],'study':study,'evidence':'selected measured observation',
                    'hypothesis_assessment':'unassessed; inspect declared paired contrasts and uncertainty'})
    write_json(output, {"experiment_id": experiment_id, "runs": records, "findings": findings, "analysis": analysis,
                        "codec_analysis":codec})
    from ncm_lab.runtime.provenance import implementation_hash
    write_json(output.with_name(experiment_id+'.analysis_manifest.json'),{'schema_version':1,'analysis':'experiment_tables_v2',
        'implementation_sha256':implementation_hash(),'inputs':[{'run_id':r['run_id'],'manifest_sha256':r['manifest_sha256']} for r in records],
        'files':{output.name:digest(output)},'parameters':{'group_bootstrap_iterations':500,'weighting':'requests within seed/condition'},
        'derived_tables_manifest':{'path':codec['analysis_manifest'],'sha256':digest(codec['analysis_manifest'])}})
    known=registry()
    index = [{"experiment_id": p.stem, "summary": p.relative_to(root()).as_posix()} for p in output.parent.glob("*.json") if p.stem in known]
    write_json(root() / "results/index.json", {"schema_version": 1, "experiments": sorted(index, key=lambda e: e["experiment_id"])})
    return {"path": str(output), "runs": len(records), "research_findings": findings}


def build(study):
    manifest, selected = selected_runs(study)
    lines = [f"# {study} experiment report", "", "This report is not a manuscript.", ""]
    lines += ["No empirical findings yet.", ""] if not selected else ["Selected evidence:", ""]
    for path, run in selected:
        metrics = read(path / "metrics.json")
        lines += [f"- {run['experiment_id']} / {run['run_id']}: {json.dumps(metrics, sort_keys=True)}"]
    lines += ["", "Required experiments:", ""]
    for eid in manifest["required_experiments"]:
        p = registry()[eid]
        lines.append(f"- {eid}: {p['question']} — no selected eligible run" if not selected else f"- {eid}: see selection manifest")
    lines += ["", "Missing evidence:", "", *[f"- {s}" for s in manifest["missing_evidence"]], ""]
    dest = root() / "results/generated" / f"{study}_report.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    refresh_readme_findings()
    return {"path": str(dest), "selected_eligible_runs": len(selected), "manuscript": False}


def refresh_readme_findings():
    path = root() / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- BEGIN MEASURED RESULTS -->", "<!-- END MEASURED RESULTS -->"
    if start not in text or end not in text:
        return
    entries = []
    for study in ("acl", "is_journal"):
        if not (root()/"publications"/study/"study.yaml").exists():
            continue
        _, selected = selected_runs(study)
        for run_path, run in selected:
            metrics = read(run_path/"metrics.json")
            entries.append(f"- {study}: {run['experiment_id']} / {run['run_id']}; "
                           f"hardware {run['runtime']['hardware_profile']}; "
                           f"validated metrics/denominators/uncertainty: {json.dumps(metrics,sort_keys=True)}")
    body = "\n\n" + ("\n".join(entries) if entries else "No empirical findings yet.") + "\n"
    prefix, rest = text.split(start,1)
    _, suffix = rest.split(end,1)
    updated = prefix + start + body + end + suffix
    if updated != text:
        path.write_text(updated,encoding="utf-8")
