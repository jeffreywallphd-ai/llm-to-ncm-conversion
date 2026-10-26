import argparse
import json
from pathlib import Path
import sys
from ncm_lab.runtime.io import root, read


def parser():
    p = argparse.ArgumentParser(prog="ncm", description="Bounded hierarchical speculative drafting research")
    p.add_argument("--version", action="version", version="ncm-lab 0.1.0")
    subs = p.add_subparsers(dest="command", required=True)
    doctor = subs.add_parser("doctor", help="inspect hardware/software without downloading models")
    doctor.add_argument("--profile", choices=["cpu", "spark", "cuda"], default="cpu")
    prepare = subs.add_parser("prepare", help="retrieve pinned pilot models/data, verify hashes, and prepare train/dev prompts")
    prepare.add_argument("--profile", choices=["cpu","spark","cuda"], default="spark")
    prepare.add_argument("--assets-root", type=Path, help="directory on a disk with space for the selected assets")
    prepare.add_argument("--budget", type=Path, help="finite preparation budget (default: configs/preparation/pilot.yaml)")
    prepare.add_argument("--dry-run", action="store_true", help="offline size/disk plan; do not download or prepare data")
    prepare.add_argument("--include-financial", action="store_true", help="also retrieve original FinQA/TAT-QA/DocFinQA train and dev releases")
    prepare.add_argument("--with-test-data", action="store_true", help="also retrieve test releases; test gold is never used to prepare training prompts")
    data = subs.add_parser("data").add_subparsers(dest="action", required=True)
    for action in ("inspect", "fetch"):
        d = data.add_parser(action)
        d.add_argument("--dataset", required=True)
        if action == "fetch":
            d.add_argument("--split", required=True)
            d.add_argument("--max-bytes", type=int)
    d = data.add_parser("check", help="validate tracked real training-sample provenance and separation")
    experiment = subs.add_parser("experiment").add_subparsers(dest="action", required=True)
    experiment.add_parser("list")
    for action in ("plan", "run"):
        e = experiment.add_parser(action)
        e.add_argument("--id", required=True)
        if action == "plan":
            e.add_argument("--profile", choices=["cpu", "spark", "cuda"], default="cpu")
        else:
            e.add_argument("--config", type=Path, required=True)
            e.add_argument("--budget", type=Path, required=True)
    results = subs.add_parser("results").add_subparsers(dest="action", required=True)
    results.add_parser("validate").add_argument("--run", required=True)
    results.add_parser("analyze").add_argument("--experiment", required=True)
    report = subs.add_parser("report").add_subparsers(dest="action", required=True)
    report.add_parser("build").add_argument("--study", required=True, choices=["acl", "is_journal"])
    pub = subs.add_parser("publication").add_subparsers(dest="action", required=True)
    export = pub.add_parser("export")
    export.add_argument("--study", required=True, choices=["acl", "is_journal"])
    export.add_argument("--anonymize", action="store_true")
    export.add_argument("--output", type=Path, required=True)
    subs.add_parser("smoke", help="two-step CPU fixture integration, checkpoint resume, validation, and reports")
    return p


def dispatch(args):
    if args.command == "prepare":
        from ncm_lab.runtime.prepare import prepare
        return prepare(args.assets_root,args.budget,args.profile,args.dry_run,args.include_financial,
                       args.with_test_data,progress=lambda message: print(message,file=sys.stderr,flush=True))
    if args.command == "doctor":
        from ncm_lab.runtime.doctor import doctor
        return doctor(args.profile)
    if args.command == "data":
        from ncm_lab.data.acquire import manifest, fetch
        if args.action == "inspect":
            return manifest(args.dataset)
        if args.action == "fetch":
            return fetch(args.dataset, args.split, args.max_bytes)
        from ncm_lab.data.loaders import load_records
        from ncm_lab.runtime.io import digest
        checks = []
        for path in (root() / "tests/fixtures").glob("*_train_sample.json"):
            receipt = read(path.with_suffix(".provenance.json"))
            if receipt["sha256"] != digest(path) or receipt["split"] != "train":
                raise ValueError("training sample provenance mismatch")
            rows = list(load_records(path, path.name.split("_")[0], "train"))
            for item, gold in rows:
                if not item.question or not item.prompt() or gold.answer is None:
                    raise ValueError("invalid separated training example")
            checks.append({"dataset": path.name.split("_")[0], "records": len(rows), "sha256": receipt["sha256"]})
        if not checks:
            raise ValueError("real training samples not present; see docs/data_provenance.md")
        return {"checks": checks, "benchmark_evaluation": False}
    if args.command == "experiment":
        from ncm_lab.experiments.registry import registry, plan
        if args.action == "list":
            return [{"id": k, "directory": p["directory"], "study": p["study"]} for k, p in registry().items()]
        if args.action == "plan":
            return plan(args.id, args.profile)
        from ncm_lab.experiments.runner import run
        return run(args.id, args.config, args.budget)
    if args.command == "results":
        from ncm_lab.reporting.reports import analyze, find_run
        from ncm_lab.runtime.provenance import validate_run
        if args.action == "analyze":
            return analyze(args.experiment)
        run = validate_run(find_run(args.run))
        return {"valid": True, "run_id": run["run_id"], "status": run["status"], "publication_eligible": run["publication_eligible"]}
    if args.command == "report":
        from ncm_lab.reporting.reports import build
        return build(args.study)
    if args.command == "publication":
        from ncm_lab.reporting.export import review_export
        return review_export(args.study, args.output, args.anonymize)
    if args.command == "smoke":
        from ncm_lab.experiments.runner import run
        from ncm_lab.runtime.provenance import validate_run
        from ncm_lab.reporting.reports import analyze, build
        result = run("SHR-001", root() / "configs/smoke.yaml", root() / "configs/budgets/cpu_smoke.yaml")
        if result["status"] != "completed":
            return result
        validate_run(result["path"])
        analyze("SHR-001")
        return {**result, "validated": True, "reports": [build(s) for s in ("acl", "is_journal")],
                "research_findings": [], "origin": "test_fixture"}
    raise ValueError("unknown command")


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        result = dispatch(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("status") in {"failed", "blocked", "cancelled"} else 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"ncm: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
