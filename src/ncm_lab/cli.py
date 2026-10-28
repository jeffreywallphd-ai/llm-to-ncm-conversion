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
    prepare = subs.add_parser(
        "prepare", help="retrieve pinned pilot models/data, verify hashes, and prepare train/dev prompts"
    )
    prepare.add_argument("--profile", choices=["cpu", "spark", "cuda"], default="spark")
    prepare.add_argument(
        "--assets-root", type=Path, help="directory on a disk with space for the selected assets"
    )
    prepare.add_argument(
        "--budget", type=Path, help="finite preparation budget (default: configs/preparation/pilot.yaml)"
    )
    prepare.add_argument(
        "--dry-run", action="store_true", help="offline size/disk plan; do not download or prepare data"
    )
    prepare.add_argument(
        "--include-financial",
        action="store_true",
        help="also retrieve original FinQA/TAT-QA/DocFinQA train and dev releases",
    )
    prepare.add_argument(
        "--with-test-data",
        action="store_true",
        help="also retrieve test releases; test gold is never used to prepare training prompts",
    )
    prepare.add_argument(
        "--include-external-baseline",
        action="store_true",
        help="also retrieve the pinned DCore checkpoint, within the preparation budget",
    )
    data = subs.add_parser("data").add_subparsers(dest="action", required=True)
    for action in ("inspect", "fetch"):
        d = data.add_parser(action)
        d.add_argument("--dataset", required=True)
        if action == "fetch":
            d.add_argument("--split", required=True)
            d.add_argument("--max-bytes", type=int)
    d = data.add_parser("check", help="validate tracked real training-sample provenance and separation")
    migration = data.add_parser(
        "migrate-teacher", help="derive a source contract from verified original immutable teacher evidence"
    )
    migration.add_argument("--path", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    migration.add_argument("--diagnostic-fixture", action="store_true")
    lock = data.add_parser("lock", help="bind a consumed JSONL artifact to its source receipt")
    for name in ("path", "receipt", "source-receipt"):
        lock.add_argument("--" + name, type=Path, required=True)
    for name in ("schema", "split", "family"):
        lock.add_argument("--" + name, required=True)
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
    analysis = results.add_parser("analyze")
    analysis.add_argument("--experiment", required=True)
    analysis.add_argument("--runs", nargs="+")
    analysis.add_argument("--bindings", type=Path)
    analysis.add_argument("--legacy-diagnostic", action="store_true")
    results.add_parser("index").add_argument("--experiment")
    report = subs.add_parser("report").add_subparsers(dest="action", required=True)
    report.add_parser("build").add_argument("--study", required=True, choices=["acl", "is_journal"])
    pub = subs.add_parser("publication").add_subparsers(dest="action", required=True)
    validate = pub.add_parser("validate-export")
    validate.add_argument("--path", type=Path, required=True)
    validate.add_argument("--resolver", type=Path)
    validate.add_argument(
        "--reanalyze",
        action="store_true",
        help="reanalyze raw JSON records with this trusted installation; execute no archive code",
    )
    export = pub.add_parser("export")
    export.add_argument("--study", required=True, choices=["acl", "is_journal"])
    export.add_argument("--anonymize", action="store_true")
    export.add_argument("--output", type=Path, required=True)
    methods = subs.add_parser("method").add_subparsers(dest="action", required=True)
    methods.add_parser("list")
    methods.add_parser("inspect").add_argument("--config", type=Path, required=True)
    cost = methods.add_parser("select-cost")
    cost.add_argument("--runs", nargs="+", required=True)
    cost.add_argument("--output", type=Path, required=True)
    cost.add_argument("--tolerance", type=float, required=True)
    matrix = subs.add_parser("matrix").add_subparsers(dest="action", required=True)
    mp = matrix.add_parser("plan")
    mp.add_argument("--spec", type=Path, required=True)
    mp.add_argument("--output", type=Path, required=True)
    mp.add_argument("--tiers", type=int, nargs="+", help="finite selected subset of declared data tiers")
    matrix.add_parser("run").add_argument("--plan", type=Path, required=True)
    assess = results.add_parser("attest")
    assess.add_argument("--run", required=True)
    assess.add_argument(
        "--claim-class",
        choices=["correctness", "performance", "full_test", "representation", "mechanism"],
        default="correctness",
    )
    assess.add_argument("--analysis", type=Path, help="explicit immutable analysis receipt")
    selection = results.add_parser(
        "select", help="select eligible immutable evidence for explicit study claims"
    )
    selection.add_argument("--run", required=True)
    selection.add_argument("--study", choices=["acl", "is_journal"], required=True)
    selection.add_argument("--attestation", type=Path, required=True)
    selection.add_argument("--claims", nargs="+", required=True)
    codec_analysis = results.add_parser("codec-analysis")
    codec_analysis.add_argument("--runs", nargs="+", required=True)
    codec_analysis.add_argument("--output", type=Path, required=True)
    codec_analysis.add_argument("--figures", action="store_true")
    engineering = pub.add_parser("engineering-export")
    engineering.add_argument("--run", required=True)
    engineering.add_argument("--output", type=Path, required=True)
    subs.add_parser(
        "acceptance", help="bounded complete codec-to-drafter CPU fixture pipeline and portable export"
    )
    checkpoint = subs.add_parser("checkpoint").add_subparsers(dest="action", required=True)
    validate_selection = checkpoint.add_parser("validate-selection")
    validate_selection.add_argument("--path", type=Path, required=True)
    validate_selection.add_argument("--development-data", type=Path)
    codec_migration = checkpoint.add_parser(
        "migrate-codec", help="derive missing source dependencies without rewriting original weights"
    )
    codec_migration.add_argument("--selection", type=Path, required=True)
    codec_migration.add_argument("--teacher-contract", type=Path, required=True)
    codec_migration.add_argument("--output", type=Path, required=True)
    codec_migration.add_argument("--diagnostic-fixture", action="store_true")
    subs.add_parser(
        "smoke", help="two-step CPU fixture integration, checkpoint resume, validation, and reports"
    )
    return p


def dispatch(args):
    if args.command == "checkpoint":
        if args.action == "migrate-codec":
            from ncm_lab.training.codec import migrate_codec

            return migrate_codec(args.selection, args.teacher_contract, args.output, args.diagnostic_fixture)
        from ncm_lab.training.selection import validate_selection

        return {"valid": True, "selection": validate_selection(args.path, args.development_data)}
    if args.command == "method":
        from ncm_lab.models.factory import BASE, LATENT, resolve_method
        from dataclasses import asdict

        if args.action == "select-cost":
            from ncm_lab.experiments.cost_selection import select_cost
            from ncm_lab.reporting.reports import find_run

            return select_cost([find_run(r) for r in args.runs], args.output, args.tolerance)
        return sorted(BASE | LATENT) if args.action == "list" else asdict(resolve_method(read(args.config)))
    if args.command == "matrix":
        from ncm_lab.experiments.matrix import resolve_plan, launch_plan

        if args.action == "run":
            return launch_plan(args.plan)
        plan = resolve_plan(args.spec, args.output, args.tiers)
        return {
            "path": str(args.output),
            "sha256": plan["sha256"],
            "jobs": len(plan["jobs"]),
            "blocked_jobs": sum(bool(j["blocked"]) for j in plan["jobs"]),
            "aggregate_limits": plan["aggregate_limits"],
            "offline_only": True,
        }
    if args.command == "acceptance":
        from ncm_lab.experiments.runner import run
        from ncm_lab.reporting.reports import analyze
        from ncm_lab.reporting.evidence import attest
        from ncm_lab.reporting.portable import engineering_export, validate_export

        result = run(
            "ACL-005", root() / "configs/acceptance.yaml", root() / "configs/budgets/acceptance.yaml"
        )
        if result["status"] != "completed":
            return result
        analysis = analyze("ACL-005", [result["path"]])
        eligibility = attest(result["path"])
        archive = root() / "publications/exports" / (result["run_id"] + "-engineering.zip")
        exported = engineering_export(result["path"], archive)
        return {
            **result,
            "analysis": analysis,
            "eligibility": eligibility,
            "engineering_export": exported,
            "export_validation": validate_export(archive),
        }
    if args.command == "prepare":
        from ncm_lab.runtime.prepare import prepare

        return prepare(
            args.assets_root,
            args.budget,
            args.profile,
            args.dry_run,
            args.include_financial,
            args.with_test_data,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
            include_external_baseline=args.include_external_baseline,
        )
    if args.command == "doctor":
        from ncm_lab.runtime.doctor import doctor

        return doctor(args.profile)
    if args.command == "data":
        from ncm_lab.data.acquire import manifest, fetch

        if args.action == "migrate-teacher":
            from ncm_lab.training.source_contracts import migrate_teacher

            return migrate_teacher(args.path, args.output, args.diagnostic_fixture)
        if args.action == "lock":
            from ncm_lab.data.artifacts import lock_input

            return lock_input(
                args.path, args.receipt, args.schema, args.split, args.family, args.source_receipt
            )
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
            checks.append(
                {"dataset": path.name.split("_")[0], "records": len(rows), "sha256": receipt["sha256"]}
            )
        if not checks:
            raise ValueError("real training samples not present; see docs/data_provenance.md")
        return {"checks": checks, "benchmark_evaluation": False}
    if args.command == "experiment":
        from ncm_lab.experiments.registry import registry, plan

        if args.action == "list":
            return [
                {"id": k, "directory": p["directory"], "study": p["study"]} for k, p in registry().items()
            ]
        if args.action == "plan":
            return plan(args.id, args.profile)
        from ncm_lab.experiments.runner import run

        return run(args.id, args.config, args.budget)
    if args.command == "results":
        from ncm_lab.reporting.reports import analyze, find_run
        from ncm_lab.runtime.provenance import validate_run

        if args.action == "attest":
            from ncm_lab.reporting.evidence import attest

            return attest(find_run(args.run), args.claim_class, args.analysis)
        if args.action == "select":
            from ncm_lab.reporting.evidence import select_evidence

            return select_evidence(args.study, find_run(args.run), args.attestation, args.claims)
        if args.action == "codec-analysis":
            from ncm_lab.evaluation.codec_analysis import analyze_codec_runs

            return analyze_codec_runs([find_run(r) for r in args.runs], args.output, args.figures)
        if args.action == "analyze":
            return analyze(args.experiment, args.runs, args.bindings, args.legacy_diagnostic)
        if args.action == "index":
            from ncm_lab.reporting.reports import run_paths

            return [
                {
                    "run_id": read(p)["run_id"],
                    "experiment_id": read(p)["experiment_id"],
                    "status": read(p)["status"],
                }
                for p in run_paths()
                if not args.experiment or read(p)["experiment_id"] == args.experiment
            ]
        run = validate_run(find_run(args.run))
        return {
            "valid": True,
            "run_id": run["run_id"],
            "status": run["status"],
            "publication_eligible": run["publication_eligible"],
        }
    if args.command == "report":
        from ncm_lab.reporting.reports import build

        return build(args.study)
    if args.command == "publication":
        if args.action == "validate-export":
            from ncm_lab.reporting.portable import validate_export

            return validate_export(args.path, args.resolver, args.reanalyze)
        if args.action == "engineering-export":
            from ncm_lab.reporting.portable import engineering_export
            from ncm_lab.reporting.reports import find_run

            return engineering_export(find_run(args.run), args.output)
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
        analyze("SHR-001", [result["path"]])
        return {
            **result,
            "validated": True,
            "reports": [build(s) for s in ("acl", "is_journal")],
            "research_findings": [],
            "origin": "test_fixture",
        }
    raise ValueError("unknown command")


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        result = dispatch(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return (
            2
            if isinstance(result, dict) and result.get("status") in {"failed", "blocked", "cancelled"}
            else 0
        )
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"ncm: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
