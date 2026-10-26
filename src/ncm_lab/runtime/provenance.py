from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
from ncm_lab.runtime.io import root, digest, read, canonical_hash, write_json


def git(*args):
    proc = subprocess.run(["git", *args], cwd=root(), capture_output=True, timeout=20, check=True)
    return proc.stdout


def source_tree_hash():
    files = []
    for folder in ("src", "configs", "tests", "third_party"):
        for p in sorted((root() / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pt"}:
                files.append((p.relative_to(root()).as_posix(), digest(p)))
    return canonical_hash(files)


def implementation_hash():
    return canonical_hash([(p.relative_to(root()).as_posix(),digest(p))
        for p in sorted((root()/"src").rglob("*")) if p.is_file() and "__pycache__" not in p.parts])


def reserve_phase(budget, run_id):
    """Exclusive local lock and append-only reservations, including failed runs.

    Reservations are conservative: recovery gets a new reservation and never
    silently refunds unrecorded work. A human may define a new phase explicitly.
    """
    base = root() / "results/generated/phases"
    base.mkdir(parents=True, exist_ok=True)
    path, lock = base / f"{budget.phase_budget_id}.jsonl", base / f"{budget.phase_budget_id}.lock"
    try:
        with lock.open("x"):
            pass
    except FileExistsError as exc:
        raise ValueError(f"phase is locked: {lock}; inspect interrupted reservation before recovery") from exc
    try:
        entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        caps = {k:v for k,v in asdict(budget).items() if k.startswith("max_phase_")}
        if any(any(e.get(k) != v for k,v in caps.items()) for e in entries):
            raise ValueError("aggregate phase caps cannot change under an existing phase ID")
        if (sum(e["steps"] for e in entries) + budget.max_steps > budget.max_phase_steps
                or sum(e["tokens"] for e in entries) + budget.max_training_source_token_passes > budget.max_phase_source_token_passes):
            raise ValueError("aggregate phase budget exhausted; no run launched")
        for key, cap in (("wall_seconds","max_phase_wall_seconds"), ("storage_bytes","max_phase_storage_bytes"),
                         ("download_bytes","max_phase_download_bytes")):
            if sum(e[key] for e in entries) + getattr(budget,"max_"+key) > getattr(budget,cap):
                raise ValueError(f"aggregate phase {key} budget exhausted; no run launched")
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "steps": budget.max_steps,
                                "tokens": budget.max_training_source_token_passes,
                                "wall_seconds": budget.max_wall_seconds, "storage_bytes": budget.max_storage_bytes,
                                "download_bytes": budget.max_download_bytes, **caps}) + "\n")
    finally:
        lock.unlink(missing_ok=True)


class Run:
    def __init__(self, protocol, config, budget, environment, parent_run_ids=()):
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        reserve_phase(budget, self.id)
        artifact_root = Path(config.get("artifact_root") or root() / "experiments").resolve()
        track, directory = Path(protocol["directory"]).parts[-2:]
        self.path = artifact_root / track / directory / "runs" / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        write_json(self.path / "config.json", config, exclusive=True)
        write_json(self.path / "protocol.json", protocol, exclusive=True)
        write_json(self.path / "environment.json", environment, exclusive=True)
        self.base = {"schema_version": 1, "run_id": self.id, "experiment_id": protocol["id"],
                     "run_kind": config["executor"], "result_origin": config.get("result_origin", "measured"),
                     "study_membership": [], "parent_run_ids": list(parent_run_ids),
                     "code": {"git_commit": git("rev-parse", "HEAD").decode().strip(),
                              "dirty_diff_hash": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
                              "source_tree_hash": source_tree_hash()},
                     "protocol_hash": digest(self.path / "protocol.json"),
                     "config_hash": digest(self.path / "config.json"),
                     "data_manifest_hashes": {p.name: digest(p) for p in (root() / "data/manifests").glob("*.yaml")},
                     "target": config.get("model", {}), "drafter": {"method_id": config.get("method", "fixture")},
                     "upstream": {"lock_hash": digest(root() / "third_party/sources.lock.yaml")},
                     "seeds": {"training": config.get("seed", 0), "data_order": config.get("seed", 0),
                               "inference": config.get("seed", 0), "timing_order": config.get("seed", 0)},
                     "runtime": {"hardware_profile": environment, "precision": config.get("precision", "float32"),
                                 "attention_backend": "sdpa", "compile_settings": {},
                                 "software_lock_hash": digest(root() / "uv.lock") if (root() / "uv.lock").exists() else None},
                     "budget": asdict(budget)}
        self.base["code"]["implementation_hash"] = implementation_hash()
        write_json(self.path / "launch.json", self.base, exclusive=True)
        self.event("running")

    def event(self, status, **details):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "status": status, **details}) + "\n")

    def close(self, status, metrics, error=None):
        write_json(self.path / "metrics.json", metrics, exclusive=True)
        self.event(status, error=error)
        hashes = {p.relative_to(self.path).as_posix(): digest(p) for p in self.path.rglob("*") if p.is_file()}
        final = {**self.base, "status": status, "error": error, "artifact_hashes": hashes,
                 "publication_eligible": False,
                 "eligibility_reason": "requires study selection, correctness gates, and completed measured research protocol"}
        write_json(self.path / "manifest.json", final, exclusive=True)
        return final


def validate_run(path, require_current_code=False):
    path = Path(path)
    manifest = read(path / "manifest.json")
    if manifest["run_id"] != path.name or manifest["schema_version"] != 1:
        raise ValueError("invalid run identity/schema")
    for name, expected in manifest["artifact_hashes"].items():
        p = (path / name).resolve()
        if not p.is_relative_to(path.resolve()) or not p.is_file() or digest(p) != expected:
            raise ValueError(f"missing/changed run artifact: {name}")
    if digest(path / "config.json") != manifest["config_hash"] or digest(path / "protocol.json") != manifest["protocol_hash"]:
        raise ValueError("resolved configuration/protocol changed")
    if manifest["result_origin"] == "test_fixture" and manifest["publication_eligible"]:
        raise ValueError("fixture runs cannot become publication evidence")
    if require_current_code:
        recorded = manifest["code"].get("implementation_hash")
        if (recorded != implementation_hash() if recorded else manifest["code"]["source_tree_hash"] != source_tree_hash()):
            raise ValueError("selected run implementation differs from current code; preserve original study snapshot")
    return manifest
