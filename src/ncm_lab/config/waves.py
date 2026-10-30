"""Offline typed wave resolution; no acquisition, model allocation, or ledger writes."""

from dataclasses import asdict, dataclass
import math
from ncm_lab.config.stages import config
from ncm_lab.runtime.io import root, read, digest

STAGES = ("prepare_sources", "data", "prepare_model", "preflight", "teacher", "codec", "targets", "generator")


@dataclass(frozen=True)
class WaveLimits:
    max_wave_wall_seconds: int
    max_controller_seconds: int
    max_phase_controller_seconds: int
    max_log_bytes: int
    reserve_free_bytes: int
    max_phase_attempts: int = 32

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) is not int or value <= 0 or not math.isfinite(value):
                raise ValueError("finite positive integer wave limits required")
        if self.max_phase_controller_seconds < self.max_controller_seconds:
            raise ValueError("aggregate controller limit below invocation allowance")


def wave_config(path):
    value = read(root() / path)
    if (
        not {"schema", "phase_id", "run_root", "limits", "stages", "notes"} <= value.keys()
        or set(value) - {"schema", "phase_id", "run_root", "limits", "stages", "notes", "qualified_data"}
        or value["schema"] != "native_wave_v1"
    ):
        raise ValueError("typed native_wave_v1 fields required")
    if not value["phase_id"] or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value["phase_id"]
    ):
        raise ValueError("safe intentional wave phase ID required")
    value["limits"] = asdict(WaveLimits(**value["limits"]))
    if set(value["stages"]) != set(STAGES) - {"preflight"}:
        raise ValueError("complete sequential wave stage configuration required")
    resolved = {name: config(path) for name, path in value["stages"].items()}
    caps = None
    for name, stage in resolved.items():
        expected = "prepare" if name.startswith("prepare_") else name
        if stage["stage"] != expected:
            raise ValueError("wave child CLI/config stage mismatch")
        if not name.startswith("prepare_"):
            if stage["budget"]["phase_id"] != value["phase_id"]:
                raise ValueError("native wave attempts must share the deliberate phase identity")
            child_caps = {k: v for k, v in stage["budget"].items() if k.startswith("max_phase_")}
            if caps is not None and caps != child_caps:
                raise ValueError("native aggregate caps must agree across all wave stages")
            caps = child_caps
    if len({str(s["model"]) for n, s in resolved.items() if not n.startswith("prepare_")}) != 1:
        raise ValueError("wave children must use the same pinned model/tokenizer")
    dependencies = {}
    for stage in resolved.values():
        for section, field in [
            ("data", "asset_lock"),
            ("data", "source_policy"),
            ("preparation", "asset_lock"),
        ]:
            if stage.get(section, {}).get(field):
                path = stage[section][field]
                dependencies[path] = digest(root() / path)
    return {
        **value,
        "run_root": str((root() / value["run_root"]).resolve()),
        "stages": resolved,
        "dependency_sha256": dependencies,
    }
