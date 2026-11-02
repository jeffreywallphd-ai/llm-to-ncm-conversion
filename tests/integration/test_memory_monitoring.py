"""Structured memory metadata must not be projected as numeric training metrics."""

import copy
import json
from pathlib import Path
from ncm_lab.runtime.runner import execute


def test_structured_memory_observations_keep_numeric_tags_compatible(native_artifacts, monkeypatch):
    import ncm_lab.runtime.memory as memory

    snapshot = {
        "host_total_bytes": 1000,
        "host_free_bytes": None,
        "host_available_bytes": 0,
        "host_effective_available_bytes": 0,
        "host_available_source": "fixture.MemAvailable",
        "host_cache_observations": {"cached_bytes": 700},
        "memory_warnings": ["software fixture"],
        "process_memory_constraint": {"status": "finite", "remaining_headroom_bytes": 0},
        "process_rss_bytes": 7,
        "process_lifetime_peak_rss_bytes": 9,
        "scope": "software fixture",
    }
    monkeypatch.setattr(memory, "observe", lambda: snapshot)
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    cfg["training"]["steps"] = 1
    result = execute(cfg)
    assert result["status"] == "completed", result
    assert result["memory"] == snapshot
    rows = [
        json.loads(line) for line in (Path(result["path"]) / "monitor_metrics.jsonl").read_text().splitlines()
    ]
    row = next(r for r in rows if "resources/host_available_bytes" in r.get("metrics", {}))
    assert row["metrics"]["resources/host_available_bytes"]["value"] == 0
    assert row["metrics"]["resources/host_free_bytes"]["value"] is None
    assert row["metrics"]["resources/process_lifetime_peak_rss_bytes"]["value"] == 9
    for field in (
        "host_available_source",
        "process_memory_constraint",
        "host_cache_observations",
        "memory_warnings",
        "scope",
    ):
        assert "resources/" + field not in row["metrics"]
