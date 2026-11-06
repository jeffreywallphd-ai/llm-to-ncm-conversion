"""Persistent conservative logical-write charges for SQLite, JSON and indexes."""

from pathlib import Path
from ncm_lab.runtime.io import read, atomic_json
from ncm_lab.runtime.budget import BudgetStop


def charge(run, size):
    settings = run.value.get("stage3")
    if not settings:
        return
    context = run.value["stage1_context"]
    folder = Path(context["experiment"]).parent
    # One experiment writer lock covers all these updates, including child workers.
    path = folder / "ledgers" / (context["stage"] + "-payload-writes.json")
    caps = settings["payload_write_caps"]
    state = (
        read(path)
        if path.exists()
        else dict(schema="stage3_payload_writes_v1", caps=caps, charged_bytes=0, attempts={})
    )
    if state["caps"] != caps:
        raise ValueError("immutable Stage 3 payload write caps changed")
    own = state["attempts"].get(run.id, 0)
    if own + size > caps["per_run"] or state["charged_bytes"] + size > caps["aggregate"]:
        raise BudgetStop("Stage 3 cumulative payload/SQLite write allowance exhausted")
    state["charged_bytes"] += size
    state["attempts"][run.id] = own + size
    state["scope"] = (
        "logical bytes plus conservative SQLite page/journal allowance; charged before writes, not physical device wear"
    )
    atomic_json(path, state)


def check_cache(run, path):
    cap = run.value["stage3"]["cache_retained_bytes"]
    if sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) > cap:
        raise BudgetStop("Stage 3 durable cache retained-byte cap exhausted")
