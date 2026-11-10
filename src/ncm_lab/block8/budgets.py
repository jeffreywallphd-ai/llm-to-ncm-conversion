"""Finite raw-source work allowances, separate from admitted training positions."""

from dataclasses import asdict
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.io import read


def positive_bound(value, name, maximum=None):
    if type(value) is not int or value <= 0 or maximum is not None and value > maximum:
        raise ValueError("invalid finite source-work bound: " + name)
    return value


def work_allowance(settings, requirements):
    budget = dict(settings["budget"])
    attempts = settings["resources"]["max_attempts_per_stage"]
    for counter, required in requirements.items():
        budget["max_" + counter] = max(budget["max_" + counter], required)
        budget["max_phase_" + counter] = max(
            budget["max_phase_" + counter], budget["max_" + counter] * attempts
        )
    return asdict(Budget(**budget))


def source_work_budget(settings):
    """Resolve a conservative tokenization allowance before an experiment is sealed.

    Raw answers are charged before length/EOS/overlap rejection. The per-row
    allowance uses character limits, not the admitted answer canvas. Sixteen
    positions per character leave room for Unicode normalization and UTF-8 byte
    tokenization; message framing and an EOS/boilerplate reserve are separate.
    The same generous allowance covers each bounded overlap-reference row. This
    is a work allowance, not a claim about corpus token counts or runtime fit.
    """
    data = settings["source"]["resolved_data"]
    rows = positive_bound(data["max_source_rows"], "max_source_rows", 1200000)
    chars = positive_bound(data["max_row_chars"], "max_row_chars", 131072)
    messages = positive_bound(data["max_messages"], "max_messages", 300)
    overlap = data.get("overlap", {})
    overlap_rows = 0
    if overlap.get("source_asset_id"):
        overlap_rows += positive_bound(overlap["max_source_rows"], "overlap source rows", 1000000)
    if overlap.get("artifact"):
        ref = settings["source"]["policy_receipts"]["overlap_artifact"]
        overlap_rows += positive_bound(read(ref["manifest"])["records"], "overlap artifact rows", 1000000)
    total_rows = rows + overlap_rows
    per_row = 16 * chars + 128 * messages + 1024
    attempts = settings["resources"]["max_attempts_per_stage"]
    budget = work_allowance(
        settings, dict(steps=total_rows, source_tokens=total_rows * per_row, answer_positions=total_rows * per_row)
    )
    return budget, dict(
        schema="block8_source_work_budget_v1",
        source_row_allowance=rows,
        overlap_reference_row_allowance=overlap_rows,
        total_row_allowance=total_rows,
        position_allowance_per_row=per_row,
        attempts_per_stage=attempts,
        scope="all raw tokenized responses, including later exclusions; not admitted training exposures",
        wall_storage_canvas_and_write_caps="unchanged",
    )


def preparation_work_budget(settings):
    """Token arrays in a bounded serialized parent cannot outnumber its bytes.

    Candidates can contain long answers: length rejection happens after inspection
    and charging. Use the raw parent-byte and parent-count allowances, while keeping
    the independent step ceiling (including intentionally reduced diagnostic caps).
    """
    policy = settings["data"]
    parents = positive_bound(policy["max_parents"], "max_parents", 1000000)
    per_parent = positive_bound(policy["max_row_bytes"], "max_row_bytes")
    budget = work_allowance(
        settings, dict(source_tokens=parents * per_parent, answer_positions=parents * per_parent)
    )
    return budget, dict(
        schema="block8_preparation_work_budget_v1",
        parent_allowance=parents,
        position_allowance_per_parent=per_parent,
        scope="raw serialized parent inspection before length/EOS exclusions; not retained dataset positions",
        step_wall_storage_canvas_and_write_caps="unchanged",
    )
