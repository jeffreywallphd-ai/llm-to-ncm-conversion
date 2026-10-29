import copy
import json
from pathlib import Path
from ncm_lab.runtime.runner import execute
from ncm_lab.runtime.io import digest
from ncm_lab.training.checkpoints import CheckpointReader


def test_earlier_checkpoint_resume_retains_true_attempt_history(native_artifacts):
    original = Path(native_artifacts["results"]["generator"]["path"])
    old_log = original / "monitor_metrics.jsonl"
    before = digest(old_log)
    checkpoint = None
    for manifest in (original / "checkpoints").glob("version-*/manifest.json"):
        reader = CheckpointReader(manifest)
        if reader.materialize(reader.field("training"))["optimizer_steps"] == 1:
            checkpoint = manifest
            break
    assert checkpoint
    cfg = copy.deepcopy(native_artifacts["configs"]["generator"])
    cfg["output_ref"] = None
    result = execute(cfg, checkpoint)
    assert result["status"] == "completed", result
    new = Path(result["path"])
    lineage = json.loads((new / "attempt_lineage.json").read_text())
    old_meta = json.loads((original / "attempt.json").read_text())
    new_meta = json.loads((new / "attempt.json").read_text())
    assert lineage["parent_attempt"] == old_meta["attempt_id"]
    assert lineage["restored_step"] == 1
    assert old_meta["attempt_id"] != new_meta["attempt_id"]
    assert digest(old_log) == before
    rows = [json.loads(line) for line in (new / "monitor_metrics.jsonl").read_text().splitlines()]
    assert {r["step"] for r in rows} == {2}
