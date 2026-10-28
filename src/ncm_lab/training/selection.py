from pathlib import Path
from ncm_lab.runtime.io import read, digest, canonical_hash
from ncm_lab.training.records import load_records


def validate_selection(path, development=None):
    path = Path(path)
    selection = read(path)
    weights = (path.parent/selection["weights"]).resolve()
    if not weights.is_relative_to(path.parent.resolve()) or digest(weights) != selection["weights_sha256"]:
        raise ValueError("selected checkpoint replaced or missing")
    if not selection.get("candidates") or any(r.get("split") != "dev" for r in selection["candidates"]):
        raise ValueError("selection requires actual development observations")
    if development is not None:
        records = load_records(development,"dev")
        expected = selection.get("development_hash", selection.get("dev_hash"))
        if canonical_hash(records) != expected:
            raise ValueError("development selection artifact changed")
    return selection
