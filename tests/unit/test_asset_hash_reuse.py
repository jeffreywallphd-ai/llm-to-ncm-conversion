"""Independent I/O optimization: verification remains mandatory on every invocation."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import pytest
from ncm_lab.teacher.model import verify_cached_assets


@pytest.fixture
def locked_assets(tmp_path, monkeypatch):
    import ncm_lab.runtime.io as io
    import ncm_lab.teacher.model as models

    folder = tmp_path / "assets"
    folder.mkdir()
    assets = []
    hashes = {}
    for number in range(3):
        body = ("tiny asset " + str(number)).encode()
        name = f"asset-{number}.bin"
        (folder / name).write_bytes(body)
        sha = hashlib.sha256(body).hexdigest()
        blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body, usedforsecurity=False).hexdigest()
        assets.append(
            {
                "role": "target",
                "filename": name,
                "size_bytes": len(body),
                "sha256": sha if number != 1 else None,
                "git_blob_sha1": blob if number != 0 else None,
            }
        )
        hashes[name] = sha
    lock = tmp_path / "configs/data/assets.lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"assets": assets}))
    monkeypatch.setattr(io, "root", lambda: tmp_path)
    monkeypatch.setattr(models, "cached_path", lambda spec: folder)
    monkeypatch.setattr(
        io, "digest", lambda *args: pytest.fail("redundant digest pass after asset verification")
    )
    reads = Counter()
    original = Path.open

    def counted(path, *args, **kwargs):
        if path.parent == folder and args and args[0] == "rb":
            reads[path.name] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    return folder, assets, hashes, reads


def test_reuses_returned_sha_with_one_read_per_asset_per_invocation(locked_assets):
    _, _, hashes, reads = locked_assets
    assert verify_cached_assets({}) == hashes
    assert dict(reads) == {name: 1 for name in hashes}
    assert verify_cached_assets({}) == hashes
    assert dict(reads) == {name: 2 for name in hashes}  # No persistent trust cache.


@pytest.mark.parametrize("case", ["sha256", "git_blob", "size", "missing"])
def test_missing_or_changed_asset_still_fails_on_next_invocation(locked_assets, case):
    folder, _, hashes, _ = locked_assets
    assert verify_cached_assets({}) == hashes
    path = folder / ("asset-1.bin" if case == "git_blob" else "asset-0.bin")
    if case == "missing":
        path.unlink()
    elif case == "size":
        path.write_bytes(path.read_bytes() + b"x")
    else:
        path.write_bytes(b"X" + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="missing/corrupt pinned Qwen asset"):
        verify_cached_assets({})
