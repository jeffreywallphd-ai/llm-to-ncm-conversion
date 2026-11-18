from copy import deepcopy

import pytest

from ncm_lab.block8 import runtime
from ncm_lab.block8.workflow import run_workflow
from ncm_lab.runtime.io import canonical_hash, digest, implementation_hash, read, write_json


def test_resume_accepts_original_block8_config_and_records_current_executor(block8_fixture):
    f = block8_fixture
    settings = deepcopy(f["settings"])
    settings.pop("identity")
    settings["implementation_sha256"] = runtime.PREDECESSOR_IMPLEMENTATION
    settings["identity"] = canonical_hash(settings)
    prepared = run_workflow(
        settings, execute=True, prepare_only=True, experiment_dir=f["folder"] / "original settings",
        data_manifest=f["data"], evaluation_manifest=f["evaluation"],
    )
    assert prepared["status"] == "completed", prepared
    state = read(prepared["experiment"])
    before = digest(runtime.checked(state["settings"]))

    resumed = run_workflow(resume=prepared["experiment"], execute=True)
    assert resumed["status"] == "completed", resumed
    state = read(prepared["experiment"])
    receipt = read(runtime.checked(state["stages"]["generator"]["receipt"]))
    assert receipt["implementation_sha256"] == implementation_hash()
    assert receipt["producer_implementation_sha256"] == runtime.PREDECESSOR_IMPLEMENTATION
    assert digest(runtime.checked(state["settings"])) == before
    resolved = read(runtime.checked(state["stages"]["generator"]["receipt"]).parent / "resolved_config.json")
    assert resolved["executor_sha256"] == implementation_hash()
    assert resolved["training"] == settings["training"]


def test_compatibility_rejects_unknown_producer_and_unrelated_code_changes(block8_fixture, monkeypatch):
    settings = deepcopy(block8_fixture["settings"])
    settings["implementation_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="implementation changed"):
        runtime.compatible_implementation(settings)
    settings["implementation_sha256"] = runtime.PREDECESSOR_IMPLEMENTATION
    original = runtime.digest
    monkeypatch.setattr(runtime, "digest", lambda path: "f" * 64 if path.name == "workflow.py" else original(path))
    with pytest.raises(ValueError, match="implementation changed"):
        runtime.compatible_implementation(settings)


def test_resume_still_rejects_edited_sealed_configuration(block8_fixture):
    f = block8_fixture
    result = run_workflow(
        f["settings"], execute=True, prepare_only=True, experiment_dir=f["folder"] / "tampered",
        data_manifest=f["data"], evaluation_manifest=f["evaluation"],
    )
    state = read(result["experiment"])
    path = runtime.checked(state["settings"])
    settings = read(path)
    settings["training"]["steps"] += 1
    write_json(path, settings)
    with pytest.raises(ValueError, match="changed block8 dependency receipt"):
        runtime.verify_settings(state)


def test_compatibility_rejects_an_unregistered_successor(block8_fixture, monkeypatch):
    settings = deepcopy(block8_fixture["settings"])
    settings["implementation_sha256"] = runtime.PREDECESSOR_IMPLEMENTATION
    original = runtime.read

    def stale_transition(path):
        value = original(path)
        if path.name == "block8_resume_paths_v1.json":
            value["successor_implementation_sha256"] = "f" * 64
        return value

    monkeypatch.setattr(runtime, "read", stale_transition)
    with pytest.raises(ValueError, match="registered resume compatibility"):
        runtime.compatible_implementation(settings)
