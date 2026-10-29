from dataclasses import asdict
import uuid
import pytest
import torch
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.config.stages import Training
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.io import atomic_json, reference
from ncm_lab.runtime.runner import execute
from ncm_lab.validation.fixtures import fixture_data, model_spec


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def native_artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("native-production")
    data = fixture_data(folder / "data", train=4, dev=2)
    refs = folder / "refs"
    refs.mkdir()
    atomic_json(refs / "data.json", reference(data))
    common = {
        "schema": "native_stage_v1",
        "model": model_spec(),
        "architecture": tiny_architecture().to_dict(),
        "budget": asdict(
            Budget("test-" + uuid.uuid4().hex, max_wall_seconds=90, max_storage_bytes=200000000)
        ),
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "run_root": str(folder / "runs"),
        "training": asdict(
            Training(
                steps=2,
                learning_rate=0.001,
                new_learning_rate=0.002,
                head_positions=8,
                head_chunk=4,
                max_validation_examples=2,
            )
        ),
    }
    results = {}
    configs = {}
    for stage, inputs in [
        ("teacher", {"data": refs / "data.json"}),
        ("codec", {"features": refs / "teacher.json"}),
        (
            "targets",
            {"data": refs / "data.json", "features": refs / "teacher.json", "codec": refs / "codec.json"},
        ),
        ("generator", {"targets": refs / "targets.json"}),
    ]:
        value = {
            **common,
            "stage": stage,
            "inputs": {k: str(p) for k, p in inputs.items()},
            "output_ref": str(refs / (stage + ".json")),
        }
        result = execute(value)
        assert result["status"] == "completed", result
        results[stage] = result
        configs[stage] = value
    return {
        "folder": folder,
        "common": common,
        "results": results,
        "configs": configs,
        "data": data,
        "refs": refs,
    }
