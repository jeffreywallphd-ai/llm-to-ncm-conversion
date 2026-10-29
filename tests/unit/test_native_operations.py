from dataclasses import asdict
import json
import uuid
import pytest
from ncm_lab.config.stages import config, Training
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.runtime.budget import Budget
from ncm_lab.runtime.io import root, digest, read
from ncm_lab.runtime.runner import execute
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.data.artifacts import Dataset


def test_all_native_configs_validate_and_large_budgets_are_explicit():
    for path in (root() / "configs").rglob("*.yaml"):
        value = read(path)
        if value.get("schema") == "native_stage_v1":
            resolved = config(path)
            assert resolved["budget"]["max_wall_seconds"] > 0
    codec = config("configs/codec/initial_10m_answer_tokens.yaml")
    generator = config("configs/generator/initial_100m_processed_tokens.yaml")
    assert (
        codec["training"]["minimum_unique_answer_positions"] == 10000000 and codec["training"]["epochs"] == 3
    )
    assert (
        generator["training"]["max_processed_tokens"] == 100000000
        and generator["training"]["warmup_processed_tokens"] < 100000000
    )


def test_preparation_plan_is_download_and_training_free(monkeypatch):
    import ncm_lab.runtime.prepare as prepare

    def forbidden(*args, **kwargs):
        raise AssertionError("planning retrieved a model")

    monkeypatch.setattr(prepare, "download_asset", forbidden)
    result = prepare.prepare(config("configs/validation/real_pilot.yaml"), True)
    assert not result["model_downloads_started"] and not result["training_started"]
    assert (
        result["assets"] == 17
        and result["cache_estimates_decimal_bytes"]["10m_teacher_states_bf16"] == 81920000000
    )
    assert result["cache_estimates_decimal_bytes"]["10m_fine_codes_bf16"] == 7680000000


def test_bounded_data_preparation_uses_identical_template_and_whole_groups(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "original_id": str(i),
                    "group_id": "tree" + str(i // 2),
                    "prompt": "token4 " + str(i),
                    "answer": "token8 token12",
                    "split": "train",
                }
            )
            + "\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )
    value = {
        "schema": "native_stage_v1",
        "stage": "data",
        "model": model_spec(),
        "architecture": tiny_architecture().to_dict(),
        "budget": asdict(Budget("native-data-" + uuid.uuid4().hex)),
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "run_root": str(tmp_path / "runs"),
        "training": asdict(Training()),
        "inputs": {},
        "data": {
            "source_path": str(path),
            "source_sha256": digest(path),
            "revision": "fixture",
            "license": "software fixture",
            "format": "jsonl",
            "max_source_rows": 40,
            "max_examples": 20,
            "max_unique_tokens": 10000,
            "max_prompt_tokens": 128,
            "dev_fraction": 0.5,
            "overlength": "exclude",
        },
    }
    result = execute(value)
    assert result["status"] == "completed", result
    dataset = Dataset(result["artifact"])
    assert not set(dataset.manifest["groups"]["train"]) & set(dataset.manifest["groups"]["dev"])
    assert result["unique_token_shortfall"] > 0
    assert all(
        dataset[i][0].answer_ids[-1] == 2 and dataset[i][0].source_type == "software_fixture"
        for i in range(len(dataset))
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_positions", 33),
        ("head_chunk", 9),
        ("batch_size", True),
        ("steps", float("inf")),
        ("gradient_clip", 0),
    ],
)
def test_invalid_native_training_options_rejected(field, value):
    with pytest.raises(ValueError):
        Training(**{field: value})
