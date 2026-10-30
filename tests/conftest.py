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


@pytest.fixture
def tulu_fixture(tmp_path):
    """Real local Parquet layout; no research corpus, tokenizer, or weight retrieval."""
    import copy
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ncm_lab.runtime.io import root, read, digest

    folder = tmp_path / "tulu-fixture"
    folder.mkdir()
    base = read(root() / "configs/data/tulu3_extension.yaml")
    phase = "tulu-fixture-" + uuid.uuid4().hex
    common = {
        "schema": "native_stage_v1",
        "model": model_spec(),
        "architecture": tiny_architecture().to_dict(),
        "budget": asdict(
            Budget(
                phase,
                max_wall_seconds=90,
                max_steps=10000,
                max_source_tokens=1000000,
                max_answer_positions=1000000,
                max_storage_bytes=200000000,
                max_phase_wall_seconds=1800,
                max_phase_storage_bytes=2000000000,
            )
        ),
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "hardware_profile": "cpu",
        "run_root": str(folder / "runs"),
        "training": asdict(Training(steps=2, head_positions=4, head_chunk=4, max_validation_examples=2)),
        "monitoring": {"enabled": True, "sample_prompts": 0},
    }
    source = "ai2-adapt-dev/no_robots_converted"
    rows = []
    for i in range(48):
        rows.append(
            {
                "id": str(i),
                "source": source,
                "group_id": str(i),
                "messages": [
                    {"role": "system", "content": "token4"},
                    {
                        "role": "user",
                        "content": f"Please explain this small software example in clear English token{4 + i % 28}",
                    },
                    {
                        "role": "assistant",
                        "content": "An earlier assistant response is retained as conversation context.",
                    },
                    {
                        "role": "user",
                        "content": f"Please explain the complete example for this follow up question token{4 + i // 28} token{4 + i % 28}",
                    },
                    {
                        "role": "assistant",
                        "content": "This answer explains the small test clearly and completely.",
                    },
                ],
            }
        )
    duplicate = copy.deepcopy(rows[0])
    duplicate["id"] = "duplicate"
    duplicate["group_id"] = "duplicate"
    alternative = copy.deepcopy(rows[0])
    alternative["id"] = "alternative"
    alternative["group_id"] = "alternative"
    alternative["messages"][-1]["content"] = (
        "This alternative answer explains the test clearly and completely."
    )
    rows += [duplicate, alternative]
    assets = []
    asset_root = folder / "assets"
    asset_root.mkdir()
    for number, partition in enumerate((rows[::2], rows[1::2])):
        path = asset_root / f"train-{number}.parquet"
        pq.write_table(pa.Table.from_pylist(partition), path)
        assets.append(
            {
                "id": f"tulu3:fixture-{number}",
                "role": "tulu3",
                "repo_id": "allenai/tulu-3-sft-mixture",
                "revision": base["data"]["revision"],
                "filename": path.name,
                "split": "train",
                "path": path.name,
                "url": "https://example.invalid/never-download-fixtures",
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
                "license": "software_fixture",
            }
        )
    lock = folder / "assets.lock.json"
    lock.write_text(json.dumps({"assets": assets}))
    overlap = fixture_data(folder / "overlap", train=2, dev=2)
    settings = {
        **base["data"],
        "asset_lock": str(lock),
        "assets_root": str(asset_root),
        "source_asset_ids": [a["id"] for a in assets],
        "max_source_rows": 1000,
        "parquet_batch_rows": 8,
        "max_answer_tokens": 16,
        "codec_minimum_answer_positions": 20,
        "codec_dev_examples": 3,
        "codec_validation_examples": 2,
        "generator_processed_tokens": 100,
        "max_examples": 100,
        "max_unique_tokens": 20000,
        "split_token_caps": {"train": 17000, "dev": 3000},
        "overlap": {"artifact": str(overlap), "policy": "explicit_both_splits_fixture_v1"},
    }
    data = {**common, "stage": "data", "data": settings, "output_ref": str(folder / "standalone-data.json")}
    return {"folder": folder, "common": common, "data": data, "rows": rows, "assets": assets, "lock": lock}


@pytest.fixture
def wave_fixture(tulu_fixture):
    import copy
    from ncm_lab.runtime.io import write_json
    from ncm_lab.config.waves import wave_config

    fixture = tulu_fixture
    folder = fixture["folder"] / "wave-configs"
    folder.mkdir()
    common = copy.deepcopy(fixture["common"])
    values = {"data": copy.deepcopy(fixture["data"])}
    for name, inputs in [
        ("teacher", {"data": "unused"}),
        ("codec", {"features": "unused"}),
        ("targets", {"data": "unused", "codec": "unused"}),
        ("generator", {"targets": "unused"}),
    ]:
        value = {**copy.deepcopy(common), "stage": name, "inputs": inputs, "output_ref": None}
        if name == "teacher":
            value["data"] = {"max_unique_answer_positions": 20, "max_dev_examples": 3}
        if name == "codec":
            value["training"].update(
                objective="epochs", epochs=1, steps=100, validation_interval=100, checkpoint_interval=100
            )
        if name == "generator":
            value["training"].update(
                objective="tokens",
                max_processed_tokens=100,
                warmup_processed_tokens=1,
                steps=20,
                epochs=100,
                validation_interval=100,
                checkpoint_interval=100,
            )
        values[name] = value
    for name in ("prepare_sources", "prepare_model"):
        value = {
            **copy.deepcopy(common),
            "stage": "prepare",
            "preparation": {
                "asset_lock": str(fixture["lock"]),
                "assets_root": fixture["data"]["data"]["assets_root"],
                "asset_ids": [a["id"] for a in fixture["assets"]],
                "stages": {},
                "budget": {
                    "schema_version": 1,
                    "phase_budget_id": common["budget"]["phase_id"] + "-" + name,
                    "max_download_bytes": 1000000,
                    "max_storage_bytes": 10000000,
                    "max_wall_seconds": 30,
                    "max_phase_wall_seconds": 300,
                    "reserve_free_bytes": 1,
                    "max_processed_bytes": 10000000,
                    "max_prompt_examples": 100,
                    "max_prompt_chars": 131072,
                    "dev_fraction": 0.15,
                },
            },
        }
        values[name] = value
    paths = {}
    for name, value in values.items():
        path = folder / (name + ".json")
        write_json(path, value, exclusive=True)
        paths[name] = str(path)
    path = folder / "wave.json"
    write_json(
        path,
        {
            "schema": "native_wave_v1",
            "phase_id": common["budget"]["phase_id"],
            "run_root": common["run_root"],
            "limits": {
                "max_wave_wall_seconds": 300,
                "max_controller_seconds": 90,
                "max_phase_controller_seconds": 720,
                "max_log_bytes": 1048576,
                "reserve_free_bytes": 1,
                "max_phase_attempts": 8,
            },
            "stages": paths,
            "notes": "software fixture only",
        },
        exclusive=True,
    )
    return {**fixture, "config_path": path, "stage_paths": paths, "wave": wave_config(path)}
