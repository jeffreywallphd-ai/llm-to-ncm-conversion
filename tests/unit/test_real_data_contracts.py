from dataclasses import asdict
import copy
import uuid
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ncm_lab.config.stages import config, Training
from ncm_lab.config.architecture import tiny_architecture, Architecture
from ncm_lab.data.prepare import validate_source_binding, raw_examples
from ncm_lab.data.artifacts import Dataset
from ncm_lab.runtime.io import read, root, digest
from ncm_lab.runtime.runner import execute
from ncm_lab.runtime.budget import Budget
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.runtime.doctor import parameter_inventory, preflight


def test_shipped_real_data_configs_resolve_actual_pinned_training_parquet():
    lock = read(root() / "configs/data/assets.lock.json")
    for path in ("configs/data/real_pilot.yaml", "configs/data/initial_100m_source_pool.yaml"):
        settings = config(path)["data"]
        selected = validate_source_binding(settings, lock)
        assert selected["split"] == "train" and selected["filename"].endswith(".parquet")
        assert selected["sha256"] == "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"
        assert settings["source_path"] == "runs/assets/" + selected["path"]


def test_license_binding_rejected_before_tokenizer_or_content_loading(monkeypatch, tmp_path):
    value = config("configs/data/real_pilot.yaml")
    license = next(
        a for a in read(root() / "configs/data/assets.lock.json")["assets"] if a["id"] == "oasst1:LICENSE"
    )
    value["data"].update(
        source_asset_id=license["id"], source_path="runs/assets/" + license["path"], source_sha256=None
    )
    import ncm_lab.data.prepare as prepare

    def forbidden(*a, **kw):
        raise AssertionError("invalid asset reached tokenizer/hash loading")

    monkeypatch.setattr(prepare, "load_tokenizer", forbidden)
    monkeypatch.setattr(prepare, "digest", forbidden)
    run = type("Run", (), {"meter": None})()
    with pytest.raises(ValueError, match="training split"):
        prepare.prepare_data(run, value)
    for changes in ({"revision": "wrong"}, {"source_path": "wrong.parquet"}, {"format": "jsonl"}):
        settings = copy.deepcopy(config("configs/data/real_pilot.yaml")["data"])
        settings.update(changes)
        with pytest.raises(ValueError):
            validate_source_binding(settings)


def test_actual_oasst_parquet_reader_and_preparation(tmp_path):
    rows = []
    for i in range(24):
        for suffix, role, parent, text in [
            ("p", "prompter", None, "token" + str(i + 4)),
            ("a", "assistant", str(i) + "p", "token8 token12"),
        ]:
            rows.append(
                {
                    "message_id": str(i) + suffix,
                    "parent_id": parent,
                    "message_tree_id": str(i),
                    "role": role,
                    "text": text,
                    "lang": "en",
                    "deleted": False,
                    "review_result": True,
                }
            )
    rows.extend(
        [
            {**rows[1], "message_id": "deleted", "deleted": True},
            {**rows[1], "message_id": "foreign", "lang": "fr"},
            {**rows[1], "message_id": "unapproved", "review_result": False},
            {**rows[1], "message_id": "wrongtree", "message_tree_id": "other"},
        ]
    )
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    settings = {
        "source_path": str(path),
        "source_sha256": digest(path),
        "revision": "software-fixture",
        "license": "software fixture",
        "format": "oasst_parquet",
        "max_source_rows": 100,
        "max_examples": 24,
        "max_unique_tokens": 10000,
        "max_prompt_tokens": 128,
        "dev_fraction": 0.5,
        "overlength": "exclude",
    }
    source = list(raw_examples(settings, tmp_path / "parents.sqlite"))
    assert len(source) == 24 and {r["group_id"] for r in source} == {str(i) for i in range(24)}
    assert all(r["prompt"].startswith("token") and r["answer"] == "token8 token12" for r in source)
    value = {
        "schema": "native_stage_v1",
        "stage": "data",
        "model": model_spec(),
        "architecture": tiny_architecture().to_dict(),
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "budget": asdict(Budget("parquet-" + uuid.uuid4().hex)),
        "training": asdict(Training()),
        "inputs": {},
        "data": settings,
        "run_root": str(tmp_path / "runs"),
    }
    result = execute(value)
    assert result["status"] == "completed", result
    data = Dataset(result["artifact"])
    assert len(data) == 24 and not set(data.manifest["groups"]["train"]) & set(data.manifest["groups"]["dev"])
    assert all(
        data[i][0].prompt_ids != data[i][0].answer_ids and data[i][0].answer_ids[-1] == 2
        for i in range(len(data))
    )


def test_actual_meta_inventory_and_selected_precision_memory(tmp_path):
    model = config("configs/validation/generator_pilot.yaml")["model"]
    inventory = parameter_inventory(model, Architecture().to_dict())
    assert (
        inventory["original"] == 8190735360
        and inventory["generator"] == 8250600480
        and inventory["codec"] == 6901504
    )
    value = config("configs/validation/generator_pilot.yaml")
    report = preflight(value, tmp_path, training=True)
    assert report["minimum_weights_gradients_moments_bytes"] == 99007205760
    assert report["one_recovery_bytes_estimate"] == 82506004800
    assert report["fp32_master_bytes"] == 0 and report["blockers"]
