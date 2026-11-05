"""Local software evidence for the unattended 64-position workflow and its data gates."""

import json
from pathlib import Path

import pytest

from ncm_lab.data.artifacts import Dataset, LOCKED
from ncm_lab.runtime.io import digest, read, write_json
from ncm_lab.stage1.config import resolve
from ncm_lab.stage1.workflow import STAGES, checked, load, plan, run
from ncm_lab.stage2.launcher import main


def recipe(fixture):
    value = {**fixture["experiment"], "schema": "stage2_single_pass_v1"}
    path = fixture["folder"] / "stage2.json"
    write_json(path, value, exclusive=True)
    return path


def test_unattended_training_locked_eval_and_chat_use_selected_export(
    tmp_path, stage1_fixture, monkeypatch, capsys
):
    config = recipe(stage1_fixture)
    folder = tmp_path / "stage2"
    manifest = folder / "manifest.json"
    calls = []

    def chat(**kwargs):
        _, state, _ = load(manifest)
        assert state["status"] == "completed" and state["locked_test_observed"]
        assert all(entry["status"] == "completed" for entry in state["stages"].values())
        assert not (folder / "execution.lock").exists()
        assert (folder / "chat.lock").is_file()
        calls.append(kwargs)
        return {"status": "stopped"}

    monkeypatch.setattr("ncm_lab.chat.server.serve", chat)
    assert main(["--config", str(config), "--experiment-dir", str(folder), "--execute"]) == 0, (
        capsys.readouterr()
    )
    state = read(manifest)
    assert [event["stage"] for event in state["history"] if event["event"] == "stage_completed"] == list(
        STAGES
    )
    assert not state["codec_review_override"]
    generator = read(checked(state["stages"]["generator"]["receipt"]))["result"]
    codec = read(checked(state["stages"]["codec"]["receipt"]))["result"]
    assert (codec["optimizer_steps"], generator["optimizer_steps"]) == (12, 3)
    assert Path(generator["artifact"]).is_file()
    assert calls[0]["checkpoint"] == generator["artifact"] and calls[0]["device"] == "cpu"
    report = read(folder / "reports/stage2_summary.json")
    assert report["origin"] == "software_fixture" and report["workflow_completed"]
    rows = [
        json.loads(line)
        for line in checked(report["evaluation"]["raw_outputs"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all(r["computed_answer_positions"] == r["allocated_answer_positions"] == 64 for r in rows)
    assert all(r["generation_backbone_traversals"] == 1 for r in rows)
    assert report["evaluation"]["splits"]["test"]["examples"] == 4

    before = digest(manifest)
    assert main(["--resume", str(manifest), "--execute", "--chat-port", "0"]) == 0, capsys.readouterr()
    assert digest(manifest) == before
    assert calls[1]["checkpoint"] == calls[0]["checkpoint"] and calls[1]["port"] == 0
    with pytest.raises(ValueError, match="original fixed generator target"):
        run(manifest, extend_generator_steps=4, reason="not allowed for Stage 2")


def test_tulu_qualification_publishes_disjoint_complete_pairs(tmp_path, stage1_fixture, tulu_fixture, capsys):
    path = recipe(stage1_fixture)
    prepare = Path(stage1_fixture["experiment"]["stages"]["prepare"])
    value = read(prepare)
    value["data"] = {**tulu_fixture["data"]["data"], "max_answer_tokens": 64}
    write_json(prepare, value)
    folder = tmp_path / "stage2-tulu"
    assert main(["--config", str(path), "--experiment-dir", str(folder), "--qualify-only"]) == 0, (
        capsys.readouterr()
    )
    state = read(folder / "manifest.json")
    receipt = read(checked(state["stages"]["prepare"]["receipt"]))["result"]
    data = Dataset(receipt["artifact"], "native_data_v1")
    test = Dataset(checked(receipt["evaluation_data"]), LOCKED)
    assert data.manifest["identity"]["qualification_mode"] == "stage2_locked_short_answers_v1"
    assert {s: len(data.indices(s)) for s in ("train", "dev", "test")} == {"train": 8, "dev": 4, "test": 0}
    assert len(test) == 4
    groups = [set(data[i][0].group_id for i in data.indices(s)) for s in ("train", "dev")]
    groups.append({test[i][0].group_id for i in range(len(test))})
    assert sum(map(len, groups)) == len(set.union(*groups))
    for artifact in (data, test):
        for i in range(len(artifact)):
            ex = artifact[i][0]
            assert ex.eos_genuine and not ex.censored and 0 < len(ex.answer_ids) <= 64
            assert ex.provenance["component_source"] and ex.provenance["reference_text"]
    assert all(state["stages"][s]["status"] == "pending" for s in STAGES[1:])
    assert run(folder / "manifest.json", stage="teacher")["status"] == "ready"
    teacher = read(checked(read(folder / "manifest.json")["stages"]["teacher"]["receipt"]))["result"]
    features = Dataset(teacher["artifact"], "native_features_v1")
    assert len(features) == len(data) and not features.indices("test")


def test_oasst_keeps_standard_stage1_test_groups_out_of_stage2(tmp_path, stage1_fixture, oasst_source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    prepare = Path(stage1_fixture["experiment"]["stages"]["prepare"])
    value = read(prepare)
    parent, child = oasst_source["rows"][:2]
    source = oasst_source["path"]
    rows = []
    for i in range(1800):
        ids = {"message_tree_id": f"tree-{i}"}
        rows += [
            {**parent, **ids, "message_id": f"parent-{i}", "text": f"token4 prompt {i}"},
            {**child, **ids, "message_id": f"child-{i}", "parent_id": f"parent-{i}"},
        ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    value["data"] = {
        **oasst_source["settings"],
        "source_sha256": digest(source),
        "max_source_rows": len(rows),
        "max_examples": 1800,
    }
    write_json(prepare, value)
    stage1 = read(stage1_fixture["config"])
    stage1["qualification"]["test_examples"] = 200
    write_json(stage1_fixture["config"], stage1)
    reports = []
    for label, cfg in (("stage1", stage1_fixture["config"]), ("stage2", recipe(stage1_fixture))):
        manifest = plan(resolve(cfg), tmp_path / label)["experiment"]
        result = run(manifest, stage="prepare")
        assert result["status"] == "ready", result
        reports.append(read(Path(result["report"]).with_suffix(".json"))["qualification"])
    prior, current = reports
    reserved = set(prior["selected"]["test"]["groups"])
    assert len(reserved) == 200 and reserved == set(current["reserved_stage1_test_groups"])
    assert all(reserved.isdisjoint(split["groups"]) for split in current["selected"].values())
    assert current["group_assignments"] == prior["group_assignments"]


def test_blocked_source_does_not_train_or_start_chat(tmp_path, stage1_fixture, monkeypatch, capsys):
    path = recipe(stage1_fixture)
    value = read(path)
    value["qualification"]["train_examples"] = 1000
    write_json(path, value)
    folder = tmp_path / "stage2-blocked"
    monkeypatch.setattr(
        "ncm_lab.chat.server.serve", lambda **_: pytest.fail("chat started on blocked source")
    )
    assert main(["--config", str(path), "--experiment-dir", str(folder), "--execute"]) == 2
    state = read(folder / "manifest.json")
    assert state["stages"]["prepare"]["status"] == "blocked", capsys.readouterr()
    assert all(state["stages"][s]["status"] == "pending" for s in STAGES[1:])
    report = read(folder / "reports/stage2_summary.json")
    assert report["qualification"]["natural_eligible_pool"]["train"]["examples"] < 1000
