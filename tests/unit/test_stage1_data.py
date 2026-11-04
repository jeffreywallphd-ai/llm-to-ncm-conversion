"""Data qualification invariants and teacher maximum-length alignment, fixture only."""

import json
from pathlib import Path
import torch
from ncm_lab.stage1.config import resolve
from ncm_lab.stage1.workflow import plan, run, load, checked
from ncm_lab.runtime.io import read, write_json, digest


def test_complete_capacity_eos_dedup_connected_groups_and_stable_partitions(tmp_path, stage1_fixture):
    rows = [json.loads(line) for line in stage1_fixture["source"].read_text(encoding="utf-8").splitlines()]
    rows[0]["answer"] = "token8 " * 7
    rows[1]["answer"] = "token8 " * 8
    rows[11].update(prompt=rows[10]["prompt"], answer=rows[10]["answer"])
    rows[12]["overlap_group_ids"] = [rows[13]["group_id"]]
    stage1_fixture["source"].write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    data_path = Path(stage1_fixture["experiment"]["stages"]["prepare"])
    cfg = read(data_path)
    cfg["data"]["source_sha256"] = digest(stage1_fixture["source"])
    write_json(data_path, cfg)
    reports = []
    for size in ("xsmall", "small"):
        path = Path(
            plan(resolve(stage1_fixture["config"], canvas_size=size), tmp_path / ("data-" + size))[
                "experiment"
            ]
        )
        assert run(path, stage="prepare")["status"] == "ready"
        _, state, _ = load(path)
        result = read(checked(state["stages"]["prepare"]["receipt"]))["result"]
        qualification = read(checked(result["qualification"]))
        reports.append(qualification)
        assert (
            qualification["group_assignments"][rows[10]["group_id"]]
            == qualification["group_assignments"][rows[11]["group_id"]]
        )
        assert (
            qualification["group_assignments"][rows[12]["group_id"]]
            == qualification["group_assignments"][rows[13]["group_id"]]
        )
        groups = [set(qualification["selected"][s]["groups"]) for s in ("train", "dev", "test")]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert reports[0]["group_assignments"] == reports[1]["group_assignments"]
    admitted8 = {i for split in reports[0]["natural_eligible_pool"].values() for i in split["ids"]}
    admitted32 = {i for split in reports[1]["natural_eligible_pool"].values() for i in split["ids"]}
    assert "0" in admitted8 and "1" not in admitted8 and "1" in admitted32
    assert reports[0]["excluded"]["overlength"] == 1 and reports[0]["excluded"]["duplicate"] == 1


def test_teacher_maximum_canvas_prediction_alignment_and_frozen_states(stage1_fixture):
    from ncm_lab.teacher.model import load_original, prediction_states

    teacher = load_original(stage1_fixture["model"], dtype=torch.bfloat16, frozen=True)
    prompt, answer = [1, 4, 5], [8] * 7 + [2]
    states = prediction_states(teacher, prompt, answer)
    with torch.no_grad():
        logits = teacher(torch.tensor([prompt + answer[:-1]])).logits[0, len(prompt) - 1 :]
        projected = teacher.lm_head(states)
    assert states.shape == (8, 8)
    assert torch.equal(logits, projected) and not states.requires_grad
    assert not any(p.requires_grad for p in teacher.parameters())
