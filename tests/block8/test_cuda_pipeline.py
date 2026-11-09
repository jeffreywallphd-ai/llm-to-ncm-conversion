"""Bounded all-stage CUDA fixture with the same 36-layer production classes."""

import pytest
import torch
from ncm_lab.block8.config import resolve
from ncm_lab.block8.workflow import run_workflow
from ncm_lab.block8.runtime import checked
from ncm_lab.runtime.io import read, write_json


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; tiny hierarchical pipeline only")
def test_tiny_cuda_teacher_codec_targets_and_hierarchical_generator(block8_fixture):
    f = block8_fixture
    cfg = read(f["config"])
    cfg.update(device="cuda", dtype="bfloat16", hardware_profile="cuda")
    cfg["features"]["teacher_dtype"] = "bfloat16"
    cfg["architecture"]["activation_checkpointing"] = True
    cfg["training"]["steps"] = 2
    cfg["codec"]["steps"] = 2
    cfg["decoding"].update(max_new_tokens=16, max_blocks=2, max_evaluation_examples=1)
    write_json(f["config"], cfg)
    result = run_workflow(
        resolve(f["config"]),
        experiment_dir=f["folder"] / "cuda hierarchical pipeline",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert result["status"] == "completed", result
    assert result["completed_step"] == result["codec_step"] == 2
    state = read(result["experiment"])
    assert set(state["stages"]) == {"prepare", "teacher", "codec", "targets", "generator", "evaluate"}
    codec = read(checked(result["codec"]))
    assert codec["teacher_head_frozen"] and codec["train_only_statistics"]
    evaluation = read(checked(result["evaluation"]["report"]))
    assert evaluation["rows"][0]["hierarchy_updates"] in {32, 64}
