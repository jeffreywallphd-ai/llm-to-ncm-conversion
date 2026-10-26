import asyncio
import json
import zipfile
import pytest
from ncm_lab.business.pipeline import parse_workpaper, execute_workpapers, workpaper_prompt
from ncm_lab.business.workload import replay
from ncm_lab.contracts import ModelInput
from ncm_lab.reporting.export import review_export
from ncm_lab.reporting.reports import selected_runs
from ncm_lab.runtime.io import root, read, write_json
from ncm_lab.training.losses import final_token_loss
import torch


def test_workpaper_contract_and_bounded_gold_free_repair():
    item = ModelInput("a","finqa","dev","report","What is revenue?",("Revenue is 7 USD.",))
    prompts = []
    def generate(prompt):
        prompts.append(prompt)
        payload = json.loads(prompt.split("\nPrevious output:")[0])
        if len(prompts) == 1:
            return "malformed"
        return json.dumps({"answers":[{"question_id":"a","answer":7,"unit":"USD","scale":"", "period":None,
            "source_refs":[payload["evidence"][0]["evidence_id"]],"program":"add(5, 2)"}]})
    result = list(execute_workpapers([item],generate,repair_attempts=1,max_questions=1))[0]
    assert len(prompts) == 2 and result["outcome"] == "returned"
    assert "gold" not in workpaper_prompt([item],[]).lower()
    with pytest.raises(ValueError):
        parse_workpaper('{"answers":[]}',{"a"})


def test_controlled_load_keeps_failed_and_deadline_requests():
    async def worker(request):
        if request["original_id"] == "bad":
            raise RuntimeError("intentional fixture failure")
        if request["original_id"] == "slow":
            await asyncio.sleep(.1)
        return {"correct":True}
    rows = [{"original_id":i,"group_id":i} for i in ("ok","bad","slow")]
    result = asyncio.run(replay(rows,worker,concurrency=3,deadline_seconds=.01,max_requests=3))
    assert result["summary"]["requests"] == 3 and result["summary"]["correct"] == 1
    assert result["summary"]["errors"] == 2
    assert sum(r["deadline_met"] for r in result["records"]) == 1


def test_study_export_is_selected_anonymous_and_keeps_notices(tmp_path):
    output = tmp_path/"acl.zip"
    review_export("acl",output,True)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "README.md" in names and "pyproject.toml" in names
        assert "src/ncm_lab/vendor/dflash/LICENSE.txt" in names
        assert not any("experiments/is_journal/" in name for name in names)
        assert not any("src/ncm_lab/business/" in name for name in names)
        registry = json.loads(archive.read("experiments/registry.yaml"))
        assert all(row["study"] in {"acl","shared"} for row in registry["experiments"])
        assert b"jeffreywallphd" not in b"".join(archive.read(n) for n in names)
        assert not json.loads(archive.read("REVIEW_REPORT.json"))["submission_ready"]


def test_missing_or_fixture_study_evidence_is_rejected(tmp_path,monkeypatch):
    import ncm_lab.reporting.reports as reports
    study = read(root()/"publications/acl/study.yaml")
    study["selected_runs"] = [{"run_id":"nonexistent", "manifest_sha256":"0"*64}]
    base = tmp_path/"repo"
    write_json(base/"publications/acl/study.yaml",study)
    monkeypatch.setattr(reports,"root",lambda:base)
    with pytest.raises(ValueError,match="uniquely"):
        selected_runs("acl")


def test_final_loss_excludes_anchor_invalid_positions_and_preserves_gradients():
    logits = torch.randn(1,3,5,requires_grad=True)
    labels = torch.tensor([[1,-100,3]])
    loss = final_token_loss(logits,labels)
    loss.backward()
    assert logits.grad[0,1].abs().sum() == 0
    assert logits.grad[0,0].abs().sum() > 0 and logits.grad[0,2].abs().sum() > 0
