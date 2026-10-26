from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import pytest
from ncm_lab.data.download import (PreparationBudget, AcquisitionMeter, asset_path,
                                    download_asset, verify_asset)
from ncm_lab.data.prepare_prompts import prepare_prompts
from ncm_lab.runtime.io import root, read, digest, write_json
from ncm_lab.runtime.prepare import prepare, selected_assets, resolve_inputs, write_launch_configs


def budget(**changes):
    return replace(PreparationBudget(**read(root()/"configs/preparation/pilot.yaml")),**changes)


def asset(body=b"licensed public artifact bytes",path="target/revision/LICENSE"):
    return {"id":"test:artifact","role":"target","path":path,"size_bytes":len(body),
            "sha256":hashlib.sha256(body).hexdigest(),"git_blob_sha1":None,
            "url":"https://fixture.invalid/pinned","revision":"a"*40,"filename":"LICENSE","groups":["pilot"]}


class Response(io.BytesIO):
    def __init__(self,body,status=200,headers=None):
        super().__init__(body)
        self.status,self.headers = status,headers or {"Content-Length":str(len(body))}


def test_resumed_download_and_verified_cache_need_no_second_fetch(tmp_path):
    body = b"pinned asset with a partial transfer"
    spec = asset(body)
    dest = asset_path(tmp_path,spec)
    dest.parent.mkdir(parents=True)
    dest.with_suffix(dest.suffix+".partial").write_bytes(body[:9])
    meter = AcquisitionMeter(tmp_path,budget(),"selection")
    requests = []
    def opener(request,timeout):
        requests.append(request)
        assert request.headers["Range"] == "bytes=9-"
        return Response(body[9:],206,{"Content-Range":f"bytes 9-{len(body)-1}/{len(body)}","Content-Length":str(len(body)-9)})
    result = download_asset(tmp_path,spec,meter,opener)
    assert result["sha256"] == spec["sha256"] and dest.read_bytes() == body
    again = download_asset(tmp_path,spec,meter,lambda *a,**k:pytest.fail("verified cached file was downloaded"))
    assert again["cached"] and len(requests)==1
    assert meter.state["charged_download_bytes"] == len(body)-9


def test_server_ignores_resume_restarts_with_actual_full_file(tmp_path):
    body = b"complete artifact"
    spec = asset(body)
    dest = asset_path(tmp_path,spec)
    dest.parent.mkdir(parents=True)
    dest.with_suffix(dest.suffix+".partial").write_bytes(body[:3])
    result = download_asset(tmp_path,spec,AcquisitionMeter(tmp_path,budget(),"s"),lambda *a,**k:Response(body))
    assert dest.read_bytes() == body and result["sha256"] == spec["sha256"]


def test_checksum_failure_and_budget_stop_never_make_final_asset(tmp_path):
    spec = asset(b"correct")
    meter = AcquisitionMeter(tmp_path,budget(),"s")
    with pytest.raises(ValueError,match="checksum"):
        download_asset(tmp_path,spec,meter,lambda *a,**k:Response(b"corrupt"))
    assert not asset_path(tmp_path,spec).exists()
    # Recover conservatively into a separate declared phase; a transfer cap must stop before reading.
    constrained = budget(phase_budget_id="limited",max_download_bytes=2)
    other = asset(b"large",path="other/model")
    with pytest.raises(ValueError,match="download budget"):
        download_asset(tmp_path,other,AcquisitionMeter(tmp_path,constrained,"s"),lambda *a,**k:Response(b"large"))
    assert not asset_path(tmp_path,other).exists()


def test_partial_range_and_git_blob_integrity(tmp_path):
    body = b"small source file"
    spec = asset(body)
    spec.update(sha256=None,git_blob_sha1=hashlib.sha1(f"blob {len(body)}\0".encode()+body,usedforsecurity=False).hexdigest())
    dest = asset_path(tmp_path,spec)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(body)
    assert verify_asset(dest,spec) == hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError,match="asset path"):
        asset_path(tmp_path,asset(path="../outside"))


def test_preparation_plan_is_offline_and_disk_block_prevents_download(tmp_path,monkeypatch):
    import ncm_lab.runtime.prepare as module
    monkeypatch.setattr(module,"disk_free",lambda base:0)
    monkeypatch.setattr(module,"download_asset",lambda *a,**k:pytest.fail("download began despite disk blocker"))
    planned = prepare(tmp_path/"assets",dry_run=True)
    assert planned["status"] == "planned" and planned["blockers"] and not planned["network_started"]
    blocked = prepare(tmp_path/"assets")
    assert blocked["status"] == "blocked" and not (tmp_path/"assets").exists()
    assert not any(a.get("split")=="test" for a in selected_assets())
    assert any(a["role"]=="docfinqa" for a in selected_assets(True))


def test_phase_budget_and_selection_cannot_silently_change(tmp_path):
    b = budget()
    AcquisitionMeter(tmp_path,b,"original")
    with pytest.raises(ValueError,match="selection changed"):
        AcquisitionMeter(tmp_path,b,"different")
    with pytest.raises(ValueError):
        replace(b,max_wall_seconds=float("inf"))


def test_acquisition_wall_time_limits_include_recovery_history(tmp_path):
    b = budget()
    meter = AcquisitionMeter(tmp_path,b,"same")
    meter.started -= b.max_wall_seconds+1
    with pytest.raises(ValueError,match="wall-time budget"):
        meter.check()
    # A fresh invocation cannot erase time already spent in this phase.
    meter.started += b.max_wall_seconds+1
    meter.state["elapsed_seconds"] = b.max_phase_wall_seconds
    meter.save()
    recovered = AcquisitionMeter(tmp_path,b,"same")
    with pytest.raises(ValueError,match="wall-time budget"):
        recovered.check()


def test_official_parquet_schema_prompts_split_without_gold_and_deduplicate(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    rows = [{"message_id":str(i),"message_tree_id":f"tree-{i}","text":f"A distinct public prompt number {i}",
             "role":"prompter","lang":"en","deleted":False,"review_result":True,"answer":"FORBIDDEN_GOLD"}
            for i in range(150)]
    rows += [{**rows[0],"message_id":"copy"}, {**rows[0],"message_id":"deleted","deleted":True},
             {**rows[0],"message_id":"rejected","review_result":False}]
    source = tmp_path/"official_train.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows),source)
    gsm = tmp_path/"gsm_train.jsonl"
    gsm.write_text(json.dumps({"question":rows[2]["text"],"answer":"HIDDEN_REFERENCE"})+"\n")
    result = prepare_prompts(source,gsm,tmp_path/"processed",budget(max_prompt_examples=1000))
    train,dev = [Path(result["outputs"][s]["path"]).read_text(encoding="utf-8") for s in ("train","dev")]
    assert "FORBIDDEN_GOLD" not in train+dev and "HIDDEN_REFERENCE" not in train+dev
    assert "deleted" not in train+dev and "rejected" not in train+dev
    train_rows,dev_rows = [[json.loads(line) for line in text.splitlines()] for text in (train,dev)]
    assert not {r["group_id"] for r in train_rows} & {r["group_id"] for r in dev_rows}
    assert result["filtered"]["duplicate"]==1 and result["filtered"]["gsm8k_overlap"]>=1
    assert all(set(row)=={"original_id","question","group_id","split"} for row in train_rows+dev_rows)


def test_generated_configs_connect_teacher_output_without_copying_dataset(tmp_path):
    from ncm_lab.training.loop import resolve_training_data
    from ncm_lab.experiments.runner import prerequisites
    base,b = resolve_inputs(tmp_path/"assets")
    assets = selected_assets()
    prompts = {"outputs":{"train":{"path":str(tmp_path/"prompts.jsonl")}}}
    paths = write_launch_configs(tmp_path/"attempt",base,assets,prompts,"spark",b)
    config = read(paths["pilot_config"])
    assert "training_data" not in config and "training_data_reference" in config
    assert any("teacher data not prepared" in message for message in prerequisites(config))
    data = tmp_path/"teacher.jsonl"
    data.write_text('{"origin":"target_distillation"}\n')
    write_json(config["training_data_reference"],{"path":str(data),"sha256":digest(data)})
    assert resolve_training_data(config)==data
    data.write_text("changed")
    with pytest.raises(ValueError,match="missing or changed"):
        resolve_training_data(config)


def test_complete_prepare_and_retry_use_checksums_provenance_and_separate_launches(tmp_path,monkeypatch):
    """Software asset fixtures only: no target forward or fake research metrics."""
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import ncm_lab.runtime.prepare as module
    import ncm_lab.runtime.doctor as health
    source = tmp_path/"source.parquet"
    rows = [{"message_id":str(i),"message_tree_id":f"tree-{i}","text":f"Distinct prompt {i}","role":"prompter",
             "lang":"en","deleted":False,"review_result":True} for i in range(100)]
    parquet.write_table(arrow.Table.from_pylist(rows),source)
    bodies = {"target":b"{}","draft":b"{}","oasst1":source.read_bytes(),
              "gsm8k":b'{"question":"unrelated task question","answer":"HIDDEN_GOLD"}\n'}
    assets = []
    for role,body in bodies.items():
        filename = "config.json" if role in {"target","draft"} else "train.parquet" if role=="oasst1" else "train.jsonl"
        item = asset(body,path=f"{role}/{'a'*40}/{filename}")
        item.update(id=role,role=role,filename=filename,split="train" if role in {"oasst1","gsm8k"} else None)
        assets.append(item)
    monkeypatch.setattr(module,"selected_assets",lambda *a:assets)
    monkeypatch.setattr(module,"disk_free",lambda base:100_000_000_000)
    monkeypatch.setattr(health,"doctor",lambda profile:{"torch":{"cuda_available":False},"spark_ready":False})
    calls = []
    def transfer(base,item,meter,progress=None):
        def opener(*a,**k):
            calls.append(item["id"])
            return Response(bodies[item["role"]])
        return download_asset(base,item,meter,opener)
    monkeypatch.setattr(module,"download_asset",transfer)
    first = prepare(tmp_path/"assets",profile="spark")
    assert first["status"]=="completed",first
    assert first["assets_verified"] and not first["ready_for_training"] and not first["teacher_generation_started"]
    assert Path(first["configs"]["teacher_config"]).exists() and Path(first["configs"]["pilot_config"]).exists()
    assert "HIDDEN_GOLD" not in Path(first["prompts"]["outputs"]["train"]["path"]).read_text()
    second = prepare(tmp_path/"assets",profile="spark")
    assert second["status"]=="completed" and len(calls)==4
    assert all(r["cached"] for r in second["download_receipts"])
    assert first["attempt"] != second["attempt"] and (Path(first["attempt"])/"manifest.json").exists()


def test_teacher_launch_caps_examples_before_budget_and_publishes_canonical_reference(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import ncm_lab.training.teacher as module
    from ncm_lab.experiments.executors import teacher
    source = tmp_path/"prompts.jsonl"
    source.write_text("\n".join(json.dumps({"question":str(i),"split":"train","group_id":str(i)}) for i in range(10)))
    run_path = tmp_path/"run"
    run_path.mkdir()
    def generate(prompts,model,output,meter,*args):
        assert len(prompts)==2
        output.write_text('{"origin":"target_distillation","input_ids":[1,2],"split":"train","group_id":"a"}\n')
        return {"status":"completed","records":2}
    monkeypatch.setattr(module,"generate_teacher",generate)
    reference = tmp_path/"reference.json"
    config = {"prompts_path":str(source),"model":{},"device":"cpu","precision":"float32",
              "max_examples":2,"teacher_output_reference":str(reference)}
    result = teacher(SimpleNamespace(path=run_path,id="fixture"),config,SimpleNamespace(budget=SimpleNamespace(max_steps=2)))
    assert result["status"]=="completed"
    assert read(reference)["sha256"]==digest(run_path/"teacher.jsonl")
