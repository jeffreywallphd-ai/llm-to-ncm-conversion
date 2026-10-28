"""Regression evidence for FIX-02..12 and PAPER-02..06; never research evidence."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
import torch
from ncm_lab.models.factory import build_drafter, resolve_method
from ncm_lab.models.codecs import MultiscaleCodec, pool
from ncm_lab.models.upstream import FeatureExtractor
from ncm_lab.decoding.policy import GenerationPolicy, resolve_policy, ar_greedy
from ncm_lab.decoding.hf import CachedProposer, cached_greedy
from ncm_lab.training.records import normalize_record, block_labels, anchors
from ncm_lab.training.loop import train, forward_block
from ncm_lab.training.checkpoints import read_checkpoint, atomic_checkpoint
from ncm_lab.training.latent_supervision import teacher_block, cache_identity
from ncm_lab.runtime.budget import Budget, Meter
from ncm_lab.runtime.io import read, root, digest
from ncm_lab.experiments.acceptance import fixture_rows


def meter(path, **limits):
    path.mkdir(exist_ok=True)
    budget = replace(Budget(**read(root()/"configs/budgets/acceptance.yaml")), **limits)
    return Meter(budget, path)


@pytest.mark.parametrize("method", ["FLAT_ADAPTED", "HIER_SKIP", "LATE_NEAR_BUDGET","EARLIER","LATER","HIER_POOLED_CONTEXT"])
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_factory_real_forward_backward_placement(pair, method, precision):
    target, backbone = pair
    before = {k: p.detach().clone() for k, p in target.named_parameters()}
    drafter = build_drafter(target, backbone, resolve_method({"method": method, "precision": precision}))
    row = fixture_rows(target, "train")[0]
    logits, labels = forward_block(target, drafter, row, row["prompt_length"])
    assert logits.shape == (1, 15, 32)
    from ncm_lab.training.losses import final_token_loss
    final_token_loss(logits, labels).backward()
    optimizer = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad])
    optimizer.step()
    assert drafter.embeddings is target.get_input_embeddings()
    assert drafter.head is target.get_output_embeddings()
    assert all(p.grad is None for p in target.parameters())
    assert all(torch.equal(p, before[k].to(p)) for k, p in target.named_parameters())
    assert drafter.backbone.fc.weight.grad is not None
    assert all(not b.is_floating_point() or b.dtype == getattr(torch, precision) for b in drafter.buffers())


def test_response_anchor_alignment_partial_terminal_and_leakage(pair):
    target, backbone = pair
    row = normalize_record({"input_ids": [1]*20+[7, 8, 2], "prompt_length": 20,
        "group_id": "a", "split": "train", "origin": "software_fixture", "generation_policy": {"eos_ids": [2,31]}},
        origin="software_fixture")
    assert anchors(row) == [20,21]
    assert block_labels(row, 20)[0].tolist() == [8,2]+[-100]*13
    states, valid, labels = teacher_block(target, row, 20)
    out = target(torch.tensor([row["input_ids"]]), output_hidden_states=True)
    assert torch.allclose(target.get_output_embeddings()(states[:, :2]), out.logits[:, 20:22])
    assert valid.sum() == 2 and labels[0,1] == 2
    drafter = build_drafter(target, backbone, resolve_method({"method":"HIER_SKIP"}))
    original, _ = forward_block(target, drafter, row, 20)
    changed = copy.deepcopy(row)
    changed["input_ids"][21:] = [12,13]
    result, _ = forward_block(target, drafter, changed, 20)
    assert torch.equal(original, result)
    ids = torch.tensor([row["input_ids"]])
    features = FeatureExtractor(target, backbone.target_layer_ids)(ids)
    positions = torch.arange(ids.shape[1])[None]
    features[:,20:] = float("nan")
    result = drafter(features,positions,torch.zeros_like(positions),torch.tensor([20]),ids[:,20],torch.tensor([0]))
    assert torch.equal(original, result)
    bad = {**row, "response_mask": [True]*23}
    with pytest.raises(ValueError, match="prompt labels"):
        normalize_record(bad, origin="software_fixture")


@pytest.mark.parametrize("eos", [2,31])
@pytest.mark.parametrize("proposal", [[5,2,9], [9,9,9], [5,9,9]])
def test_common_policy_multieos_rejection_and_bonus(pair, eos, proposal):
    target, _ = pair
    original = target.forward
    prompt = torch.tensor([[1,4,8]])
    expected = [4,5,eos]
    def controlled(*args, **kwargs):
        old = kwargs["past_key_values"].get_seq_length()
        out = original(*args, **kwargs)
        logits = torch.full_like(out.logits, -100)
        for j in range(logits.shape[1]):
            token = expected[min(max(old+j-2,0),2)]
            logits[:,j,token] = 100
        out.logits = logits
        return out
    target.forward = controlled
    policy = GenerationPolicy(15, (2,31))
    q = [eos if token == 2 else token for token in proposal]
    actual = cached_greedy(target, lambda *_: torch.tensor([q]), prompt, policy=policy)
    assert actual.tokens == ar_greedy(target,prompt,policy) == expected
    assert len(actual.tokens) == 1+actual.accepted+actual.corrections+actual.bonus
    assert actual.telemetry["policy_hash"] == policy.digest
    target.generation_config.eos_token_id = [2,31]
    assert resolve_policy(target).eos_ids == (2,31)
    target.generation_config.repetition_penalty = 1.2
    with pytest.raises(ValueError, match="processor"):
        resolve_policy(target)


def test_incremental_features_target_call_counts_and_oracle(pair):
    target, backbone = pair
    drafter = build_drafter(target, backbone, resolve_method({"method":"HIER_SKIP"})).eval()
    proposer = CachedProposer(drafter)
    original = target.forward
    lengths = []
    def measured(*args, **kwargs):
        lengths.append(kwargs["input_ids"].shape[1])
        return original(*args, **kwargs)
    target.forward = measured
    compare = proposer.from_features
    def checked(features, confirmed, anchor):
        recomputed = FeatureExtractor(target, backbone.target_layer_ids)(confirmed)
        assert torch.allclose(features, recomputed, atol=1e-5, rtol=1e-5)
        return compare(features, confirmed, anchor)
    prompt = torch.tensor([[1,4,8]])
    policy = GenerationPolicy(18)
    actual = cached_greedy(target,proposer,prompt,policy=policy)
    assert len(lengths) == 1+actual.verification_calls
    assert lengths[0] == 3
    assert sum(lengths) == actual.telemetry["target_processed_positions"]
    proposer.from_features = checked
    other = cached_greedy(target,proposer,prompt,policy=policy)
    assert actual.tokens == other.tokens == ar_greedy(target,prompt,policy)
    lengths.clear()
    assert cached_greedy(target,proposer,prompt,policy=GenerationPolicy(0)).tokens == []
    assert not lengths


def test_seed_initialization_and_exact_resume(pair, tmp_path):
    target, backbone = pair
    config = {"method":"HIER_SKIP","init_seed":9,"data_seed":4}
    models = [build_drafter(target,copy.deepcopy(backbone),resolve_method(config)) for _ in range(3)]
    different = build_drafter(target,copy.deepcopy(backbone),resolve_method({**config,"init_seed":10}))
    assert torch.equal(models[0].skip_down.weight,models[1].skip_down.weight)
    assert not torch.equal(models[0].skip_down.weight,different.skip_down.weight)
    rows = fixture_rows(target,"train")
    full = train(target,models[0],rows,meter(tmp_path/"full"),"data",steps=3,seed=4)
    first = train(target,models[1],rows,meter(tmp_path/"first"),"data",steps=1,seed=4)
    resumed = train(target,models[2],rows,meter(tmp_path/"resumed"),"data",steps=3,seed=4,
                    resume=tmp_path/"first/final.pt")
    assert first["optimizer_steps"] == 1
    assert full["sampling_order"] == resumed["sampling_order"]
    assert full["source_token_passes"] == resumed["source_token_passes"]
    flat_model=build_drafter(target,copy.deepcopy(backbone),resolve_method({'method':'FLAT_ADAPTED','init_seed':9}))
    flat=train(target,flat_model,rows,meter(tmp_path/'flat'),'data',steps=3,seed=4)
    assert flat_model.parameter_counts()['trainable']!=models[0].parameter_counts()['trainable']
    assert flat['sampling_order']==full['sampling_order']
    for key,value in models[0].state_dict().items():
        assert torch.equal(value,models[2].state_dict()[key]), key
    saved = read_checkpoint(tmp_path/"resumed/final.pt")
    assert not any(k.startswith(("embeddings.","head.")) for k in saved["model"])
    a,b = read_checkpoint(tmp_path/"full/final.pt")["optimizer"],saved["optimizer"]
    for key,value in a["state"].items():
        assert all(torch.equal(tensor,b["state"][key][k]) for k,tensor in value.items())


def test_storage_preflight_and_interrupted_write_preserve_recovery(pair,tmp_path,monkeypatch):
    target, backbone = pair
    drafter = build_drafter(target,backbone,resolve_method({"method":"HIER_SKIP"}))
    result = train(target,drafter,fixture_rows(target,"train"),meter(tmp_path/"small",max_storage_bytes=1024),"a")
    assert result["status"] == "cancelled" and not (tmp_path/"small/final.pt").exists()
    path = tmp_path/"checkpoint.pt"
    atomic_checkpoint(path,{"identity":"x","model":{"x":torch.ones(3)}})
    before = digest(path)
    def interrupted(payload,stream):
        stream.write(b"interrupted")
        raise KeyboardInterrupt
    monkeypatch.setattr(torch,"save",interrupted)
    with pytest.raises(KeyboardInterrupt):
        atomic_checkpoint(path,{"identity":"x","model":{}})
    assert digest(path) == before and not path.with_suffix(".partial").exists()


@pytest.mark.parametrize("method", ["FLAT_MULTI_AE","HIER_MULTI_AE","HIER_MULTI_AE_NO_SKIP",
    "HIER_MULTI_AE_AUX_ONLY","HIER_CODEC_CE_ONLY","HIER_LATENT_CE_ONLY","HIER_SINGLE_AE",
    "HIER_LINEAR_CODEC","HIER_INDEPENDENT_AE"])
def test_causal_transfer_zero_bridge_and_enabled_gradient(pair,method):
    target, backbone = pair
    single = method == "HIER_SINGLE_AE"
    kind="linear" if method=="HIER_LINEAR_CODEC" else "independent" if single or method=="HIER_INDEPENDENT_AE" else "nested"
    codec = MultiscaleCodec(16,4,kind=kind,
                            levels=(8,) if single else (4,8,15)).requires_grad_(False)
    config = {"method":method,"codec_sha256":"fixture-codec","seed":7}
    model = build_drafter(target,copy.deepcopy(backbone),resolve_method(config),
                           None if method == "HIER_LATENT_CE_ONLY" else codec,latent_width=4)
    base_name = "FLAT_ADAPTED" if method.startswith("FLAT") else "HIER_NO_SKIP" if method.endswith("NO_SKIP") else "HIER_SKIP"
    baseline = build_drafter(target,copy.deepcopy(backbone),resolve_method({"method":base_name,"seed":7}))
    row = fixture_rows(target,"train")[0]
    a = row["prompt_length"]
    original, _ = forward_block(target,baseline,row,a)
    initial, _ = forward_block(target,model,row,a)
    assert torch.equal(original,initial)
    assert model.stage_lengths == ([16]*5 if method.startswith("FLAT") else [5,5,9,16,16])
    if method != "HIER_MULTI_AE_AUX_ONLY":
        with torch.no_grad():
            for gate in model.latent.gates.values():
                gate.fill_(.5)
        changed, _ = forward_block(target,model,row,a)
        assert not torch.equal(initial,changed)
        changed.square().sum().backward()
        assert all(p.weight.grad is not None for p in model.latent.predictors.values())
        assert all(p.grad is None for p in codec.parameters())
    assert not hasattr(model.latent,"encoder")
    altered = copy.deepcopy(row)
    altered['response_mask'][a+2:] = [False]*len(altered['response_mask'][a+2:])
    normal,_ = forward_block(target,model,row,a)
    masked,_ = forward_block(target,model,altered,a)
    assert torch.equal(normal,masked)


def test_nested_codec_rates_pooling_and_no_fine_skip():
    codec = MultiscaleCodec(16,256)
    assert codec.rates()["standalone_scalars"] == {"4":1024,"8":2048,"15":3840}
    assert codec.rates()["all_levels_scalars"] == 6912
    states = torch.arange(15).float()[None,:,None].expand(1,15,16)
    valid = torch.tensor([[True]*3+[False]*12])
    pooled,mask = pool(states,4,valid)
    assert pooled[0,0,0] == 1 and mask.tolist() == [[True,False,False,False]]
    code = torch.randn(1,4,256)
    codec.encode = lambda *_: (_ for _ in ()).throw(AssertionError("encoder invoked"))
    assert codec.reconstruct(code,4).shape == (1,15,16)
    assert cache_identity("data",1,"codec1","policy") != cache_identity("data",1,"codec2","policy")


def test_method_conflicts_fixed_offsets_and_reduced_depth(pair):
    with pytest.raises(ValueError,match="contradictory"):
        resolve_method({"method":"HIER_SKIP","schedule":[15]*5})
    with pytest.raises(ValueError,match="unavailable"):
        resolve_method({"method":"STRONG_EXTERNAL"})
    target,backbone = pair
    reduced = build_drafter(target,copy.deepcopy(backbone),resolve_method({"method":"FLAT_COST",
        "layer_indices":[0,2,4],"cost_selection":"development-receipt"}))
    assert len(reduced.backbone.layers) == 3 and reduced.backbone.target_layer_ids == backbone.target_layer_ids
    fixed = build_drafter(target,backbone,resolve_method({"method":"FIXED_BROADCAST"}))
    row = fixture_rows(target,"train")[0]
    logits,_ = forward_block(target,fixed,row,row["prompt_length"])
    logits.sum().backward()
    torch.optim.AdamW([p for p in fixed.parameters() if p.requires_grad]).step()
    assert fixed.initial_span.requires_grad
    assert all(not e.offset.requires_grad and not e.offset.any() for e in fixed.expansions.values())


def test_interrupted_update_uses_last_committed_checkpoint(pair,tmp_path,monkeypatch):
    target,backbone = pair
    model = build_drafter(target,backbone,resolve_method({'method':'HIER_SKIP'}))
    rows = fixture_rows(target,'train')
    train(target,model,rows,meter(tmp_path/'first'),'a',steps=1)
    before = digest(tmp_path/'first/final.pt')
    original = torch.optim.AdamW.step
    def interrupted(self,*args,**kwargs):
        original(self,*args,**kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(torch.optim.AdamW,'step',interrupted)
    result = train(target,model,rows,meter(tmp_path/'second'),'a',steps=2,resume=tmp_path/'first/final.pt')
    assert result['status']=='cancelled' and not (tmp_path/'second/final.pt').exists()
    assert digest(tmp_path/'first/final.pt')==before


def test_actual_input_hash_and_nested_ladder_failures(tmp_path):
    from ncm_lab.data.artifacts import lock_input,verify_input
    from ncm_lab.data.ladder import resolve_ladder
    rows = [normalize_record({'input_ids':[1,4,5,6],'prompt_length':1,'original_id':str(i),
        'group_id':str(i),'split':'train','origin':'software_fixture'},origin='software_fixture') for i in range(4)]
    tiers = resolve_ladder(rows,[3,6,20])
    assert tiers[0]['admitted_positions']==3
    assert set(tiers[0]['original_ids']) <= set(tiers[1]['original_ids'])
    assert tiers[2]['status']=='unavailable'
    path,receipt=tmp_path/'data.jsonl',tmp_path/'input.json'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    lock_input(path,receipt,'teacher','train','fixture',fixture=True)
    verify_input(receipt,'train','teacher')
    with pytest.raises(ValueError,match='split/schema'):
        verify_input(receipt,'dev')
    path.write_text('changed')
    with pytest.raises(ValueError,match='data changed'):
        verify_input(receipt)


def test_real_development_selection_and_replacement_rejection(pair,tmp_path):
    from ncm_lab.training.selection import validate_selection
    target,backbone=pair
    model=build_drafter(target,backbone,resolve_method({'method':'FLAT_ADAPTED'}))
    records,dev=fixture_rows(target,'train'),fixture_rows(target,'dev')
    result=train(target,model,records,meter(tmp_path/'run',max_steps=2),'a',steps=2,dev_records=dev)
    assert result['status']=='completed'
    selection=validate_selection(tmp_path/'run/selection.json')
    assert len(selection['candidates'])==2
    (tmp_path/'run/selected.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='replaced'):
        validate_selection(tmp_path/'run/selection.json')


@pytest.mark.parametrize('kind,levels',[('linear',(4,8,15)),('independent',(4,8,15)),
    ('independent',(8,)),('channel',(15,)),('pooling',(4,8,15))])
def test_codec_controls_fit_select_and_reload(pair,tmp_path,kind,levels):
    from ncm_lab.training.codec import fit_codec,load_selected_codec
    target,_=pair
    training,dev=fixture_rows(target,'train'),fixture_rows(target,'dev')
    directory=tmp_path/'codec'
    selected=fit_codec(target,training,dev,meter(directory),[0.0],steps=1,width=4,kind=kind,levels=levels)
    codec=load_selected_codec(directory/'codec-selection.json')
    assert codec.spec['kind']==kind and codec.levels==levels
    assert not any(p.requires_grad for p in codec.parameters())
    assert selected['candidates'][0]['metrics'][str(levels[0])]['valid_positions']>0


def test_pinned_policy_metadata_is_an_asset_and_checks_before_weights():
    from types import SimpleNamespace
    from ncm_lab.models.upstream import validate_generation_artifact
    from ncm_lab.runtime.prepare import selected_assets
    assets=[a for a in selected_assets() if a['id']=='target:generation_config.json']
    assert len(assets)==1 and assets[0]['sha256']==digest(root()/'third_party/configs/qwen3-8b-generation.json')
    config=read(root()/'configs/models/qwen3_8b_dflash.yaml')
    with pytest.raises(ValueError,match='missing/changed'):
        validate_generation_artifact(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=151645)),config)


def test_analysis_pairs_known_difference_and_keeps_condition_missingness():
    from ncm_lab.evaluation.tables import analyze_records
    rows=[]
    for i in range(3):
        for method,seconds in [('flat',1),('hier',2)]:
            rows.append({'original_id':str(i),'group_id':str(i),'method':method,'training_seed':1,
                         'request_seconds':seconds,'correct':i!=0,'error':'failure' if i==0 else None,
                         'official_scores':{} if i==0 else {'exact_match':1.0}})
    result=analyze_records(rows)
    assert result['paired_latency_differences'][0]['mean']==1
    assert result['paired_latency_differences'][0]['ci95']==[1,1]
    assert all(r['failed_requests_in_denominator']==1 and r['decoder_missing_requests']==3 for r in result['per_seed'])


def test_source_identity_ignores_recreated_build_metadata(tmp_path,monkeypatch):
    import ncm_lab.runtime.provenance as provenance
    (tmp_path/'src').mkdir()
    source=tmp_path/'src/a.py'
    source.write_text('answer = 1\n')
    metadata=tmp_path/'src/ncm_lab.egg-info'
    metadata.mkdir()
    (metadata/'PKG-INFO').write_text('build metadata')
    monkeypatch.setattr(provenance,'root',lambda:tmp_path)
    before=provenance.implementation_hash()
    (metadata/'PKG-INFO').write_text('rebuilt metadata')
    assert provenance.implementation_hash()==before
    source.write_text('answer = 2\n')
    assert provenance.implementation_hash()!=before


def test_dcore_uses_distinct_source_and_matches_its_direct_flat_forward(pair):
    from ncm_lab.models.external_dcore import DCORE_ID,DCORE_REVISION,DCoreReferenceDraft
    target,original=pair
    backbone=DCoreReferenceDraft(copy.deepcopy(original.config)).eval()
    method=resolve_method({'method':'STRONG_EXTERNAL','model':{'draft_id':DCORE_ID,'draft_revision':DCORE_REVISION}})
    model=build_drafter(target,backbone,method).eval()
    prefix=8
    ids=torch.tensor([[1,4,8,4,8,7,6,5]])
    features=FeatureExtractor(target,backbone.target_layer_ids)(ids)
    positions=torch.arange(prefix)[None]
    actual=model(features,positions,torch.zeros_like(positions),torch.tensor([prefix]),torch.tensor([1]),torch.tensor([0]))
    query=torch.tensor([[1]+[backbone.mask_token_id]*15])
    mask=torch.ones(1,1,16,prefix+16,dtype=torch.bool)
    with torch.no_grad():
        hidden=backbone(position_ids=torch.arange(prefix+16)[None],attention_mask=mask,
                        noise_embedding=target.get_input_embeddings()(query),target_hidden=features)
        expected=target.get_output_embeddings()(hidden[:,1:])
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)
    assert model.stage_lengths==[16]*5 and not model.initial_span.requires_grad
    with pytest.raises(ValueError,match='audited DCore'):
        build_drafter(target,original,method)


def test_derived_fixture_receipt_cannot_become_research(tmp_path):
    from ncm_lab.data.artifacts import lock_input,verify_input
    path=tmp_path/'data.jsonl'
    path.write_text('{"original_id":"a","split":"train"}\n')
    parent=tmp_path/'parent.json'
    lock_input(path,parent,'teacher','train','fixture',fixture=True)
    child=tmp_path/'child.json'
    result=lock_input(path,child,'teacher','train','fixture',source_receipt=parent)
    assert result['fixture']
    result['fixture']=False
    child.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='fixture lineage'):
        verify_input(child)


def test_actual_teacher_codec_transfer_and_parity_executors(tmp_path):
    from ncm_lab.models.fixtures import save_tiny_artifacts
    from ncm_lab.data.artifacts import lock_input
    from ncm_lab.experiments.matrix import resolve_plan,launch_plan
    from ncm_lab.runtime.io import write_json
    from ncm_lab.reporting.evidence import attest
    import uuid
    model=save_tiny_artifacts(tmp_path/'models')
    budget=read(root()/'configs/budgets/acceptance.yaml')
    budget.update(phase_budget_id='executor-contract-'+uuid.uuid4().hex[:8],max_steps=16)
    budget_path=tmp_path/'budget.json'
    write_json(budget_path,budget)
    common={'profile':'cpu','device':'cpu','precision':'float32','result_origin':'test_fixture',
            'model':model,'artifact_root':str(tmp_path/'artifacts'),'cpu_threads':1,'seed':9,'data_seed':4,'init_seed':9}
    jobs=[]
    for split in ('train','dev'):
        path=tmp_path/(split+'.jsonl')
        path.write_text(''.join(json.dumps({'question':split+str(i),'original_id':split+str(i),
            'group_id':split+str(i),'split':split})+'\n' for i in range(2)))
        receipt=tmp_path/(split+'.input.json')
        lock_input(path,receipt,'prompts',split,'software_fixture',fixture=True)
        config={**common,'experiment_id':'SHR-001','executor':'teacher','purpose':split,'prompts_path':str(path),
            'input_receipts':[str(receipt)],'max_examples':2,'max_new_tokens':12,
            'teacher_output_reference':str(tmp_path/(split+'.reference.json'))}
        jobs.append({'id':split,'config':config,'budget':str(budget_path),'depends_on':[]})
    data={'training_data':'@train/teacher.jsonl','development_data':'@dev/teacher.jsonl',
          'input_receipts':['@train/teacher.input.json','@dev/teacher.input.json']}
    jobs.append({'id':'codec','config':{**common,**data,'experiment_id':'ACL-005','executor':'codec',
        'steps':2,'prediction_weights':[0.0,.1],'latent_width':4,'codec_seed':3},
        'budget':str(budget_path),'depends_on':['train','dev']})
    cache_budget=tmp_path/'cache-budget.json'
    write_json(cache_budget,{**budget,'max_steps':32})
    jobs.append({'id':'cache','config':{**common,'experiment_id':'ACL-005','executor':'latent_cache',
        'training_data':'@train/teacher.jsonl','input_receipts':['@train/teacher.input.json'],
        'codec_selection':'@codec/codec-selection.json','codec_sha256':'@codec/codec-selected.pt#sha256',
        'cache_anchors_per_sequence':16},'budget':str(cache_budget),'depends_on':['codec']})
    transfer={**common,**data,'experiment_id':'ACL-005','executor':'train','method':'HIER_MULTI_AE',
              'codec_selection':'@codec/codec-selection.json','codec_sha256':'@codec/codec-selected.pt#sha256','steps':2,
              'latent_cache':'@cache/latent-cache.json'}
    jobs.append({'id':'transfer','config':transfer,'budget':str(budget_path),'depends_on':['cache']})
    evaluation=tmp_path/'gsm.jsonl'
    evaluation.write_text(json.dumps({'question':'train0','answer':'#### 1'})+'\n')
    receipt=tmp_path/'gsm.input.json'
    lock_input(evaluation,receipt,'gsm8k','train','fixture',fixture=True)
    parity={**common,'experiment_id':'SHR-002','executor':'verification','method':'HIER_MULTI_AE',
        'codec_selection':'@codec/codec-selection.json','codec_sha256':'@codec/codec-selected.pt#sha256',
        'checkpoint':'@transfer/selected.pt','checkpoint_sha256':'@transfer/selected.pt#sha256',
        'dataset':'gsm8k','split':'train','data_path':str(evaluation),'input_receipts':[str(receipt)],
        'max_requests':1,'max_new_tokens':12}
    jobs.append({'id':'parity','config':parity,'budget':str(budget_path),'depends_on':['transfer']})
    jobs.append({'id':'latent-diagnostics','config':{**common,'experiment_id':'ACL-005',
        'executor':'latent_diagnostics','method':'HIER_MULTI_AE','dataset':'oasst1','split':'dev','max_requests':2,
        'evaluation_data':'@dev/teacher.jsonl','input_receipts':['@dev/teacher.input.json'],
        'codec_selection':'@codec/codec-selection.json','codec_sha256':'@codec/codec-selected.pt#sha256',
        'checkpoint':'@transfer/selected.pt','checkpoint_sha256':'@transfer/selected.pt#sha256',
        'checkpoint_selection':'@transfer/selection.json'},'budget':str(budget_path),'depends_on':['transfer']})
    jobs.append({'id':'evaluation','config':{**parity,'executor':'evaluate',
        'checkpoint_selection':'@transfer/selection.json','output_instruction':'Finish with #### and the number.',
        'warmup_requests':1},'budget':str(budget_path),'depends_on':['parity']})
    jobs.append({'id':'cost-profile','config':{**common,'experiment_id':'ACL-001','executor':'profile',
        'methods':['FLAT_COST','HIER_SKIP'],'cost_candidate':True,'method_overrides':{'FLAT_COST':{'layer_indices':[0,2,4]}},
        'prefix_lengths':[4],'concurrency':[1],'repetitions':1},'budget':str(budget_path),'depends_on':[]})
    jobs.append({'id':'cost-selection','config':{**common,'experiment_id':'ACL-001','executor':'cost_selection',
        'profile_runs':['@cost-profile'],'tolerance':1.0},'budget':str(budget_path),'depends_on':['cost-profile']})
    spec=tmp_path/'spec.json'
    write_json(spec,{'jobs':jobs,'aggregate_limits':{'steps':176,'wall_seconds':1200,'source_positions':200000,
        'storage_bytes':640000000,'download_bytes':0},'timing_seed':4})
    plan=tmp_path/'plan.json'
    resolve_plan(spec,plan)
    results=launch_plan(plan)['results']
    assert all(r['status']=='completed' for r in results.values()),results
    assert read(Path(results['parity']['path'])/'metrics.json')['all_equal']
    trained=read(Path(results['transfer']['path'])/'metrics.json')
    assert trained['latent_cache_hits']==2 and trained['new_label_target_forwards']==0
    assert read(Path(results['latent-diagnostics']['path'])/'metrics.json')['diagnostic_requests']==2
    assert read(Path(results['evaluation']['path'])/'metrics.json')['requests']==1
    assert read(Path(results['cost-selection']['path'])/'cost-selection.json')['layer_indices']==[0,2,4]
    assert not attest(results['codec']['path'],'representation')['eligible']


def test_complete_production_matrix_export_outside_checkout(tmp_path):
    from ncm_lab.experiments.matrix import resolve_plan, launch_plan
    from ncm_lab.reporting.portable import engineering_export, validate_export
    from ncm_lab.reporting.evidence import attest
    plan = tmp_path/"resolved.json"
    resolve_plan(root()/"experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/pilot.yaml",plan)
    launched = launch_plan(plan)
    result = launched["results"]["cpu-acceptance"]
    assert result["status"] == "completed",result
    path = Path(result["path"])
    metrics=read(path/"metrics.json")
    assert metrics["codec_candidates"] == 4
    assert metrics['optimizer_steps']==13 and metrics['total_steps']==20
    assert metrics['resources']['optimizer_steps']==13
    assert metrics['resources']['teacher_generated_tokens']==48
    assert metrics['resources']['target_processed_positions']==metrics['target_processed_positions']
    assert not attest(path)["eligible"]
    archive = tmp_path/"engineering.zip"
    engineering_export(path,archive)
    assert validate_export(archive)["valid"]
    outside = tmp_path/"outside"
    with zipfile.ZipFile(archive) as zipfile_:
        zipfile_.extractall(outside)
    process = subprocess.run([sys.executable,'-m','ncm_lab.cli','publication','validate-export','--path',str(archive),'--reanalyze'],
        cwd=outside,capture_output=True,text=True,timeout=30)
    assert process.returncode == 0,process.stderr
    validation=json.loads(process.stdout)
    assert not validation["research_eligible"] and validation['archive_code_executed'] is False
    assert validation['analysis']['per_seed']


def test_codec_exact_resume_across_search_candidates(pair,tmp_path):
    from ncm_lab.training.codec import fit_codec
    from ncm_lab.runtime.budget import BudgetExceeded
    target,_=pair
    rows,dev=fixture_rows(target,'train'),fixture_rows(target,'dev')
    settings={'candidates':[0,.1],'steps':2,'seed':19,'width':4}
    full=fit_codec(target,rows,dev,meter(tmp_path/'full'),**settings)
    with pytest.raises(BudgetExceeded):
        fit_codec(target,rows,dev,meter(tmp_path/'partial',max_steps=1),**settings)
    resumed=fit_codec(target,rows,dev,meter(tmp_path/'resumed'),
        resume=tmp_path/'partial/codec-latest.pt',**settings)
    assert full['search_cost']['target_processed_positions']==resumed['search_cost']['target_processed_positions']
    assert full['search_cost']['optimizer_steps']==resumed['search_cost']['optimizer_steps']==4
    assert full['latent_scales']==resumed['latent_scales']
    for name in ('codec-latest.pt','codec-selected.pt'):
        a,b=(read_checkpoint(tmp_path/part/name) for part in ('full','resumed'))
        assert all(torch.equal(value,b['model'][key]) for key,value in a['model'].items())
    assert a['candidate']==b['candidate'] and a['step']==b['step']
    assert [(r['candidate'],r['step'],r['criterion']) for r in full['candidates']]==[
        (r['candidate'],r['step'],r['criterion']) for r in resumed['candidates']]


def test_source_split_cannot_be_relabeled_and_holdout_is_recomputed(tmp_path):
    from ncm_lab.data.artifacts import lock_input
    source=tmp_path/'source.json'
    source.write_text(json.dumps({'official_split':'test'}))
    path=tmp_path/'data.jsonl'
    path.write_text('{"original_id":"a","group_id":"g","question":"q","split":"train"}\n')
    with pytest.raises(ValueError,match='cannot be relabeled'):
        lock_input(path,tmp_path/'invalid.json','prompts','train','oasst1',source)
    source.write_text(json.dumps({'official_split':'train','derived_split_policy':{
        'version':'oasst_tree_hash_holdout_v1','dev_fraction':1.0}}))
    with pytest.raises(ValueError,match='assignment mismatch'):
        lock_input(path,tmp_path/'invalid.json','prompts','train','oasst1',source)
    path.write_text('{"original_id":"a","group_id":"g","question":"q","split":"dev"}\n')
    assert lock_input(path,tmp_path/'valid.json','prompts','dev','oasst1',source)['official_split']=='train'


def test_gold_free_overlap_audit_and_content_invalidation(tmp_path):
    from types import SimpleNamespace
    from ncm_lab.data.artifacts import lock_input
    from ncm_lab.data.overlap import audit,validate_audit
    text='there are exactly seven red apples in the basket'
    training=tmp_path/'train.jsonl'
    training.write_text(json.dumps({'original_id':'source','group_id':'tree','split':'train','question':text})+'\n')
    receipt=tmp_path/'training.input.json'
    lock_input(training,receipt,'prompts','train','fixture',fixture=True)
    benchmark=tmp_path/'benchmark.jsonl'
    benchmark.write_text(json.dumps({'question':text,'answer':'PRIVATE GOLD MUST NEVER APPEAR'})+'\n')
    bench_receipt=tmp_path/'bench.input.json'
    lock_input(benchmark,bench_receipt,'gsm8k','test','fixture',fixture=True)
    config={'training_prompt_receipt':str(receipt),'benchmark_receipts':[str(bench_receipt)]}
    run=SimpleNamespace(path=tmp_path/'audit')
    result=audit(run,config,meter(run.path))
    assert result['status']=='failed' and result['comparisons']==1
    assert 'PRIVATE GOLD' not in (run.path/'overlap_audit.json').read_text()
    with pytest.raises(ValueError,match='failed overlap'):
        validate_audit(run.path/'overlap_audit.json')


def test_policy_rejects_eos_suppression_and_guard_keeps_partial_tokens(pair):
    from ncm_lab.runtime.budget import BudgetExceeded
    target,_=pair
    target.generation_config.eos_token_id=[2,31]
    with pytest.raises(ValueError,match='suppress'):
        resolve_policy(target,overrides={'eos_ids':[2]})
    with pytest.raises(ValueError,match='precision'):
        resolve_policy(target,overrides={'precision':'bfloat16'})
    calls=0
    def guard():
        nonlocal calls
        calls+=1
        if calls==2:
            raise BudgetExceeded('stop at the second target call')
    with pytest.raises(BudgetExceeded) as caught:
        ar_greedy(target,torch.tensor([[1,4]]),GenerationPolicy(5),guard)
    assert len(caught.value.partial_tokens)==1


def test_matrix_preserves_phase_failure_and_continues_independent_jobs(tmp_path,monkeypatch):
    import ncm_lab.experiments.runner as runner
    from ncm_lab.experiments.matrix import resolve_plan,launch_plan
    spec=read(root()/'experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/pilot.yaml')
    original=copy.deepcopy(spec['jobs'][0])
    original['id']='first'
    second=copy.deepcopy(original)
    second['id']='second'
    spec['jobs']=[original,second]
    spec['aggregate_limits']={k:2*v for k,v in spec['aggregate_limits'].items()}
    path=tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    plan=tmp_path/'plan.json'
    resolve_plan(path,plan)
    calls=0
    def fail_then_complete(*args):
        nonlocal calls
        calls+=1
        if calls==1:
            raise ValueError('aggregate phase budget exhausted')
        return {'status':'completed'}
    monkeypatch.setattr(runner,'run',fail_then_complete)
    results=launch_plan(plan)['results']
    assert sorted(r['status'] for r in results.values())==['completed','failed']


def test_plan_rejects_changed_consumed_file_before_launch(tmp_path):
    from ncm_lab.experiments.matrix import resolve_plan,launch_plan
    spec=read(root()/'experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/pilot.yaml')
    data=tmp_path/'input.jsonl'
    data.write_text('original input')
    spec['jobs'][0]['config']=read(root()/'configs/acceptance.yaml')
    spec['jobs'][0]['config']['data_path']=str(data)
    source,plan=tmp_path/'spec.json',tmp_path/'plan.json'
    source.write_text(json.dumps(spec))
    resolve_plan(source,plan)
    data.write_text('changed after plan')
    with pytest.raises(ValueError,match='input dependency changed'):
        launch_plan(plan)


def test_preparation_wires_distinct_optional_external_checkpoint(tmp_path):
    from ncm_lab.runtime.prepare import selected_assets,write_launch_configs
    from ncm_lab.models.external_dcore import DCORE_ID,DCORE_REVISION
    attempt=tmp_path/'attempt'
    attempt.mkdir()
    assets=selected_assets(include_external_baseline=True)
    prompts={'outputs':{split:{'path':str(tmp_path/(split+'.jsonl'))} for split in ('train','dev')}}
    write_launch_configs(attempt,tmp_path/'assets',assets,prompts,'cpu',None)
    external=read(attempt/'launch/external_verification.yaml')
    ordinary=read(attempt/'launch/verification.yaml')
    pilot=read(attempt/'launch/pilot.yaml')
    assert 'training_data' not in pilot and 'development_data' not in pilot and 'input_receipts' not in pilot
    assert 'overlap_audit' not in pilot and pilot['development_data_reference'].endswith('teacher_dev_output.json')
    assert ordinary['model']['draft_id']!=DCORE_ID
    assert external['model']['draft_id']==DCORE_ID and external['model']['draft_revision']==DCORE_REVISION
    assert resolve_method(external).name=='STRONG_EXTERNAL'
    matrix=read(attempt/'launch/codec_pilot_matrix.yaml')
    assert len(matrix['jobs'])==7 and matrix['jobs'][-1]['id']=='external-parity'
