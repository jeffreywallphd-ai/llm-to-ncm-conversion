"""Bounded production-interface integration. All records are software fixtures."""
from dataclasses import asdict, replace
import copy
import json
import time
import torch
from ncm_lab.models.fixtures import tiny_pair
from ncm_lab.models.factory import resolve_method, build_drafter
from ncm_lab.decoding.policy import GenerationPolicy, ar_greedy
from ncm_lab.decoding.hf import CachedProposer, cached_greedy
from ncm_lab.training.records import normalize_record
from ncm_lab.training.codec import fit_codec, load_selected_codec
from ncm_lab.training.loop import train
from ncm_lab.training.checkpoints import read_checkpoint, load_student
from ncm_lab.runtime.budget import Meter
from ncm_lab.runtime.io import write_json, canonical_hash, digest
from ncm_lab.evaluation.latent_mechanisms import diagnose


def fixture_rows(target, split, meter=None):
    policy = GenerationPolicy(12, (30, 31))
    rows = []
    for index in range(2):
        prompt = [1, 4 + index + (0 if split == "train" else 4), 8]
        if meter:
            meter.check(len(prompt)+policy.max_new_tokens-1,0)
        completion = ar_greedy(target, torch.tensor([prompt]), policy)
        processed=len(prompt)+max(0,len(completion)-1)
        if meter:
            meter.record(processed,0)
        rows.append(normalize_record({"input_ids": prompt + completion, "prompt_length": len(prompt),
            "generated_tokens": len(completion), "original_id": f"{split}-{index}", "group_id": f"{split}-{index}",
            "split": split, "origin": "software_fixture", "generation_policy": asdict(policy),
            "policy_hash": policy.digest, "teacher_revision": "tiny-random-fixture"}, split, "software_fixture"))
        rows[-1]['teacher_processed_positions']=processed
    return rows


def full_acceptance(run, config, meter):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target, backbone = tiny_pair()
    training, dev = fixture_rows(target, "train",meter), fixture_rows(target, "dev",meter)
    inputs=[]
    for name, rows in (("train", training), ("dev", dev)):
        path = run.path/(name+".jsonl")
        path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        from ncm_lab.data.artifacts import lock_input
        receipt=run.path/(name+".input.json")
        identity=lock_input(path, receipt, "teacher", name, "software_fixture", fixture=True)
        inputs.append({'receipt':str(receipt),'identity':identity})
    write_json(run.path/'inputs.json',inputs)
    stages = []
    def check_aggregate(extra_positions=0,extra_operations=0):
        from ncm_lab.runtime.budget import BudgetExceeded
        meter.check(step=False)
        if (meter.steps+sum(s.new_steps for s in stages)+extra_operations>meter.budget.max_steps
            or meter.source_tokens+sum(s.new_source_tokens for s in stages)+extra_positions>meter.budget.max_training_source_token_passes):
            raise BudgetExceeded('finite aggregate acceptance budget reached')
    def stage(name):
        check_aggregate()
        path = run.path/name
        path.mkdir()
        remaining_steps = meter.budget.max_steps-meter.steps-sum(s.new_steps for s in stages)
        remaining_tokens = meter.budget.max_training_source_token_passes-meter.source_tokens-sum(s.new_source_tokens for s in stages)
        remaining_positions = meter.budget.max_supervised_future_positions-meter.supervised_positions-sum(s.new_supervised_positions for s in stages)
        remaining_wall = meter.budget.max_wall_seconds - (time.monotonic() - meter.started)
        child = Meter(replace(meter.budget, max_steps=remaining_steps, max_wall_seconds=remaining_wall,
            max_training_source_token_passes=remaining_tokens, max_supervised_future_positions=remaining_positions), path)
        child.accounting_dir = run.path
        stages.append(child)
        return child
    codec_meter = stage("codec")
    selection = fit_codec(target, training, dev, codec_meter, [0.0, .1], steps=2, seed=3, width=4,
                          dependencies={"target": "tiny-random-fixture"})
    codec_path = codec_meter.artifact_dir/"codec-selection.json"
    codec = load_selected_codec(codec_path)
    method_config = {"method": "HIER_MULTI_AE", "codec_sha256": selection["weights_sha256"], "seed": 9}
    drafter = build_drafter(target, copy.deepcopy(backbone), resolve_method(method_config), codec)
    initial = stage("transfer-initial")
    first = train(target, drafter, training, initial, canonical_hash(training), steps=1, seed=4,
                  dev_records=dev, codec=codec)
    resumed = stage("transfer-resumed")
    second = train(target, drafter, training, resumed, canonical_hash(training), steps=2, seed=4,
                   resume=initial.artifact_dir/"final.pt", dev_records=dev, codec=codec)
    load_student(drafter, read_checkpoint(resumed.artifact_dir/"selected.pt")["model"])
    controls = {}
    for method in config.get("controls", ["FLAT_ADAPTED", "HIER_NO_SKIP", "HIER_LATENT_CE_ONLY",
            "HIER_CODEC_CE_ONLY", "HIER_MULTI_AE_AUX_ONLY", "FLAT_MULTI_AE", "HIER_MULTI_AE_NO_SKIP"]):
        resolved = resolve_method({**method_config, "method": method})
        control = build_drafter(target, copy.deepcopy(backbone), resolved,
                               None if method == "HIER_LATENT_CE_ONLY" else codec, latent_width=4)
        control_meter = stage(method)
        controls[method] = train(target, control, training, control_meter, canonical_hash(training), steps=1,
            seed=4, dev_records=dev, codec=codec if resolved.latent and method != "HIER_LATENT_CE_ONLY" else None,
            latent_weight=0 if method == "HIER_CODEC_CE_ONLY" else 1.0)
    # A failing encoder sentinel establishes that timed generation cannot use future-state labels.
    def forbidden(*args, **kwargs):
        raise AssertionError("future-state encoder reached deployed generation")
    codec.encode = forbidden
    policy = GenerationPolicy(20, (30, 31))
    observations = []
    for row in dev:
        prompt = torch.tensor([row["input_ids"][:row["prompt_length"]]])
        check_aggregate(2*prompt.numel()+17*policy.max_new_tokens,1)
        expected = ar_greedy(target, prompt, policy)
        actual = cached_greedy(target, CachedProposer(drafter), prompt, policy=policy)
        meter.record(actual.telemetry['target_processed_positions']+prompt.numel()+max(0,len(expected)-1),0)
        if actual.tokens != expected:
            raise ValueError("tiny production generation parity failed")
        observations.append({"original_id": row["original_id"], "group_id": row["group_id"],
            "method": method_config["method"], "training_seed": 9, "correct": True,
            "request_seconds": actual.telemetry['generation_seconds'],'generation_seconds':actual.telemetry['generation_seconds'],
            "output_tokens": len(actual.tokens), "decoder": asdict(actual), "policy_hash": policy.digest})
    (run.path/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in observations), encoding="utf-8")
    codec = load_selected_codec(codec_path)
    check_aggregate((2+2*len(drafter.latent.levels)+int(drafter.skip_enabled))*len(dev[0]['input_ids']),1)
    diagnostics = diagnose(target, drafter, codec, dev[0], dev[0]["prompt_length"])
    meter.record(diagnostics['target_processed_positions'],0)
    write_json(run.path/"latent-diagnostics.json", diagnostics, exclusive=True)
    total_steps = meter.steps+sum(m.new_steps for m in stages)
    optimizer_steps=selection['search_cost']['optimizer_steps']+first['new_invocation_optimizer_steps']+second['new_invocation_optimizer_steps']+sum(c['new_invocation_optimizer_steps'] for c in controls.values())
    total_positions = meter.source_tokens+sum(m.new_source_tokens for m in stages)
    if total_steps > meter.budget.max_steps or total_positions > meter.budget.max_training_source_token_passes:
        raise ValueError("aggregate acceptance budget exceeded")
    return {"status": "completed", "evidence_class": "software_test_fixture", "research_evidence": "none",
        "all_equal": True, "policy_hash": policy.digest, "codec_candidates": len(selection["candidates"]),
        "initial_steps": first["optimizer_steps"], "resumed_steps": second["optimizer_steps"], "controls": controls,
        "total_steps": total_steps, "target_processed_positions": total_positions,
        "optimizer_steps":optimizer_steps,"teacher_generated_tokens":sum(r['generated_tokens'] for r in training+dev),
        "codec_selection_sha256": digest(codec_path), "prediction_requests": len(observations),
        "stage_lengths": drafter.stage_lengths, "rates": codec.rates()}
