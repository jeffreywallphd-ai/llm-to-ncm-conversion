import json
import time
import torch
from dataclasses import asdict
from ncm_lab.decoding.policy import resolve_policy, ar_greedy
from transformers import AutoModelForCausalLM, AutoTokenizer
from ncm_lab.runtime.budget import BudgetExceeded


def generate_teacher(prompts, model_config, output, meter, max_new_tokens=128, device="cpu", precision="float32",
                     purpose="train",policy_overrides=None):
    """Bounded cache-only target continuation generation; no benchmark gold input."""
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("teacher output cap must be 1..4096")
    model_id, revision = model_config.get("target_path",model_config["target_id"]), model_config["target_revision"]
    from ncm_lab.models.upstream import preflight_generation_metadata
    preflight_generation_metadata(model_config,policy_overrides)
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=model_config["tokenizer_revision"],
                                              local_files_only=True,trust_remote_code=False)
    target = AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=getattr(torch,precision),
                                                 local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to(device).eval()
    target.requires_grad_(False)
    from ncm_lab.models.upstream import validate_generation_artifact
    validate_generation_artifact(target,model_config)
    if purpose not in {"train", "dev","test_evaluation"}:
        raise ValueError("teacher generation purpose must be train, dev, or test_evaluation")
    split='test' if purpose=='test_evaluation' else purpose
    policy = resolve_policy(target, tokenizer, {"max_new_tokens": max_new_tokens,
        "tokenizer_revision": model_config["tokenizer_revision"], "precision": precision,**(policy_overrides or {})})
    max_new_tokens=policy.max_new_tokens
    if not 1<=max_new_tokens<=4096:
        raise ValueError('effective teacher output cap must be 1..4096')
    output.parent.mkdir(parents=True,exist_ok=True)
    counters = {"generated_teacher_tokens":0,"prompt_tokens":0,"records":0,"status":"completed"}
    started = time.monotonic()
    with output.open("x",encoding="utf-8") as stream:
        try:
            for row in prompts:
                allowed = {"question","group_id","original_id","split"}
                if not set(row) <= allowed or row.get("split") != split or not row.get("group_id"):
                    raise ValueError("teacher inputs must match declared train/dev purpose with whole-group IDs; no gold")
                text = tokenizer.apply_chat_template([{"role":"user","content":row["question"]}],
                                                      tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids = tokenizer(text,return_tensors="pt").input_ids.to(device)
                meter.check(ids.numel()+max_new_tokens,0)
                with torch.inference_mode():
                    completion = ar_greedy(target, ids, policy,guard=lambda:meter.check(step=False))
                    sequence = torch.cat([ids[0], torch.tensor(completion, device=device)])
                generated = sequence.numel()-ids.numel()
                item = {"input_ids":sequence.tolist(),"group_id":row["group_id"],"original_id":row.get("original_id"),
                        "split":split,"origin":"target_distillation","generated_tokens":generated,
                        'use':'evaluation_only' if split=='test' else purpose,
                        "prompt_length":ids.numel(), "response_mask":[False]*ids.numel()+[True]*generated,
                        "generation_policy":asdict(policy), "policy_hash":policy.digest,
                        "tokenizer_revision":model_config["tokenizer_revision"],
                        "teacher_revision":revision,"thinking":False}
                stream.write(json.dumps(item)+"\n")
                stream.flush()
                meter.record(ids.numel()+max(0,generated-1),0)
                counters["records"] += 1
                counters["prompt_tokens"] += ids.numel()
                counters["generated_teacher_tokens"] += generated
        except BudgetExceeded as exc:
            counters.update(status="cancelled",reason=str(exc))
            partial=getattr(exc,'partial_tokens',[])
            if partial:
                meter.record(ids.numel()+len(partial)-1,0)
            counters['partial_generated_teacher_tokens']=len(partial)
            counters['generated_teacher_tokens']+=len(partial)
            from ncm_lab.runtime.io import write_json
            write_json(output.with_suffix('.partial.json'),{'complete':False,'use':'cancelled_request_diagnostic',
                'partial_tokens':partial,'reason':str(exc)},exclusive=True)
    return {**counters,"elapsed_seconds":time.monotonic()-started,"path":str(output),
            "origin":"target_distillation; never financial gold"}
