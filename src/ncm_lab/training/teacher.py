import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ncm_lab.runtime.budget import BudgetExceeded


def generate_teacher(prompts, model_config, output, meter, max_new_tokens=128, device="cpu", precision="float32"):
    """Bounded cache-only target continuation generation; no benchmark gold input."""
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("teacher output cap must be 1..4096")
    model_id, revision = model_config.get("target_path",model_config["target_id"]), model_config["target_revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=model_config["tokenizer_revision"],
                                              local_files_only=True,trust_remote_code=False)
    target = AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=getattr(torch,precision),
                                                 local_files_only=True,trust_remote_code=False).to(device).eval()
    target.requires_grad_(False)
    output.parent.mkdir(parents=True,exist_ok=True)
    counters = {"generated_teacher_tokens":0,"prompt_tokens":0,"records":0,"status":"completed"}
    started = time.monotonic()
    with output.open("x",encoding="utf-8") as stream:
        try:
            for row in prompts:
                allowed = {"question","group_id","original_id","split"}
                if not set(row) <= allowed or row.get("split") != "train" or not row.get("group_id"):
                    raise ValueError("teacher inputs must be allowlisted train prompts with whole-group IDs; no gold fields")
                text = tokenizer.apply_chat_template([{"role":"user","content":row["question"]}],
                                                      tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids = tokenizer(text,return_tensors="pt").input_ids.to(device)
                meter.check(ids.numel()+max_new_tokens,0)
                with torch.inference_mode():
                    sequence = target.generate(ids,max_new_tokens=max_new_tokens,do_sample=False)[0]
                generated = sequence.numel()-ids.numel()
                item = {"input_ids":sequence.tolist(),"group_id":row["group_id"],"original_id":row.get("original_id"),
                        "split":"train","origin":"target_distillation","generated_tokens":generated,
                        "teacher_revision":revision,"thinking":False}
                stream.write(json.dumps(item)+"\n")
                stream.flush()
                meter.record(sequence.numel(),0)
                counters["records"] += 1
                counters["prompt_tokens"] += ids.numel()
                counters["generated_teacher_tokens"] += generated
        except BudgetExceeded as exc:
            counters.update(status="cancelled",reason=str(exc))
    return {**counters,"elapsed_seconds":time.monotonic()-started,"path":str(output),
            "origin":"target_distillation; never financial gold"}
