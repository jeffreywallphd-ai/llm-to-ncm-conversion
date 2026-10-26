from dataclasses import asdict
import copy
import json
import random
from pathlib import Path
import time
import torch
from ncm_lab.contracts import PredictionRecord
from ncm_lab.data.loaders import load_records
from ncm_lab.decoding.reference import autoregressive, speculative
from ncm_lab.decoding.hf import cached_greedy
from ncm_lab.evaluation.analysis import summarize
from ncm_lab.models.fixtures import tiny_pair
from ncm_lab.models.hierarchy import ProgressiveDrafter
from ncm_lab.models.upstream import FeatureExtractor, load_pair
from ncm_lab.runtime.io import canonical_hash, root, write_json
from ncm_lab.training.loop import train


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def fixture_workflow(run, config, meter):
    torch.set_num_threads(1)
    torch.manual_seed(config.get("seed", 0))
    target, backbone = tiny_pair()
    drafter = ProgressiveDrafter(backbone, target)
    records = [{"input_ids": [1] + [4 + (i % 27) for i in range(35)], "group_id": "software-fixture",
                "origin": "software_fixture", "split": "train"}]
    data_hash = canonical_hash(records)
    first = train(target, drafter, records, meter, data_hash, steps=1)
    # Resume checks optimizer and RNG restoration and adds one complete optimizer step.
    second = train(target, drafter, records, meter, data_hash, steps=2, resume=run.path / "final.pt")
    if first["optimizer_steps"] != 1 or second["optimizer_steps"] != 2:
        raise ValueError("fixture save/resume failed to complete two bounded steps")
    def probs(prefix):
        return [.1, .8, .1] if prefix[-1] != 1 else [.7, .2, .1]
    expected = autoregressive(probs, [0], 12)
    started = time.monotonic()
    actual = speculative(probs, lambda prefix: [[.8, .1, .1]] * 3, [0], 12)
    prediction = PredictionRecord("software-fixture:0", "software-fixture", str(actual.tokens),
                                  actual.tokens == expected, time.monotonic() - started,
                                  input_tokens=1, output_tokens=len(actual.tokens), accepted_tokens=actual.accepted,
                                  correction_tokens=actual.corrections, bonus_tokens=actual.bonus,
                                  verification_calls=actual.verification_calls)
    (run.path / "predictions.jsonl").write_text(json.dumps(asdict(prediction)) + "\n", encoding="utf-8")
    real_data_check = None
    sample = root() / "tests/fixtures/finqa_train_sample.json"
    if sample.exists():
        items = list(load_records(sample, "finqa", "train"))
        from ncm_lab.evaluation.financial import finqa_score
        real_data_check = {"records": len(items), "official_gold_self_checks":
                           [finqa_score(gold.program, gold, item.table) for item, gold in items],
                           "meaning": "released training gold consistency, not model accuracy"}
    return {"status": second["status"], "training": second, "stage_lengths": drafter.stage_lengths,
            "prediction_summary": summarize([asdict(prediction)]), "real_training_sample_check": real_data_check,
            "evidence_class": "software_test_fixture", "research_findings": []}


def profile(run, config, meter):
    fixture = config.get("result_origin") == "test_fixture"
    device = config.get("device", "cpu")
    if fixture:
        target, base = tiny_pair()
        target, base = target.to(device), base.to(device)
    else:
        target, base, _ = load_pair(config["model"], device, getattr(torch, config["precision"]))
    torch.set_num_threads(config.get("cpu_threads", 1))
    shape_records = []
    rng = random.Random(config.get("seed",0))
    for batch in config.get("concurrency", [1, 4]):
        for prefix in config.get("prefix_lengths", [8, 32]):
            methods = [("FLAT_ADAPTED", [15] * 5, False),("HIER_SKIP", [4, 4, 8, 15, 15], True),
                       ("HIER_NO_SKIP", [4, 4, 8, 15, 15], False)]
            rng.shuffle(methods)
            for method_order,(method, schedule, skip) in enumerate(methods):
                drafter = ProgressiveDrafter(copy.deepcopy(base), target, schedule, skip).to(device).eval()
                features = torch.randn(batch, prefix, len(base.target_layer_ids) * base.config.hidden_size,
                                       device=device, dtype=next(base.parameters()).dtype)
                positions = torch.arange(prefix, device=device)[None].expand(batch, -1)
                args = (features, positions, torch.zeros_like(positions),
                        torch.full((batch,), prefix, device=device),
                        torch.ones(batch, dtype=torch.long, device=device),
                        torch.zeros(batch, dtype=torch.long, device=device))
                for repetition in range(config.get("repetitions", 3) + 1):
                    meter.check()
                    synchronize(device)
                    start = time.monotonic()
                    with torch.inference_mode():
                        logits = drafter(*args)
                    synchronize(device)
                    seconds = time.monotonic() - start
                    meter.record(0, 0)
                    if repetition:
                        shape_records.append({"method": method, "batch": batch, "prefix_length": prefix,
                                              "repetition": repetition, "seconds": seconds,
                                              "method_order_in_condition":method_order,"timing_order_seed":config.get("seed",0),
                                              "stage_lengths": drafter.stage_lengths,
                                              "future_logits_shape": list(logits.shape),
                                              "scope": "draft forward including expansion, skip and head; target verification excluded"})
    write_json(run.path / "component_timings.json", shape_records)
    return {"status": "completed", "timing_observations": len(shape_records), "result_origin": "artificial_tensor_microbenchmark",
            "performance_claim_eligible": False, "limitation": "component diagnostics; no proposal quality or cycle speed conclusion"}


def build_proposer(target, drafter):
    extractor = FeatureExtractor(target, drafter.backbone.target_layer_ids)
    @torch.no_grad()
    def propose(confirmed, anchor):
        features = extractor(confirmed)
        positions = torch.arange(confirmed.shape[1], device=confirmed.device)[None]
        logits = drafter(features, positions, torch.zeros_like(positions),
                         torch.tensor([confirmed.shape[1]], device=confirmed.device), anchor[:, 0],
                         torch.zeros(1, dtype=torch.long, device=confirmed.device))
        return logits.argmax(-1)
    return propose


def train_real(run, config, meter):
    from ncm_lab.training.loop import load_training, resolve_training_data
    from ncm_lab.runtime.io import digest
    target, backbone, _ = load_pair(config["model"], config["device"], getattr(torch, config["precision"]))
    drafter = ProgressiveDrafter(backbone, target, config["schedule"], config.get("skip", True))
    data = resolve_training_data(config)
    records = load_training(data)
    resume = None
    if config.get("resume_checkpoint"):
        resume = Path(config["resume_checkpoint"])
        if digest(resume) != config.get("resume_sha256"):
            raise ValueError("resume requires the expected trusted checkpoint SHA256")
    return train(target, drafter, records, meter, digest(data), steps=meter.budget.max_steps,
                 seed=config["seed"], resume=resume, learning_rate=config.get("learning_rate", 1e-5))


def teacher(run, config, meter):
    from ncm_lab.training.teacher import generate_teacher
    path = root() / config["prompts_path"]
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "max_examples" in config:
        cap = config["max_examples"]
        if isinstance(cap,bool) or not isinstance(cap,int) or not 1 <= cap <= meter.budget.max_steps:
            raise ValueError("max_examples must be positive and fit within the teacher step budget")
        prompts = prompts[:cap]
    result = generate_teacher(prompts,config["model"],run.path/"teacher.jsonl",meter,
                              config.get("max_new_tokens",128),config["device"],config["precision"])
    if config.get("teacher_output_reference") and result["status"]=="completed" and result["records"]>0:
        from ncm_lab.data.download import atomic_json
        from ncm_lab.runtime.io import digest
        path = run.path/"teacher.jsonl"
        atomic_json(root()/config["teacher_output_reference"],{"path":str(path),"sha256":digest(path),"run_id":run.id})
    return result


def verification(run,config,meter):
    """Real target AR versus actual cached speculative generation, identical policy."""
    from ncm_lab.runtime.io import digest
    target,backbone,tokenizer = load_pair(config["model"],config["device"],getattr(torch,config["precision"]))
    drafter = ProgressiveDrafter(backbone,target,config["schedule"],config.get("skip",False)).eval()
    if config.get("checkpoint"):
        path = Path(config["checkpoint"])
        if digest(path) != config.get("checkpoint_sha256"):
            raise ValueError("parity checkpoint hash mismatch")
        drafter.load_state_dict(torch.load(path,map_location=config["device"],weights_only=False)["model"])
    proposer = build_proposer(target,drafter)
    records = []
    for index,(item,_) in enumerate(load_records(root()/config["data_path"],config["dataset"],config["split"])):
        if index >= config["max_requests"]:
            break
        meter.check()
        text = tokenizer.apply_chat_template([{"role":"user","content":item.prompt()}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
        ids = tokenizer(text,return_tensors="pt").input_ids.to(config["device"])
        with torch.no_grad():
            expected = target.generate(ids,max_new_tokens=config["max_new_tokens"],do_sample=False,
                                       eos_token_id=tokenizer.eos_token_id)[0,ids.shape[1]:].tolist()
        actual = cached_greedy(target,proposer,ids,config["max_new_tokens"],eos=[tokenizer.eos_token_id])
        records.append({"original_id":item.original_id,"input_tokens":ids.numel(),"target_tokens":expected,
                        "speculative_tokens":actual.tokens,"equal":expected==actual.tokens,
                        "accepted":actual.accepted,"corrections":actual.corrections,"bonus":actual.bonus,
                        "verification_calls":actual.verification_calls,"cycles":actual.cycles})
        meter.record(0,0)
    write_json(run.path/"parity_observations.json",records)
    all_equal = bool(records) and all(r["equal"] for r in records)
    return {"status":"completed" if all_equal else "failed","requests":len(records),"all_equal":all_equal,
            "mismatches":sum(not r["equal"] for r in records),"configuration_scope":"Transformers recomputing proposer reference",
            "optimized_engine_validated":False,"sampling":"greedy","coverage":"declared input subset only; inspect EOS/rejection coverage before broader claims"}


def evaluate(run, config, meter):
    from ncm_lab.evaluation.financial import finqa_score, tatqa_score
    from ncm_lab.evaluation.tasks import gsm8k_score
    from ncm_lab.runtime.io import digest
    target, backbone, tokenizer = load_pair(config["model"], config["device"], getattr(torch, config["precision"]))
    drafter = ProgressiveDrafter(backbone, target, config["schedule"], config.get("skip", False)).eval()
    if config.get("checkpoint"):
        checkpoint = Path(config["checkpoint"])
        if digest(checkpoint) != config.get("checkpoint_sha256"):
            raise ValueError("evaluation checkpoint hash mismatch")
        drafter.load_state_dict(torch.load(checkpoint, weights_only=False, map_location=config["device"])["model"])
    proposer = build_proposer(target, drafter)
    dataset = config["dataset"]
    scorers = {"finqa": lambda text, gold, item: finqa_score(text.strip(), gold, item.table),
               "tatqa": lambda text, gold, item: tatqa_score(json.loads(text)["answer"], json.loads(text)["scale"], gold),
               "gsm8k": lambda text, gold, item: gsm8k_score(text, gold)}
    if dataset not in scorers:
        raise ValueError(f"{dataset}: official evaluation harness/isolation must be configured; see manifest")
    predictions = []
    workload_start = time.monotonic()
    for index, (item, gold) in enumerate(load_records(root() / config["data_path"], dataset, config["split"])):
        if index >= config["max_requests"]:
            break
        meter.check()
        messages = [{"role": "user", "content": item.prompt() + "\n" + config["output_instruction"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(config["device"])
        if inputs.input_ids.shape[1] + config["max_new_tokens"] > target.config.max_position_embeddings:
            failure = asdict(PredictionRecord(item.original_id, item.group_id, "", False, 0,
                                              error="context_limit; no silent truncation"))
            predictions.append(failure)
            with (run.path/"predictions.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(failure)+"\n")
            meter.record(0, 0)
            continue
        started, error, output, scores = time.monotonic(), None, "", {}
        try:
            if config["method"] == "AR":
                with torch.inference_mode():
                    ids = target.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False)
                output_ids = ids[0, inputs.input_ids.shape[1]:].tolist()
            else:
                decoded = cached_greedy(target, proposer, inputs.input_ids, config["max_new_tokens"],
                                        eos=[tokenizer.eos_token_id])
                output_ids = decoded.tokens
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            scores = scorers[dataset](output, gold, item)
        except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            error, output_ids = str(exc), []
        synchronize(config["device"])
        metric = scores.get("execution_accuracy", scores.get("exact_match", 0))
        record = asdict(PredictionRecord(item.original_id, item.group_id, output, bool(metric == 1) and not error,
                                         time.monotonic() - started, error, inputs.input_ids.numel(), len(output_ids)))
        record["official_scores"] = scores
        predictions.append(record)
        with (run.path / "predictions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        meter.record(0, 0)
    return {"status": "completed", **summarize(predictions, time.monotonic() - workload_start),
            "evaluation_scope": config["split"], "max_requests": config["max_requests"],
            "executor_limitation": "correctness reference; not an optimized deployment speed baseline"}
