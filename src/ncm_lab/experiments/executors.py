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
from ncm_lab.models.factory import build_drafter, resolve_method
from ncm_lab.decoding.policy import resolve_policy, ar_greedy
from ncm_lab.decoding.hf import CachedProposer
from ncm_lab.training.checkpoints import read_checkpoint, load_student
from ncm_lab.runtime.budget import BudgetExceeded
from ncm_lab.models.upstream import FeatureExtractor, load_pair
from ncm_lab.runtime.io import canonical_hash, root, write_json, read
from ncm_lab.training.loop import train


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def fixture_workflow(run, config, meter):
    torch.set_num_threads(1)
    torch.manual_seed(config.get("seed", 0))
    target, backbone = tiny_pair()
    drafter = build_drafter(target, backbone, resolve_method({"method": "HIER_SKIP"}))
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
    started = time.perf_counter()
    actual = speculative(probs, lambda prefix: [[.8, .1, .1]] * 3, [0], 12)
    prediction = PredictionRecord("software-fixture:0", "software-fixture", str(actual.tokens),
                                  actual.tokens == expected, time.perf_counter() - started,
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
    device, precision = config.get("device", "cpu"), config.get("precision", "float32")
    dtype = getattr(torch, precision)
    if fixture:
        torch.manual_seed(config.get("seed", 0))
        target, base = tiny_pair()
    else:
        target, base, _ = load_pair(config["model"], device, dtype)
    torch.set_num_threads(config.get("cpu_threads", 1))
    observations = []
    rng = random.Random(config.get("seed", 0))
    names = config.get("methods", ["FLAT_ADAPTED", "HIER_SKIP", "HIER_NO_SKIP"])
    models = {}
    for name in names:
        options = {**config, "method": name}
        options.pop("schedule", None)
        options.pop("skip", None)
        options.update(config.get('method_overrides',{}).get(name,{}))
        if name == "FLAT_COST" and config.get("cost_candidate"):
            options["cost_selection"] = "unselected-development-profile-candidate"
        codec = None
        from ncm_lab.models.factory import LATENT
        if name in LATENT and name != "HIER_LATENT_CE_ONLY":
            if fixture:
                from ncm_lab.models.codecs import MultiscaleCodec
                kind = "linear" if name == "HIER_LINEAR_CODEC" else "independent" if name in {"HIER_INDEPENDENT_AE", "HIER_SINGLE_AE"} else "nested"
                codec = MultiscaleCodec(base.config.hidden_size, 4, kind, (8,) if name == "HIER_SINGLE_AE" else (4,8,15))
                options["codec_sha256"] = "software-fixture-untrained-codec"
            else:
                from ncm_lab.training.codec import load_selected_codec
                codec = load_selected_codec(Path(config["codec_selection"]),device,dtype,config["codec_sha256"])
        method = resolve_method(options)
        models[name] = build_drafter(target,copy.deepcopy(base),method,codec,latent_width=4).eval()
    write_json(run.path/'resolved_methods.json',{name:model.compatibility for name,model in models.items()},exclusive=True)
    generator = torch.Generator(device=device).manual_seed(config.get("data_seed",0))
    for batch in config.get("concurrency", [1, 4]):
        for prefix in config.get("prefix_lengths", [8, 32]):
            features = torch.randn(batch,prefix,len(base.target_layer_ids)*base.config.hidden_size,
                                   generator=generator,device=device,dtype=dtype)
            positions = torch.arange(prefix,device=device)[None].expand(batch,-1)
            args = (features,positions,torch.zeros_like(positions),torch.full((batch,),prefix,device=device),
                    torch.ones(batch,dtype=torch.long,device=device),torch.zeros(batch,dtype=torch.long,device=device))
            for repetition in range(config.get("repetitions",3)+1):
                order = list(names)
                rng.shuffle(order)
                for method_order,name in enumerate(order):
                    meter.check()
                    synchronize(device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        logits = models[name](*args)
                    synchronize(device)
                    seconds = time.perf_counter()-start
                    meter.record(0,0)
                    if repetition:
                        observations.append({"method":name,"batch":batch,"prefix_length":prefix,
                            "repetition":repetition,"seconds":seconds,"method_order_in_condition":method_order,
                            "timing_order_seed":config.get("seed",0),"stage_lengths":models[name].stage_lengths,
                            "parameter_counts":models[name].parameter_counts(),"resolved_method":models[name].resolved_method,
                            "future_logits_shape":list(logits.shape),
                            "scope":"draft forward including latent branches; verification excluded"})
    write_json(run.path/"component_timings.json",observations)
    return {"status":"completed","timing_observations":len(observations),
        "result_origin":"artificial_tensor_microbenchmark","performance_claim_eligible":False,
        "limitation":"component diagnostic; no measured proposal-quality or complete-generation conclusion"}


def recomputing_proposer(target, drafter):
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


def build_proposer(target, drafter):
    return CachedProposer(drafter)


def construct(config, operation="diagnostic"):
    if config.get('cpu_threads'):
        torch.set_num_threads(config['cpu_threads'])
    method = resolve_method(config, operation)
    from ncm_lab.models.upstream import preflight_generation_metadata
    preflight_generation_metadata(config["model"],config.get("generation_policy"))
    if operation == "evaluate" and method.name not in {"AR", "FLAT_RELEASED","STRONG_EXTERNAL"}:
        from ncm_lab.training.selection import validate_selection
        selection = validate_selection(config["checkpoint_selection"])
        if selection["weights_sha256"] != config["checkpoint_sha256"]:
            raise ValueError("evaluation checkpoint differs from held-out selection")
    target, backbone, tokenizer = load_pair(config["model"], method.device, getattr(torch, method.precision))
    codec = None
    if method.latent and method.name != "HIER_LATENT_CE_ONLY":
        from ncm_lab.training.codec import load_selected_codec
        codec = load_selected_codec(Path(config["codec_selection"]), method.device,
                                    getattr(torch, method.precision), method.codec_id)
    drafter = build_drafter(target, backbone, method, codec, config.get("latent_width", 256))
    if config.get("checkpoint"):
        from ncm_lab.runtime.io import digest
        path = Path(config["checkpoint"])
        if digest(path) != config.get("checkpoint_sha256"):
            raise ValueError("checkpoint digest mismatch")
        payload = read_checkpoint(path)
        if canonical_hash(payload.get("architecture")) != canonical_hash(drafter.compatibility):
            raise ValueError("checkpoint method/target/codec/execution identity mismatch")
        load_student(drafter, payload["model"])
    return target, drafter, tokenizer, codec


def development(run, config, meter):
    from ncm_lab.training.loop import evaluate_development
    from ncm_lab.training.records import load_records
    target, drafter, _, _ = construct(config)
    rows = load_records(root()/config["development_data"],"dev")
    observation = evaluate_development(target,drafter,rows,meter)
    observation.update(checkpoint_sha256=config["checkpoint_sha256"],method=config["method"],seed=config["seed"])
    write_json(run.path/"development-observation.json",observation,exclusive=True)
    return {"status":"completed",**observation}


def latent_diagnostics(run,config,meter):
    from ncm_lab.training.records import load_records,anchors
    from ncm_lab.evaluation.latent_mechanisms import diagnose
    target,drafter,_,codec=construct(config,'evaluate')
    if codec is None or drafter.latent is None:
        raise ValueError('latent diagnostics require a selected fitted codec and latent student')
    rows=load_records(root()/config['evaluation_data'],config['split'])
    observations=[]
    for index,row in enumerate(rows):
        if index>=config.get('max_requests',4):
            break
        eligible=anchors(row)
        if not eligible:
            observations.append({'original_id':row['original_id'],'skipped':'no observed response successor'})
            continue
        positions=len(row['input_ids'])*(2+2*len(drafter.latent.levels)+int(drafter.skip_enabled))
        meter.check(positions,0)
        meter.source_tokens+=positions
        observation=diagnose(target,drafter,codec,row,eligible[0])
        observation.update(original_id=row['original_id'],group_id=row['group_id'],anchor=eligible[0])
        folder=run.path/('diagnostic-%06d'%index)
        write_json(folder/'latent-diagnostics.json',observation,exclusive=True)
        observations.append({'original_id':row['original_id'],'artifact':str(folder.relative_to(run.path))})
        meter.record(0,0)
    write_json(run.path/'diagnostic_inputs.json',observations,exclusive=True)
    return {'status':'completed','diagnostic_requests':meter.steps,'target_processed_positions':meter.source_tokens,
            'split':config['split'],'selection':False,'oracle_available_in_generation':False}


def cost_selection_stage(run,config,meter):
    from ncm_lab.experiments.cost_selection import select_cost
    meter.check()
    result=select_cost(config['profile_runs'],run.path/'cost-selection.json',config.get('tolerance',.1))
    meter.record(0,0)
    return {'status':'completed','selection':result}


def train_real(run, config, meter):
    from ncm_lab.training.loop import load_training, resolve_training_data
    from ncm_lab.runtime.io import digest
    data = resolve_training_data(config)
    from ncm_lab.training.records import subset_for_tier
    records = subset_for_tier(load_training(data),config)
    if config.get("total_new_training_wall_seconds"):
        from ncm_lab.experiments.cost_selection import training_cost_budget
        meter.budget, charges = training_cost_budget(config,meter.budget)
        write_json(run.path/"resolved_training_cost.json",charges,exclusive=True)
    target, drafter, _, codec = construct(config, "train")
    write_json(run.path/'resolved_method.json',drafter.compatibility,exclusive=True)
    resume = None
    if config.get("resume_checkpoint"):
        resume = Path(config["resume_checkpoint"])
        if digest(resume) != config.get("resume_sha256"):
            raise ValueError("resume requires the expected trusted checkpoint SHA256")
    from ncm_lab.training.records import load_records
    dev_path = config.get("development_data")
    if config.get("development_data_reference"):
        ref = read(root()/config["development_data_reference"])
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("development teacher artifact changed")
        dev_path = ref["path"]
    dev = load_records(root()/dev_path, "dev") if dev_path else None
    if dev and config.get("max_development_examples"):
        dev = dev[:config["max_development_examples"]]
    cache=None
    if config.get('latent_cache'):
        from ncm_lab.training.latent_supervision import LatentCache,cache_execution
        cache=LatentCache(config['latent_cache'],config['codec_sha256'],cache_execution(config,codec))
    return train(target, drafter, records, meter, canonical_hash({"file":digest(data),"subset":canonical_hash(records)}),
                 steps=config.get("steps", meter.budget.max_steps),
                 seed=config.get("data_seed", config["seed"]), resume=resume,
                 learning_rate=config.get("learning_rate", 1e-5), dev_records=dev, codec=codec,
                 checkpoint_interval=config.get("checkpoint_interval",1),
                 evaluation_interval=config.get("evaluation_interval",1),
                 latent_level_weights=config.get("latent_level_weights"),
                 latent_cache=cache,fixed_codec_transfer=config.get('fixed_codec_transfer',False),
                 latent_weight=0 if config["method"] in {"HIER_CODEC_CE_ONLY", "HIER_LATENT_CE_ONLY"}
                 else config.get("latent_weight", 1.0))


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
                              config.get("max_new_tokens",128),config["device"],config["precision"],
                              config.get("purpose", "train"),config.get('generation_policy'))
    if config.get("teacher_output_reference") and result["status"]=="completed" and result["records"]>0:
        from ncm_lab.data.download import atomic_json
        from ncm_lab.runtime.io import digest
        path = run.path/"teacher.jsonl"
        reference = {"path":str(path),"sha256":digest(path),"run_id":run.id}
        if config.get("input_receipts"):
            from ncm_lab.data.artifacts import lock_input
            receipt = run.path/"teacher.input.json"
            split='test' if config.get('purpose')=='test_evaluation' else config.get('purpose','train')
            lock_input(path,receipt,"teacher",split,config.get('dataset','oasst1'),
                       source_receipt=config["input_receipts"][0])
            reference["input_receipt"] = str(receipt)
        atomic_json(root()/config["teacher_output_reference"],reference)
    return result


def verification(run,config,meter):
    """Real target AR versus actual cached speculative generation, identical policy."""
    if config["method"] == "AR":
        from ncm_lab.models.upstream import load_target
        target,tokenizer = load_target(config["model"],config["device"],getattr(torch,config["precision"]))
        drafter = None
        from ncm_lab.models.factory import execution_settings
        write_json(run.path/'resolved_method.json',{'method':asdict(resolve_method(config)),
            'target':target.config.to_dict(),'execution':execution_settings()},exclusive=True)
    else:
        target, drafter, tokenizer, _ = construct(config)
        drafter.eval()
        write_json(run.path/'resolved_method.json',drafter.compatibility,exclusive=True)
    policy = resolve_policy(target, tokenizer, {"max_new_tokens": config["max_new_tokens"],
        "tokenizer_revision": config["model"]["tokenizer_revision"], "precision": config["precision"],
        **config.get("generation_policy", {})})
    write_json(run.path/"generation_policy.json", {**asdict(policy), "sha256": policy.digest})
    proposer = build_proposer(target,drafter) if drafter else None
    records = []
    for index,(item,_) in enumerate(load_records(root()/config["data_path"],config["dataset"],config["split"])):
        if index >= config["max_requests"]:
            break
        meter.check()
        text = tokenizer.apply_chat_template([{"role":"user","content":item.prompt()}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
        ids = tokenizer(text,return_tensors="pt").input_ids.to(config["device"])
        meter.check(2*ids.numel()+17*policy.max_new_tokens,0)
        with torch.no_grad():
            expected = ar_greedy(target, ids, policy,guard=lambda:meter.check(step=False))
        if proposer is None:
            if policy.stop_sequences:
                raise ValueError("independent HF AR parity adapter currently requires EOS-only stopping")
            from transformers import GenerationConfig
            from ncm_lab.decoding.reference import DecodeResult
            generation = GenerationConfig(max_new_tokens=policy.max_new_tokens,do_sample=False,num_beams=1,
                eos_token_id=list(policy.eos_ids) or None,bos_token_id=policy.bos_id,pad_token_id=policy.pad_id)
            with torch.inference_mode():
                tokens = target.generate(ids,generation_config=generation)[0,ids.shape[1]:].tolist() if policy.max_new_tokens else []
            actual = DecodeResult(tokens,telemetry={"target_processed_positions":ids.numel()+len(tokens)-1 if tokens else 0})
        else:
            actual = cached_greedy(target, proposer, ids, policy=policy,guard=lambda:meter.check(step=False))
        records.append({"original_id":item.original_id,"input_tokens":ids.numel(),"target_tokens":expected,
                        "speculative_tokens":actual.tokens,"equal":expected==actual.tokens,
                        "accepted":actual.accepted,"corrections":actual.corrections,"bonus":actual.bonus,
                        "verification_calls":actual.verification_calls,"cycles":actual.cycles,
                        "telemetry": actual.telemetry, "policy_hash": policy.digest})
        with (run.path/'parity_observations.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(records[-1])+'\n')
        meter.record(actual.telemetry["target_processed_positions"]+(ids.numel()+len(expected)-1 if expected else 0),0)
    write_json(run.path/"parity_observations.json",records)
    all_equal = bool(records) and all(r["equal"] for r in records)
    return {"status":"completed" if all_equal else "failed","requests":len(records),"all_equal":all_equal,
            'generated_tokens_compared':sum(len(r['target_tokens']) for r in records),
            "mismatches":sum(not r["equal"] for r in records),"configuration_scope":"Transformers incremental SDPA",
            "policy_hash": policy.digest,
            "optimized_engine_validated":False,"sampling":"greedy","coverage":"declared input subset only; inspect EOS/rejection coverage before broader claims"}


def evaluate(run, config, meter):
    from ncm_lab.evaluation.financial import finqa_score, tatqa_score
    from ncm_lab.evaluation.tasks import gsm8k_score
    official=None
    if config['dataset'] in {'ifeval','longbench','humaneval_plus'}:
        from ncm_lab.evaluation.tasks import official_scorer
        official=official_scorer(config['dataset'],config['official_evaluator'])
    method = resolve_method(config, "evaluate")
    if method.name == "AR":
        from ncm_lab.models.upstream import load_target
        target, tokenizer = load_target(config["model"], method.device, getattr(torch, method.precision))
        proposer = None
        from ncm_lab.models.factory import execution_settings
        write_json(run.path/'resolved_method.json',{'method':asdict(method),'target':target.config.to_dict(),
                                                  'execution':execution_settings()},exclusive=True)
    else:
        target, drafter, tokenizer, _ = construct(config, "evaluate")
        proposer = build_proposer(target, drafter)
        write_json(run.path/'resolved_method.json',drafter.compatibility,exclusive=True)
    policy = resolve_policy(target, tokenizer, {"max_new_tokens": config["max_new_tokens"],
        "tokenizer_revision": config["model"]["tokenizer_revision"], "precision": config["precision"],
        **config.get("generation_policy", {})})
    write_json(run.path/"generation_policy.json", {**asdict(policy), "sha256": policy.digest})
    dataset = config["dataset"]
    scorers = {"finqa": lambda text, gold, item: finqa_score(text.strip(), gold, item.table),
               "tatqa": lambda text, gold, item: tatqa_score(json.loads(text)["answer"], json.loads(text)["scale"], gold),
               "gsm8k": lambda text, gold, item: gsm8k_score(text, gold)}
    if dataset in {"ifeval", "longbench", "humaneval_plus"}:
        scorers[dataset] = official
    if dataset not in scorers:
        raise ValueError(f"{dataset}: official evaluation harness/isolation must be configured; see manifest")
    predictions = []
    warmups = []
    for index, (item, _) in enumerate(load_records(root()/config["data_path"],dataset,config["split"])):
        if index >= config.get("warmup_requests",0):
            break
        meter.check()
        content = item.prompt()+"\n"+config["output_instruction"]
        if dataset in {"ifeval","longbench"}:
            from ncm_lab.evaluation.tasks import official_prompt
            content = official_prompt(dataset,item,config["official_evaluator"])
        text = tokenizer.apply_chat_template([{"role":"user","content":content}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
        ids = tokenizer(text,return_tensors="pt").input_ids.to(config["device"])
        if ids.shape[1]+policy.max_new_tokens>target.config.max_position_embeddings:
            warmups.append({'original_id':item.original_id,'seconds':None,'target_processed_positions':0,
                            'error':'context_limit; warmup not attempted'})
            meter.record(0,0)
            continue
        meter.check(ids.numel()+16*policy.max_new_tokens,0)
        synchronize(config["device"])
        started = time.perf_counter()
        if proposer is None:
            output_ids = ar_greedy(target,ids,policy,guard=lambda:meter.check(step=False))
            processed = ids.numel()+len(output_ids)-1 if output_ids else 0
        else:
            decoded_warmup = cached_greedy(target,proposer,ids,policy=policy,guard=lambda:meter.check(step=False))
            processed = decoded_warmup.telemetry["target_processed_positions"]
        synchronize(config["device"])
        warmups.append({"original_id":item.original_id,"seconds":time.perf_counter()-started,'target_processed_positions':processed})
        meter.record(processed,0)
    write_json(run.path/"warmup.json",warmups)
    workload_start = time.perf_counter()
    evaluation_status='completed'
    for index, (item, gold) in enumerate(load_records(root() / config["data_path"], dataset, config["split"])):
        if index >= config["max_requests"]:
            break
        meter.check()
        request_started=time.perf_counter()
        messages = [{"role": "user", "content": item.prompt() + "\n" + config["output_instruction"]}]
        if dataset in {"ifeval","longbench"}:
            from ncm_lab.evaluation.tasks import official_prompt
            messages = [{"role": "user", "content": official_prompt(dataset,item,config["official_evaluator"])}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(config["device"])
        if inputs.input_ids.shape[1] + config["max_new_tokens"] > target.config.max_position_embeddings:
            failure = asdict(PredictionRecord(item.original_id, item.group_id, "", False, time.perf_counter()-request_started,
                                              error="context_limit; no silent truncation",input_tokens=inputs.input_ids.numel(),output_tokens=0))
            failure.update(method=config['method'],training_seed=None if config['method'] in {'AR','FLAT_RELEASED','STRONG_EXTERNAL'} else config.get('seed'),
                dataset=dataset,comparison=config.get('comparison','equal_data'),policy_hash=policy.digest,
                repetition_id=config.get('timing_repetition',0),target_processed_positions=0,generation_seconds=None,
                generation_missing_reason='context limit: generation not attempted',input_ids=inputs.input_ids[0].tolist(),output_ids=[])
            predictions.append(failure)
            with (run.path/"predictions.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(failure)+"\n")
            meter.record(0, 0)
            continue
        started, error, output, scores = time.perf_counter(), None, "", {}
        generation_started=started
        generation_begun=False
        output_ids=[]
        decoded = None
        generation_seconds = scoring_seconds = None
        processed = 0
        try:
            meter.check(inputs.input_ids.numel()+16*policy.max_new_tokens,0)
            synchronize(config["device"])
            generation_started = time.perf_counter()
            generation_begun=True
            if config["method"] == "AR":
                with torch.inference_mode():
                    output_ids = ar_greedy(target, inputs.input_ids, policy,guard=lambda:meter.check(step=False))
            else:
                decoded = cached_greedy(target, proposer, inputs.input_ids, policy=policy,guard=lambda:meter.check(step=False))
                output_ids = decoded.tokens
            synchronize(config["device"])
            generation_seconds = time.perf_counter()-generation_started
            processed = decoded.telemetry["target_processed_positions"] if decoded else inputs.input_ids.numel()+len(output_ids)-1 if output_ids else 0
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            scoring_started = time.perf_counter()
            scores = scorers[dataset](output, gold, item)
            scoring_seconds = time.perf_counter()-scoring_started
        except BudgetExceeded as exc:
            error=str(exc)
            output_ids=getattr(exc,'partial_tokens',[])
            decoded=getattr(exc,'decode_result',None)
            if decoded:
                decoded.telemetry['stop_reason']='budget'
            processed=(decoded.telemetry['target_processed_positions'] if decoded else
                       inputs.input_ids.numel()+len(output_ids)-1 if output_ids else 0)
            output=tokenizer.decode(output_ids,skip_special_tokens=True)
            generation_seconds=time.perf_counter()-generation_started if generation_begun else None
            evaluation_status='cancelled'
        except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            error = str(exc)
            if generation_begun and generation_seconds is None:
                processed=None
        synchronize(config["device"])
        metric = scores.get("execution_accuracy", scores.get("exact_match", 0))
        record = asdict(PredictionRecord(item.original_id, item.group_id, output, bool(metric == 1) and not error,
                                         time.perf_counter() - request_started, error, inputs.input_ids.numel(), len(output_ids)))
        record["official_scores"] = scores
        record.update(input_ids=inputs.input_ids[0].tolist(), output_ids=output_ids, policy_hash=policy.digest,
                      method=config["method"], training_seed=None if config['method'] in {'AR','FLAT_RELEASED','STRONG_EXTERNAL'} else config.get("seed"), dataset=dataset,
                      shared_reference=config['method'] in {'AR','FLAT_RELEASED','STRONG_EXTERNAL'},
                      comparison=config.get("comparison","equal_data"),
                      repetition_id=config.get("timing_repetition",0), inference_seed=config.get("inference_seed",0),
                      method_order=config.get("job_order"), warmup_requests=sum(w.get('seconds') is not None for w in warmups),
                      request_timing_scope='chat template, tokenization, synchronized generation, decoding and scoring',
                      generation_seconds=generation_seconds, scoring_seconds=scoring_seconds,
                      target_processed_positions=processed,
                      decoder=asdict(decoded) if decoded else None,
                      decoder_missing_reason=None if decoded else "target-only AR")
        if decoded:
            record.update(accepted_tokens=decoded.accepted, correction_tokens=decoded.corrections,
                          bonus_tokens=decoded.bonus, verification_calls=decoded.verification_calls)
        predictions.append(record)
        with (run.path / "predictions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        meter.record(processed if processed is not None else inputs.input_ids.numel()+16*policy.max_new_tokens, 0)
        if evaluation_status=='cancelled':
            break
    return {"status": evaluation_status, **summarize(predictions, time.perf_counter() - workload_start),
            'target_processed_positions':sum(r.get('target_processed_positions',0) for r in predictions)+sum(w['target_processed_positions'] for w in warmups)
                if all(r.get('target_processed_positions') is not None for r in predictions) else None,
            "policy_hash":policy.digest,
            "evaluation_scope": config["split"], "max_requests": config["max_requests"],
            "executor_limitation": "correctness reference; not an optimized deployment speed baseline"}
