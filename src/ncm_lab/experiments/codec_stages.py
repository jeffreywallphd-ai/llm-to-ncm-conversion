from pathlib import Path
import torch
from ncm_lab.runtime.io import root, canonical_hash, write_json
from ncm_lab.models.upstream import load_target
from ncm_lab.training.records import load_records, anchors, subset_for_tier, disjoint
from ncm_lab.training.codec import fit_codec, load_selected_codec, evaluate_codec
from ncm_lab.training.latent_supervision import teacher_block, cache_identity,cache_execution
from ncm_lab.training.checkpoints import atomic_checkpoint


def codec_stage(run, config, meter):
    train = subset_for_tier(load_records(root()/config["training_data"], "train"),config)
    dev = load_records(root()/config["development_data"], "dev")
    if config.get("max_development_examples"):
        dev = dev[:config["max_development_examples"]]
    disjoint(train,dev)
    target, _ = load_target(config["model"], config["device"], getattr(torch, config["precision"]))
    resume=None
    if config.get('resume_checkpoint'):
        from ncm_lab.runtime.io import digest
        resume=Path(config['resume_checkpoint'])
        if digest(resume)!=config.get('resume_sha256'):
            raise ValueError('codec resume checkpoint identity changed')
    return {"status": "completed", "codec": fit_codec(target, train, dev, meter,
        config.get("prediction_weights", [0.0, .1]), config.get("steps", 2), config.get("codec_seed", 0),
        config.get("latent_width", 256), config.get("codec_kind", "nested"),
        tuple(config.get("levels", [4, 8, 15])), dependencies=config["model"],resume=resume)}


def codec_evaluate_stage(run, config, meter):
    target, _ = load_target(config["model"], config["device"], getattr(torch, config["precision"]))
    codec = load_selected_codec(Path(config["codec_selection"]), config["device"],
                                getattr(torch, config["precision"]), config["codec_sha256"])
    split=config.get('split','dev')
    rows = load_records(root()/config.get('evaluation_data',config.get('development_data')), split)
    plan=[{'original_id':r['original_id'],'anchor':anchors(r)[0] if anchors(r) else None,
           'skip_reason':None if anchors(r) else 'no observed response successor'} for r in rows]
    write_json(run.path/'diagnostic_input_plan.json',plan,exclusive=True)
    return {"status": "completed", "diagnostics": evaluate_codec(codec, target, rows, meter,
        normalization=codec.normalization,selection=False),"rates": codec.rates(),'scope':split,'checkpoint_selection':False}


def latent_cache_stage(run, config, meter):
    target, _ = load_target(config["model"], config["device"], getattr(torch, config["precision"]))
    codec = load_selected_codec(Path(config["codec_selection"]), config["device"],
                                getattr(torch, config["precision"]), config["codec_sha256"])
    rows = load_records(root()/config["training_data"], "train")
    references = []
    execution=cache_execution(config,codec)
    count=config.get('cache_anchors_per_sequence',1)
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=meter.budget.max_steps:
        raise ValueError('cached anchors per sequence must be positive and bounded by the run step cap')
    for row in rows:
        for anchor in anchors(row)[:count]:
            meter.check(len(row["input_ids"]), 15)
            states, valid, _ = teacher_block(target, row, anchor)
            with torch.no_grad():
                codes = codec.encode(states, valid)
            key = cache_identity(canonical_hash(row), anchor, config["codec_sha256"], row["policy_hash"],execution,codec.levels)
            path = run.path / (key + ".pt")
            artifact = atomic_checkpoint(path, {"identity": key, "codes": codes, "valid": valid,
                "source_hash": canonical_hash(row), "anchor": anchor, "codec_sha256": config["codec_sha256"]}, meter)
            references.append({"artifact": artifact, "path": path.name, "identity": key})
            meter.record(len(row["input_ids"]), int(valid.sum()))
    write_json(run.path/"latent-cache.json", {"schema_version": 2, "shards": references,'execution':execution,
        'plan':'first declared number of eligible anchors per source','anchors_per_sequence':count,
        "codec_sha256": config["codec_sha256"], "inference_input": False}, exclusive=True)
    return {"status": "completed", "shards": len(references), "target_positions": meter.source_tokens}
