"""Prepare allowlisted pilot prompts from the official OASST training release."""
import json
from pathlib import Path
from ncm_lab.data.loaders import oasst_prompts, normalized
from ncm_lab.runtime.io import digest, write_json


def oasst_rows(path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValueError("prompt preparation requires the project 'prepare' extra (pyarrow); see docs/prepare.md") from exc
    source = parquet.ParquetFile(path)
    required = {"message_id","message_tree_id","text","role","lang","deleted","review_result"}
    if not required <= set(source.schema.names):
        raise ValueError("pinned OASST parquet schema differs from the allowed fields")
    for batch in source.iter_batches(batch_size=1024,columns=sorted(required)):
        yield from batch.to_pylist()


def trigrams(text):
    words = normalized(text).split()
    return set(zip(words,words[1:],words[2:])) or {tuple(words)}


def prepare_prompts(oasst_path,gsm8k_path,output,budget,rows=None):
    """Use official OASST train only, then hold out whole trees for development.

    Only questions from GSM8K train enter overlap filtering; no reference answer
    enters either output. Deduplication coverage is deliberately pilot-limited.
    """
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    gsm_rows = [json.loads(line) for line in Path(gsm8k_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluation = [row["question"] for row in gsm_rows]
    eval_grams = [trigrams(text) for text in evaluation]
    inverted = {}
    for index,grams in enumerate(eval_grams):
        for gram in grams:
            inverted.setdefault(gram,set()).add(index)
    selected,seen = {"train":[],"dev":[]},set()
    skipped = {"unapproved":0,"too_long":0,"duplicate":0,"gsm8k_overlap":0}
    visible_bytes = 0
    iterator = rows if rows is not None else oasst_rows(oasst_path)
    for row in iterator:
        if row.get("review_result") is not True:
            skipped["unapproved"] += 1
            continue
        for item in oasst_prompts([row],budget.dev_fraction):
            if len(item.question) > budget.max_prompt_chars:
                skipped["too_long"] += 1
                continue
            text = normalized(item.question)
            if text in seen:
                skipped["duplicate"] += 1
                continue
            grams = trigrams(item.question)
            candidates = set().union(*(inverted.get(g,set()) for g in grams))
            if any(len(grams & eval_grams[i])/max(1,len(grams | eval_grams[i])) >= .85 for i in candidates):
                skipped["gsm8k_overlap"] += 1
                continue
            payload = {"original_id":item.original_id,"question":item.question,"group_id":item.group_id,"split":item.split}
            visible_bytes += len(json.dumps(payload,ensure_ascii=False).encode())+1
            if visible_bytes > budget.max_processed_bytes:
                raise ValueError("prepared prompt storage budget exceeded")
            selected[item.split].append(payload)
            seen.add(text)
            if sum(map(len,selected.values())) >= budget.max_prompt_examples:
                break
        if sum(map(len,selected.values())) >= budget.max_prompt_examples:
            break
    if not selected["train"] or not selected["dev"]:
        raise ValueError("no adequate train/dev prompt partition; inspect filtering and the selected preparation cap")
    if {r["group_id"] for r in selected["train"]} & {r["group_id"] for r in selected["dev"]}:
        raise ValueError("conversation-tree leakage in prepared prompts")
    paths = {}
    for split,records in selected.items():
        dest = output/f"oasst1_{split}_prompts.jsonl"
        with dest.open("x",encoding="utf-8",newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record,ensure_ascii=False)+"\n")
        paths[split] = {"path":str(dest),"records":len(records),"sha256":digest(dest)}
    provenance = {"schema_version":1,"source_split":"official OASST train only",'official_split':'train',
        'derived_split_policy':{'version':'oasst_tree_hash_holdout_v1','dev_fraction':budget.dev_fraction},
        "source_sha256":digest(oasst_path),"gsm8k_prompt_source_sha256":digest(gsm8k_path),
        "derived_development_split":"deterministic whole-message-tree holdout",
        "outputs":paths,"filtered":skipped,"gold_in_prompts":False,
        "deduplication_scope":"GSM8K train prompt overlap and exact OASST duplicates; full ACL suite overlap audit still required",
        "publication_training_ready":False}
    write_json(output/"prompt_provenance.json",provenance,exclusive=True)
    return provenance
