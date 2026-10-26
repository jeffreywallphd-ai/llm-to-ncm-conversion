import hashlib
import json
from pathlib import Path
import re
from ncm_lab.contracts import ModelInput, EvaluationGold


def group_hash(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_records(path, dataset, split):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(x) for x in text.splitlines() if x.strip()] if path.suffix == ".jsonl" else json.loads(text)
    for index, row in enumerate(rows):
        if dataset == "finqa":
            qa = row["qa"]
            context = tuple(row.get("pre_text", []) + row.get("post_text", []))
            table = tuple(tuple(str(c) for c in r) for r in row.get("table", []))
            rid = str(row["id"])
            # Report IDs conventionally contain a path prefix before a question suffix.
            gid = rid.split(".pdf")[0] if ".pdf" in rid else group_hash([context, table])
            yield (ModelInput(rid, dataset, split, gid, qa["question"], context, table),
                   EvaluationGold(rid, qa.get("exe_ans", qa.get("answer")), qa.get("program"),
                                  evidence_labels=qa.get("gold_inds"), official=qa))
        elif dataset == "tatqa":
            table = tuple(tuple(str(c) for c in r) for r in row["table"]["table"])
            context = tuple(p["text"] for p in row.get("paragraphs", []))
            gid = str(row["table"].get("uid") or group_hash([context, table]))
            for qa in row["questions"]:
                rid = str(qa["uid"])
                yield (ModelInput(rid, dataset, split, gid, qa["question"], context, table),
                       EvaluationGold(rid, qa.get("answer"), qa.get("derivation"), qa.get("scale", ""),
                                      qa.get("rel_paragraphs"), qa))
        elif dataset == "docfinqa":
            # Released schema is inspected separately; preserve filings as text, not guessed geometry.
            rid = str(row.get("id", index))
            question = row.get("question", row.get("Question"))
            filing = row.get("context", row.get("Context"))
            if not isinstance(question, str) or not isinstance(filing, str):
                raise ValueError("DocFinQA schema mismatch: expected Question/Context strings; inspect pinned release")
            yield (ModelInput(rid, dataset, split, group_hash(filing), question, (filing,)),
                   EvaluationGold(rid, row.get("answer", row.get("Answer")),
                                  row.get("program", row.get("Program"))))
        elif dataset == "gsm8k":
            rid = f"{split}:{index}"
            yield (ModelInput(rid, dataset, split, group_hash(row["question"]), row["question"]),
                   EvaluationGold(rid, row["answer"]))
        elif dataset == "ifeval":
            rid = str(row["key"])
            yield (ModelInput(rid, dataset, split, rid, row["prompt"]), EvaluationGold(rid, None, official=row))
        elif dataset == "humaneval_plus":
            rid = row["task_id"]
            yield (ModelInput(rid, dataset, split, rid, row["prompt"]),
                   EvaluationGold(rid, row.get("canonical_solution"), official=row))
        elif dataset == "longbench":
            rid = str(row.get("_id", index))
            yield (ModelInput(rid, dataset, split, group_hash(row["context"]), row["input"], (row["context"],)),
                   EvaluationGold(rid, row["answers"], official={"task": row.get("dataset", "gov_report")}))
        else:
            raise ValueError(f"unsupported adapter: {dataset}")


def assert_disjoint(records):
    groups, prompts = {}, {}
    for item in records:
        family = "finqa_family" if item.dataset in {"finqa", "docfinqa"} else item.dataset
        for key, store in (((family, item.group_id), groups), ((family, normalized(item.question)), prompts)):
            previous = store.setdefault(key, item.split)
            if previous != item.split:
                raise ValueError(f"split leakage in {family}: {item.original_id}")


def normalized(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def near_duplicate(a, b, threshold=0.85):
    def grams(text):
        words = normalized(text).split()
        return set(zip(words, words[1:], words[2:])) or {tuple(words)}
    x, y = grams(a), grams(b)
    return len(x & y) / max(1, len(x | y)) >= threshold


def oasst_prompts(rows, dev_fraction=0.1):
    """Deterministic whole-tree split; deleted/non-English/disallowed rows excluded."""
    for row in rows:
        if (row.get("lang") != "en" or row.get("deleted") or row.get("role") != "prompter"
                or row.get("review_result") is False or not row.get("text")):
            continue
        tree = str(row["message_tree_id"])
        fraction = int(hashlib.sha256(tree.encode()).hexdigest()[:8], 16) / 2**32
        yield ModelInput(str(row["message_id"]), "oasst1", "dev" if fraction < dev_fraction else "train",
                         tree, row["text"])
