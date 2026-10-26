from dataclasses import asdict
import json
import time
from ncm_lab.business.evidence import BM25, chunks
from ncm_lab.business.workflow import FinancialAnswer, validate, workpapers


def workpaper_prompt(items, evidence, errors=None):
    payload = {"questions": [{"question_id": item.original_id, "question": item.question, "table": item.table}
                             for item in items], "evidence": [asdict(e) for e in evidence],
               "output_contract": {"answers": ["question_id", "answer", "unit", "scale", "period", "source_refs", "program"]}}
    if errors:
        payload["detected_structural_errors"] = errors
    return json.dumps(payload, ensure_ascii=False)


def parse_workpaper(text, expected_ids):
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("oversized workpaper output")
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"answers"} or not isinstance(payload["answers"], list):
        raise ValueError("expected one JSON object containing an answers array")
    answers, seen = [], set()
    required = {"question_id", "answer", "unit", "scale", "period", "source_refs"}
    for row in payload["answers"]:
        if not isinstance(row, dict) or not required <= row.keys() or set(row) - (required | {"program"}):
            raise ValueError("answer contract missing required fields or has undeclared fields")
        if row["question_id"] in seen or row["question_id"] not in expected_ids:
            raise ValueError("duplicate or unknown question ID")
        if not isinstance(row["source_refs"], list) or not all(isinstance(x,str) for x in row["source_refs"]):
            raise ValueError("source_refs must be evidence IDs")
        answers.append(FinancialAnswer(**row))
        seen.add(row["question_id"])
    if seen != set(expected_ids):
        raise ValueError("one answer per original question is required")
    return answers


def execute_workpapers(inputs, generate, bundle_size=1, validation="checks", repair_attempts=0,
                       max_context_chars=8000, max_questions=8, retrieval="bm25"):
    """Deployment workflow accepts no EvaluationGold; scoring is downstream only."""
    if validation not in {"contract", "checks"} or repair_attempts not in (0,1):
        raise ValueError("unsupported validation/repair condition")
    if max_questions <= 0:
        raise ValueError("workflow question cap must be positive")
    completed = 0
    for bundle in workpapers(inputs,bundle_size):
        if completed + len(bundle) > max_questions:
            break
        started = time.monotonic()
        contexts = "\n\n".join(bundle[0].context)
        # If table is separately released, preserve complete rows in a single evidence block.
        if bundle[0].table:
            contexts += "\n\n" + "\n".join(" | ".join(r) for r in bundle[0].table)
        evidence = list(chunks(bundle[0].group_id,contexts))
        if retrieval != "bm25":
            raise ValueError("select a pinned cache-only dense retriever through DenseRetriever before enabling that condition")
        selected = BM25(evidence).retrieve(" ".join(q.question for q in bundle),max_chars=max_context_chars)
        refs = selected["records"]
        ids = {q.original_id for q in bundle}
        attempts, answers, checks = [], [], []
        prompt = workpaper_prompt(bundle,refs)
        for attempt in range(1 + repair_attempts):
            output = generate(prompt)
            try:
                answers = parse_workpaper(output,ids)
                tables = {q.original_id:q.table for q in bundle}
                checks = ([validate(a,refs,tables[a.question_id]) for a in answers] if validation == "checks"
                          else [{"status":"contract_only", "support":"unassessed"} for a in answers])
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                checks = [{"status":"missing_required_field","reason":str(exc)}]
                answers = []
            attempts.append({"output":output,"checks":checks})
            if all(c["status"] in {"valid","contract_only"} for c in checks):
                break
            prompt = workpaper_prompt(bundle,refs,errors=checks) + "\nPrevious output:\n" + output
        completed += len(bundle)
        yield {"question_ids":[q.original_id for q in bundle], "group_id":bundle[0].group_id,
               "answers":[asdict(a) for a in answers], "checks":checks, "attempts":attempts,
               "request_seconds":time.monotonic()-started, "retrieval_missing_evidence":selected["missing_evidence"],
               "excluded_for_context_budget":selected["excluded_for_budget"],
               "outcome":"returned" if answers and all(c["status"] in {"valid","contract_only"} for c in checks) else "requires_review"}
