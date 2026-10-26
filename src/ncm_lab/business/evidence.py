from collections import Counter
import hashlib
import math
import re
from ncm_lab.contracts import EvidenceRecord


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def chunks(document_id, text, max_chars=2000, structure="text_only"):
    """Preserve released paragraphs and contiguous table blocks; never guess geometry.

    Oversized blocks are retained intact and may be excluded by the context budget.
    Offsets always refer to the original source string.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    blocks = list(re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text))
    output, start, end = [], None, None
    for block in blocks:
        if start is not None and block.end() - start > max_chars:
            output.append((start, end))
            start = None
        if start is None:
            start = block.start()
        end = block.end()
    if start is not None:
        output.append((start, end))
    for start, end in output:
        body = text[start:end]
        eid = hashlib.sha256(f"{document_id}:{start}:{end}:{body}".encode()).hexdigest()[:24]
        yield EvidenceRecord(eid, document_id, body, start, end, structure)


class BM25:
    def __init__(self, records, k1=1.5, b=0.75):
        self.records = list(records)
        self.words = [Counter(tokenize(r.text)) for r in self.records]
        self.lengths = [sum(c.values()) for c in self.words]
        self.average = sum(self.lengths) / max(1, len(self.words))
        self.df = Counter(t for words in self.words for t in words)
        self.k1, self.b = k1, b

    def retrieve(self, question, max_chars=8000, top_k=5):
        if max_chars < 0 or top_k <= 0:
            raise ValueError("retrieval requires a finite context budget and positive top_k")
        scores = []
        for i, words in enumerate(self.words):
            score = 0.0
            for word in set(tokenize(question)):
                count = words[word]
                idf = math.log(1 + (len(self.words) - self.df[word] + .5) / (self.df[word] + .5))
                score += idf * count * (self.k1 + 1) / (count + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(1, self.average)))
            scores.append((score, i))
        selected, used, excluded = [], 0, []
        for score, i in sorted(scores, key=lambda pair: (-pair[0], self.records[pair[1]].evidence_id)):
            if score <= 0 or len(selected) >= top_k:
                continue
            record = self.records[i]
            if used + len(record.text) > max_chars:
                excluded.append(record.evidence_id)
                continue
            selected.append(record)
            used += len(record.text)
        return {"records": selected, "excluded_for_budget": excluded, "missing_evidence": not selected}


def deduplicate_reports(inputs):
    documents, mapping = {}, {}
    for item in inputs:
        text = "\n\n".join(item.context)
        key = hashlib.sha256(text.encode()).hexdigest()
        documents.setdefault(key, text)
        mapping[item.original_id] = key
    return documents, mapping
