import torch
from transformers import AutoModel, AutoTokenizer


class DenseRetriever:
    """Cache-only mean-pooled sentence-transformer encoder; fixed stronger option.

    The chosen encoder/revision and truncation policy must be frozen on development
    data. Oversized blocks are rejected, never silently truncated.
    """
    def __init__(self, model_id, revision, records, device="cpu", max_tokens=256):
        if len(revision) != 40:
            raise ValueError("dense retrieval requires an exact encoder revision")
        self.records, self.device, self.max_tokens = list(records), device, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True,trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model_id,revision=revision,local_files_only=True,trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)
        self.vectors = self.encode([r.text for r in self.records]) if self.records else None

    @torch.no_grad()
    def encode(self, texts):
        rows = self.tokenizer(texts,padding=True,truncation=False,return_tensors="pt")
        if rows.input_ids.shape[1] > self.max_tokens:
            raise ValueError("dense retrieval input exceeds locked token cap; rechunk explicitly")
        rows = rows.to(self.device)
        hidden = self.model(**rows).last_hidden_state
        weights = rows.attention_mask[...,None]
        pooled = (hidden*weights).sum(1)/weights.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled,p=2,dim=-1)

    def retrieve(self, question, max_chars=8000, top_k=5):
        if self.vectors is None:
            return {"records":[],"excluded_for_budget":[],"missing_evidence":True}
        scores = (self.vectors @ self.encode([question])[0]).tolist()
        selected, excluded, used = [], [], 0
        for index in sorted(range(len(scores)),key=lambda i:(-scores[i],self.records[i].evidence_id)):
            if len(selected) >= top_k:
                break
            record = self.records[index]
            if used+len(record.text) > max_chars:
                excluded.append(record.evidence_id)
                continue
            selected.append(record)
            used += len(record.text)
        return {"records":selected,"excluded_for_budget":excluded,"missing_evidence":not selected}
