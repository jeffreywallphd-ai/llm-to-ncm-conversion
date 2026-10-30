"""Pinned CPU heuristic; code/math is removed from language evidence, never rewritten."""

import hashlib
import re
from importlib.metadata import version


class EnglishFilter:
    def __init__(self, policy):
        from langid.langid import LanguageIdentifier, model

        if version("langid") != "1.1.6":
            raise ValueError("language policy requires langid==1.1.6")
        self.identity = {
            "package": "langid",
            "version": version("langid"),
            "embedded_model_sha256": hashlib.sha256(model).hexdigest(),
        }
        if policy["model_sha256"] != self.identity["embedded_model_sha256"]:
            raise ValueError("pinned language classifier model changed")
        self.classifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.policy = policy

    def evidence(self, text):
        text = re.sub(r"```[\s\S]*?```|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]", " ", text)
        text = re.sub(r"`[^`]*`|\$[^$\n]*\$|https?://\S+", " ", text)
        lines = [
            line
            for line in text.splitlines()
            if (
                len(re.findall(r"[^\W\d_]{2,}", line, re.UNICODE)) >= 4
                or sum(ch.isalpha() for ch in line) >= 24
            )
            and not re.search(r"^\s*(def |class |import |from |#include|[{}]|\\)", line)
        ]
        return "\n".join(lines)[: self.policy["max_evidence_chars"]]

    def admit(self, context, answer):
        prompt = self.evidence("\n".join(m["content"] for m in context if m["role"] == "user"))
        target = self.evidence(answer)
        structured_target = bool(re.search(r"```|\$|\\\[|[=+*/<>]", answer)) or not any(
            ch.isalpha() for ch in answer
        )
        if len(target) < self.policy["min_evidence_chars"] and not structured_target:
            return "language_uncertain"
        evidence = [part for part in (prompt, target) if len(part) >= self.policy["min_evidence_chars"]]
        if not evidence:
            return "language_uncertain"
        for text in evidence:
            language, probability = self.classifier.classify(text)
            if language != "en":
                return "language_non_english"
            if probability < self.policy["minimum_probability"]:
                return "language_uncertain"
        return None
