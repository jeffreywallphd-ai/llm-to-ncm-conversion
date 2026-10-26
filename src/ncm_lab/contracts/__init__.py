"""Typed boundaries shared by loaders, deployment, and evaluators."""
from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    document_id: str
    text: str
    start: int
    end: int
    structure: str = "text_only"


@dataclass(frozen=True)
class ModelInput:
    original_id: str
    dataset: str
    split: str
    group_id: str
    question: str
    context: tuple[str, ...] = ()
    table: tuple[tuple[str, ...], ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()

    def prompt(self) -> str:
        # Explicit allowlist; a raw dataset row never enters a prompt.
        return json.dumps({"question": self.question, "context": self.context,
                           "table": self.table, "evidence": [asdict(e) for e in self.evidence]},
                          ensure_ascii=False)


@dataclass(frozen=True)
class EvaluationGold:
    original_id: str
    answer: Any
    program: str | None = None
    scale: str = ""
    evidence_labels: Any = None
    official: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingExample:
    input_ids: tuple[int, ...]
    group_id: str
    origin: str  # target_distillation or software_fixture
    split: str = "train"


@dataclass
class PredictionRecord:
    original_id: str
    group_id: str
    output_text: str
    correct: bool | None
    request_seconds: float
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_seconds: float | None = None
    generation_seconds: float | None = None
    accepted_tokens: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    verification_calls: int = 0
    retries: int = 0
    cache_policy: str = "none"
    null_reason: str = "component timing not collected by this executor"
