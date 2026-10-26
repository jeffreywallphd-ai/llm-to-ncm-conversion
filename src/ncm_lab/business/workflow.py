from collections import defaultdict
from dataclasses import dataclass, field
import math
import time
from ncm_lab.evaluation.financial import execute_program


@dataclass
class FinancialAnswer:
    question_id: str
    answer: object
    unit: str
    scale: str
    period: str | None
    source_refs: list[str]
    program: str | None = None
    status: str = "requires_review"
    checks: dict = field(default_factory=dict)


def validate(answer, evidence, table=()):
    """No gold argument exists. Structural validity does not prove claim support."""
    if not isinstance(answer, FinancialAnswer):
        return {"status": "missing_required_field", "reason": "output does not match FinancialAnswer"}
    if answer.answer is None or not answer.unit or not answer.source_refs:
        return {"status": "missing_required_field"}
    ids = {r.evidence_id for r in evidence}
    if not set(answer.source_refs) <= ids:
        return {"status": "unsupported_reference"}
    checks = {"reference_ids_valid": True, "source_support": "requires_independent_assessment",
              "period_and_unit": "requires_independent_assessment"}
    if answer.program:
        try:
            value = execute_program(answer.program, table)
            checks["program_executes"] = True
            try:
                consistent = math.isclose(float(value), float(answer.answer), rel_tol=0, abs_tol=1e-5)
            except (ValueError, TypeError):
                consistent = value == answer.answer
            checks["calculation_consistent"] = consistent
            if not consistent:
                return {"status": "calculation_error", **checks}
        except ValueError as exc:
            return {"status": "calculation_error", "reason": str(exc), **checks}
    return {"status": "valid", **checks}


def bounded_repair(answer, evidence, repair, table=(), max_attempts=1):
    if max_attempts not in (0, 1):
        raise ValueError("the protocol allows at most one repair")
    started = time.monotonic()
    check = validate(answer, evidence, table)
    attempts = []
    if check["status"] != "valid" and max_attempts:
        # Only deployment evidence, existing output, and detected error enter repair.
        answer = repair(answer, evidence, check)
        attempts.append(check)
        check = validate(answer, evidence, table)
    return {"answer": answer, "validation": check, "attempts": attempts,
            "elapsed_seconds": time.monotonic() - started,
            "outcome": check["status"] if check["status"] == "valid" else "requires_review"}


def workpapers(inputs, size):
    if size not in (1, 2, 4, 8):
        raise ValueError("workpaper size must be 1, 2, 4, or 8")
    groups = defaultdict(list)
    seen = set()
    for item in inputs:
        if item.original_id in seen:
            raise ValueError("duplicate original question")
        seen.add(item.original_id)
        groups[(item.dataset, item.split, item.group_id)].append(item)
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: r.original_id)
        for start in range(0, len(rows), size):
            yield rows[start:start + size]


@dataclass(frozen=True)
class RoutingFeatures:
    prompt_tokens: int
    question_count: int
    output_schema: str


class DevelopmentPolicy:
    """Empirical mean development latency per pre-generation feature bucket."""
    def __init__(self, eligible_methods):
        self.methods, self.routes = set(eligible_methods), {}
        if not self.methods:
            raise ValueError("routing needs at least one parity-validated method")

    @staticmethod
    def bucket(features):
        if not isinstance(features, RoutingFeatures):
            raise ValueError("routing accepts only typed pre-generation features")
        return (features.prompt_tokens // 1024, features.question_count, features.output_schema)

    def fit(self, rows, split="dev"):
        if split != "dev":
            raise ValueError("routing may be fitted on development records only")
        buckets = defaultdict(lambda: defaultdict(list))
        for features, method, seconds in rows:
            if method not in self.methods or seconds <= 0 or not math.isfinite(seconds):
                raise ValueError("invalid development routing observation")
            buckets[self.bucket(features)][method].append(seconds)
        for key, observations in buckets.items():
            self.routes[key] = min(observations, key=lambda m: (sum(observations[m]) / len(observations[m]), m))
        return self

    def choose(self, features):
        return self.routes.get(self.bucket(features), sorted(self.methods)[0])
