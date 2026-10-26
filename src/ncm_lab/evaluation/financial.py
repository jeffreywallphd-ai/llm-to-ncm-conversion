import math
import re
from ncm_lab.vendor.finqa import evaluate as official_finqa

OPS = set(official_finqa.all_ops)
STEP = re.compile(r"\s*([a-z_]+)\(\s*([^(),]+)\s*,\s*([^(),]+)\s*\)\s*(?:,\s*|$)")
NUMBER = re.compile(r"(?:[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)%?|const_(?:m1|\d+(?:_\d+)?))\Z")


def parse_program(program, table=()):
    if not isinstance(program, str) or len(program) > 8192:
        raise ValueError("program must be a bounded DSL string")
    pos, steps, table_rows = 0, [], {str(r[0]) for r in table if r}
    while pos < len(program):
        m = STEP.match(program, pos)
        if not m or len(steps) >= 32:
            raise ValueError("malformed or oversized FinQA DSL program")
        op, a, b = (s.strip() for s in m.groups())
        if op not in OPS:
            raise ValueError(f"unsupported DSL operator: {op}")
        for arg in (a, b):
            if len(arg) > 256:
                raise ValueError("oversized DSL argument")
            if arg.startswith("#"):
                if not arg[1:].isdigit() or int(arg[1:]) >= len(steps):
                    raise ValueError("reference must point to an earlier step")
            elif arg not in {"none", *table_rows} and not NUMBER.fullmatch(arg):
                raise ValueError("argument is neither a numeric literal, table row, nor previous-step reference")
            elif NUMBER.fullmatch(arg):
                try:
                    value = official_finqa.str_to_num(arg)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e15:
                        raise ValueError("numeric literal exceeds interpreter bounds")
                except (OverflowError, ValueError) as exc:
                    raise ValueError("unsupported numeric literal") from exc
        steps.append((op, a, b))
        pos = m.end()
    if not steps:
        raise ValueError("empty DSL program")
    return steps


def execute_program(program, table=()):
    steps = parse_program(program, table)
    results = []
    rows = {str(r[0]): list(r[1:]) for r in table if r}

    def num(arg):
        value = results[int(arg[1:])] if arg.startswith("#") else official_finqa.str_to_num(arg)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e30:
            raise ValueError("non-numeric or oversized operand")
        return value

    for op, a, b in steps:
        if op.startswith("table_"):
            if a not in rows or b != "none":
                raise ValueError("table operation needs a named released row and none")
            values = official_finqa.process_row(rows[a])
            if values == "n/a" or not values or len(values) > 1024:
                raise ValueError("unsupported table row")
            value = {"table_max": max, "table_min": min, "table_sum": sum,
                     "table_average": lambda xs: sum(xs) / len(xs)}[op](values)
        else:
            x, y = num(a), num(b)
            if op == "exp" and abs(y) > 8:
                raise ValueError("exponent exceeds bounded interpreter support")
            try:
                value = {"add": lambda: x + y, "subtract": lambda: x - y,
                         "multiply": lambda: x * y, "divide": lambda: x / y,
                         "exp": lambda: x ** y, "greater": lambda: "yes" if x > y else "no"}[op]()
            except (ArithmeticError, OverflowError) as exc:
                raise ValueError("invalid arithmetic") from exc
        if not isinstance(value, str) and (not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e30):
            raise ValueError("calculation exceeds numeric bounds")
        results.append(value)
    return results[-1] if isinstance(results[-1], str) else round(results[-1], 5)


def finqa_score(program, gold, table):
    """Official 5-decimal execution and symbolic program equivalence; no new tolerance."""
    try:
        result = execute_program(program, table)
        tokens = official_finqa.program_tokenization(program)
        invalid, upstream_result = official_finqa.eval_program(tokens, table)
        if invalid or result != upstream_result:
            raise ValueError("bounded interpreter and pinned evaluator disagree")
        gold_tokens = official_finqa.program_tokenization(gold.program)
        parse_program(gold.program, table)
        program_correct = bool(official_finqa.equal_program(gold_tokens, tokens))
        return {"execution_accuracy": float(result == gold.answer), "program_accuracy": float(program_correct),
                "parse_failure": False, "result": result}
    except (ValueError, ArithmeticError, KeyError, AssertionError):
        return {"execution_accuracy": 0.0, "program_accuracy": 0.0, "parse_failure": True, "result": None}


def tatqa_score(answer, scale, gold):
    from ncm_lab.vendor.tatqa.tatqa_metric import TaTQAEmAndF1
    metric = TaTQAEmAndF1()
    metric(gold.official, answer, scale)
    em, f1, scale_accuracy, _ = metric.get_overall_metric()
    return {"exact_match": em, "f1": f1, "scale_accuracy": scale_accuracy}
