import importlib.util
import re


def gsm8k_score(output, gold):
    # Original answer delimiter; numeric extraction is explicit, not a financial tolerance.
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", output)
    reference = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", str(gold.answer))
    return {"exact_match": float(bool(match and reference and match[1].replace(",", "") == reference[1].replace(",", "")))}


def official_task_prerequisites(dataset):
    requirements = {"ifeval": "instruction_following_eval", "humaneval_plus": "evalplus", "longbench": "rouge_score"}
    if dataset not in requirements:
        raise ValueError(f"unknown task evaluator: {dataset}")
    module = requirements[dataset]
    if importlib.util.find_spec(module) is None:
        raise ValueError(f"install the pinned {module} evaluator in a separate evaluation environment; see data manifest")
    if dataset == "humaneval_plus":
        raise ValueError("HumanEval+ execution blocked until a verified network-disabled bounded sandbox is configured")
    return module
