import importlib.util
import re
from pathlib import Path
import subprocess
import sys
from ncm_lab.runtime.io import digest


def gsm8k_score(output, gold):
    # Original answer delimiter; numeric extraction is explicit, not a financial tolerance.
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", output)
    reference = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", str(gold.answer))
    return {"exact_match": float(bool(match and reference and match[1].replace(",", "") == reference[1].replace(",", "")))}


def official_task_prerequisites(dataset):
    requirements = {"ifeval": "instruction_following_eval", "humaneval_plus": "evalplus", "longbench": "rouge"}
    if dataset not in requirements:
        raise ValueError(f"unknown task evaluator: {dataset}")
    module = requirements[dataset]
    if importlib.util.find_spec(module) is None:
        raise ValueError(f"install the pinned {module} evaluator in a separate evaluation environment; see data manifest")
    if dataset == "humaneval_plus":
        raise ValueError("HumanEval+ execution blocked until a verified network-disabled bounded sandbox is configured")
    return module


def official_scorer(dataset, evaluator):
    """Invoke inspected official APIs from a hash-verified, explicitly provisioned checkout."""
    if dataset == "humaneval_plus":
        raise ValueError("generated Python execution requires a verified external isolation boundary")
    directory = Path(evaluator["root"]).resolve()
    actual = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if actual != evaluator["revision"] or len(actual) != 40:
        raise ValueError("official evaluator checkout revision mismatch")
    files = evaluator["files_sha256"]
    if not files:
        raise ValueError("official evaluator file hashes are required")
    for name, expected in files.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError("official evaluator source changed")
    if dataset == "ifeval":
        required={'instruction_following_eval/'+name+'.py' for name in
                  ('evaluation_lib','instructions','instructions_registry','instructions_util')}
        if not required<=files.keys():
            raise ValueError("IFEval scorer dependency hashes missing")
        for name,module in list(sys.modules.items()):
            if name.startswith('instruction_following_eval') and getattr(module,'__file__',None):
                if not Path(module.__file__).resolve().is_relative_to(directory):
                    raise ValueError('IFEval imported from a different evaluator checkout; use a fresh process')
        sys.path.insert(0, str(directory))
        from instruction_following_eval import evaluation_lib
        def score(text, gold, item):
            row = gold.official
            example = evaluation_lib.InputExample(key=row["key"], prompt=row["prompt"],
                instruction_id_list=row["instruction_id_list"], kwargs=row["kwargs"])
            strict = evaluation_lib.test_instruction_following_strict(example, {example.prompt: text})
            loose = evaluation_lib.test_instruction_following_loose(example, {example.prompt: text})
            return {"prompt_strict_accuracy": float(strict.follow_all_instructions),
                    "prompt_loose_accuracy": float(loose.follow_all_instructions),
                    "strict_instruction_correct": sum(strict.follow_instruction_list),
                    "loose_instruction_correct": sum(loose.follow_instruction_list),
                    "instruction_count": len(strict.follow_instruction_list)}
        return score
    if dataset == "longbench":
        if "LongBench/metrics.py" not in files:
            raise ValueError("LongBench metrics.py hash missing")
        spec = importlib.util.spec_from_file_location("ncm_official_longbench_metrics",directory/"LongBench/metrics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def score(text, gold, item):
            if gold.official["task"] != "gov_report":
                raise ValueError("locked LongBench adapter currently supports official GovReport ROUGE-L")
            return {"rouge_l": max(module.rouge_score(text, answer) for answer in gold.answer)}
        return score
    raise ValueError("unknown official task")


def official_prompt(dataset, item, evaluator):
    if dataset == "ifeval":
        return item.question
    if dataset == "longbench":
        from ncm_lab.runtime.io import read
        name = "LongBench/config/dataset2prompt.json"
        path = Path(evaluator["root"])/name
        if digest(path) != evaluator["files_sha256"].get(name):
            raise ValueError("official LongBench prompt-template identity missing or changed")
        return read(path)["gov_report"].format(context="\n".join(item.context),input=item.question)
    raise ValueError("no official prompt adapter")
