from dataclasses import replace
import pytest
from ncm_lab.contracts import ModelInput, EvaluationGold
from ncm_lab.data.loaders import load_records, assert_disjoint, oasst_prompts, near_duplicate
from ncm_lab.evaluation.financial import execute_program, finqa_score, tatqa_score
from ncm_lab.runtime.io import root, read, digest
from ncm_lab.business.evidence import chunks, BM25
from ncm_lab.business.workflow import FinancialAnswer, validate, bounded_repair, workpapers, DevelopmentPolicy, RoutingFeatures


def test_released_training_examples_separate_gold_and_match_official():
    for dataset in ("finqa", "tatqa"):
        path = root() / f"tests/fixtures/{dataset}_train_sample.json"
        receipt = read(path.with_suffix(".provenance.json"))
        assert receipt["sha256"] == digest(path)
        for item,gold in load_records(path,dataset,"train"):
            prompt = item.prompt()
            assert '"program"' not in prompt and '"gold_inds"' not in prompt and '"answer"' not in prompt
            secret = replace(gold, answer="SECRET_GOLD", program="SECRET_PROGRAM", evidence_labels="SECRET_EVIDENCE")
            assert "SECRET" not in item.prompt() and secret.answer == "SECRET_GOLD"
            if dataset == "finqa":
                score = finqa_score(gold.program,gold,item.table)
                assert score["execution_accuracy"] == 1
                assert score["program_accuracy"] == 1
            else:
                assert tatqa_score(gold.answer,gold.scale,gold)["exact_match"] == 1


@pytest.mark.parametrize("program,expected", [("divide(20%, 2)",.1), ("multiply(-2, const_100), add(#0, 3)",-197),
    ("table_sum(revenue, none)",12), ("greater(4, 3)","yes"), ("divide(1, 3)",.33333)])
def test_allowlisted_dsl_agrees_with_official(program,expected):
    table = (("revenue","5","7"),)
    assert execute_program(program,table) == expected
    gold = EvaluationGold("trusted",expected,program)
    assert finqa_score(program,gold,table)["execution_accuracy"] == 1


@pytest.mark.parametrize("program", ["__import__(os, system)","add(__class__, 2)","exp(2, 999999)",
    "add(#0, 2)","divide(1, 0)","open(secret, none)","add(1, 2); import os", "add(1e9999, 1)"])
def test_malicious_or_unsupported_programs_fail(program):
    with pytest.raises(ValueError):
        execute_program(program)


def test_split_family_tree_and_duplicate_guards():
    a = ModelInput("a","finqa","train","filing","what was revenue?")
    with pytest.raises(ValueError,match="leakage"):
        assert_disjoint([a,replace(a,dataset="docfinqa",split="test",original_id="b")])
    rows = [{"lang":"en","role":"prompter","text":"question", "message_tree_id":"tree", "message_id":str(i)} for i in range(10)]
    assert len({r.split for r in oasst_prompts(rows)}) == 1
    assert len(list(oasst_prompts([{**rows[0],"deleted":True}]))) == 0
    assert near_duplicate("A question with four words", "A question with four words!")


def test_evidence_offsets_budget_workpaper_and_gold_free_repair():
    text = "Revenue table\nYear | 2022 | 2023\nRevenue | 5 | 7\n\nNotes on revenue growth."
    records = list(chunks("filing",text,max_chars=40,structure="released_table_text"))
    assert all(text[r.start:r.end] == r.text for r in records)
    assert "Year | 2022" in records[0].text and "Revenue | 5" in records[0].text
    retrieved = BM25(records).retrieve("revenue",max_chars=10)
    assert retrieved["excluded_for_budget"] and not retrieved["records"]
    answer = FinancialAnswer("q",12,"USD","million",None,[records[0].evidence_id],"add(5, 7)")
    assert validate(answer,records)["status"] == "valid"
    answer.answer = 13
    assert validate(answer,records)["status"] == "calculation_error"
    calls = []
    def repair(old,evidence,error):
        calls.append(error)
        old.answer = 12
        return old
    result = bounded_repair(answer,records,repair)
    assert len(calls) == 1 and result["outcome"] == "valid"
    items = [ModelInput(str(i),"finqa","train","same","q") for i in range(5)]
    assert [len(bundle) for bundle in workpapers(items,4)] == [4,1]


def test_policy_accepts_only_pre_generation_features_and_dev():
    features = RoutingFeatures(500,1,"answer")
    policy = DevelopmentPolicy(["AR","HIER_SKIP"])
    policy.fit([(features,"AR",2),(features,"HIER_SKIP",1)])
    assert policy.choose(features) == "HIER_SKIP"
    with pytest.raises(ValueError):
        policy.fit([],split="test")
    with pytest.raises(ValueError):
        policy.choose({"actual_future_output_length":99})
