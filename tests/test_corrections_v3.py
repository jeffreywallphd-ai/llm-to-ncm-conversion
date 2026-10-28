"""Scientific/execution contracts for CORR-01..14; fixtures are never research evidence."""

import copy
import json
import shutil
import uuid
import time
from pathlib import Path
import pytest
import torch
from ncm_lab.runtime.io import root, read, write_json, digest
from ncm_lab.models.factory import resolve_method, build_drafter
from ncm_lab.models.codecs import MultiscaleCodec, pool
from ncm_lab.experiments.configuration import resolve_config
from ncm_lab.experiments.runner import run
from ncm_lab.training.checkpoints import atomic_checkpoint, read_checkpoint


@pytest.fixture(scope="module")
def teacher_artifacts(tmp_path_factory):
    from ncm_lab.models.fixtures import save_tiny_artifacts
    from ncm_lab.data.artifacts import lock_input

    directory = tmp_path_factory.mktemp("v3-production-teachers")
    torch.set_num_threads(1)
    model = save_tiny_artifacts(directory / "models")
    budget = {
        **read(root() / "configs/budgets/acceptance.yaml"),
        "phase_budget_id": "v3-tests-" + uuid.uuid4().hex[:10],
        "max_steps": 32,
        "max_wall_seconds": 120,
    }
    budget_path = directory / "budget.json"
    write_json(budget_path, budget)
    common = {
        "profile": "cpu",
        "device": "cpu",
        "precision": "float32",
        "result_origin": "test_fixture",
        "model": model,
        "artifact_root": str(directory / "artifacts"),
        "seed": 17,
        "init_seed": 17,
        "data_seed": 17,
        "cpu_threads": 1,
    }
    outputs = {}
    for split in ("train", "dev"):
        preparation_started = time.perf_counter()
        path = directory / (split + ".jsonl")
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "question": split + str(i),
                        "original_id": split + str(i),
                        "group_id": split + str(i),
                        "split": split,
                    }
                )
                + "\n"
                for i in range(2)
            ),
            encoding="utf-8",
        )
        receipt = directory / (split + ".input.json")
        lock_input(path, receipt, "prompts", split, "fixture", fixture=True)
        prep = directory / (split + ".preparation.json")
        elapsed = time.perf_counter() - preparation_started
        write_json(
            prep,
            {
                "status": "completed",
                "result_origin": "test_fixture",
                "training_started": False,
                "teacher_generation_started": False,
                "resources": {
                    "preprocessing_wall_seconds": elapsed,
                    "preparation_phase_wall_seconds": elapsed,
                    "charged_download_bytes": 0,
                },
            },
        )
        prep_reference = directory / (split + ".preparation.reference.json")
        write_json(prep_reference, {"manifest": str(prep), "sha256": digest(prep)})
        config = {
            **common,
            "experiment_id": "SHR-001",
            "executor": "teacher",
            "purpose": split,
            "prompts_path": str(path),
            "input_receipts": [str(receipt)],
            "max_examples": 2,
            "max_new_tokens": 12,
            "teacher_output_reference": str(directory / (split + ".reference.json")),
            "preparation_reference": str(prep_reference),
        }
        config_path = directory / (split + ".config.json")
        write_json(config_path, config)
        result = run("SHR-001", config_path, budget_path)
        assert result["status"] == "completed", result
        outputs[split] = Path(result["path"])
    return {
        "root": directory,
        "common": common,
        "budget": budget_path,
        **outputs,
        "data": {
            "training_data": str(outputs["train"] / "teacher.jsonl"),
            "development_data": str(outputs["dev"] / "teacher.jsonl"),
            "input_receipts": [str(outputs[s] / "teacher.input.json") for s in ("train", "dev")],
        },
    }


@pytest.fixture(scope="module")
def fitted_paths(teacher_artifacts):
    fixture = teacher_artifacts
    outputs = {}
    for name, kind, weight in [
        ("prediction", "nested", 0.1),
        ("reconstruction", "nested", 0),
        ("linear", "linear", 0.1),
    ]:
        config = {
            **fixture["common"],
            **fixture["data"],
            "experiment_id": "ACL-005",
            "executor": "codec",
            "steps": 1,
            "latent_width": 4,
            "codec_kind": kind,
            "prediction_weights": [weight],
            "codec_seed": 17,
            "data_tier": 22,
            "codec_objective": "reconstruction_only" if weight == 0 else "prediction_aware",
        }
        path = fixture["root"] / (name + ".codec.config.json")
        write_json(path, config)
        result = run("ACL-005", path, fixture["budget"])
        assert result["status"] == "completed", result
        outputs[name] = Path(result["path"])
    config = {
        **fixture["common"],
        **fixture["data"],
        "experiment_id": "ACL-005",
        "executor": "train",
        "method": "HIER_MULTI_AE",
        "data_tier": 22,
        "steps": 1,
        "codec_objective": "prediction_aware",
        "codec_selection": str(outputs["prediction"] / "codec-selection.json"),
        "codec_sha256": digest(outputs["prediction"] / "codec-selected.pt"),
    }
    path = fixture["root"] / "student.config.json"
    write_json(path, config)
    result = run("ACL-005", path, fixture["budget"])
    assert result["status"] == "completed", result
    outputs["student"] = Path(result["path"])
    return outputs


def test_actual_acl003_config_has_valid_warmup_scoring_policy(
    teacher_artifacts, fitted_paths, tmp_path, monkeypatch
):
    from ncm_lab.experiments.study import materialize
    from ncm_lab.data.artifacts import lock_input
    from transformers import AutoTokenizer

    spec, _ = materialize(root() / "experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml")
    original = next(
        j
        for j in spec["jobs"]
        if j["config"]["experiment_id"] == "ACL-003"
        and j["config"]["executor"] == "evaluate"
        and j["config"]["method"] == "HIER_MULTI_AE"
    )
    resolved = resolve_config(original["config"], planning=True, job_id=original["id"])
    invalid = copy.deepcopy(original["config"])
    invalid.pop("output_instruction")
    with pytest.raises(ValueError, match=original["id"] + ".*output_instruction"):
        resolve_config(invalid, planning=True, job_id=original["id"])
    corpus = tmp_path / "gsm.jsonl"
    corpus.write_text(
        json.dumps({"question": "count the apples", "answer": "EVALUATOR_SECRET #### 1"}) + "\n"
    )
    receipt = tmp_path / "gsm.input.json"
    lock_input(corpus, receipt, "gsm8k", "train", "fixture", fixture=True)
    captures = []
    tokenizer_loader = AutoTokenizer.from_pretrained

    class ObservedTokenizer:
        def __init__(self, value):
            self.value = value

        def __getattr__(self, key):
            return getattr(self.value, key)

        def __call__(self, *args, **kwargs):
            return self.value(*args, **kwargs)

        def apply_chat_template(self, messages, *args, **kwargs):
            captures.append(copy.deepcopy(messages))
            return self.value.apply_chat_template(messages, *args, **kwargs)

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: ObservedTokenizer(tokenizer_loader(*args, **kwargs)),
    )
    student = fitted_paths["student"]
    codec = fitted_paths["prediction"]
    config = {
        **resolved,
        **teacher_artifacts["common"],
        "data_path": str(corpus),
        "input_receipts": [str(receipt)],
        "data_tier": 22,
        "max_requests": 1,
        "max_new_tokens": 12,
        "warmup_requests": 1,
        "require_gpu_parity": False,
        "checkpoint": str(student / "selected.pt"),
        "checkpoint_sha256": digest(student / "selected.pt"),
        "checkpoint_selection": str(student / "selection.json"),
        "codec_selection": str(codec / "codec-selection.json"),
        "codec_sha256": digest(codec / "codec-selected.pt"),
    }
    config.pop("parity_run", None)
    config.pop("overlap_audit", None)
    path = tmp_path / "evaluation.json"
    write_json(path, config)
    result = run("ACL-003", path, teacher_artifacts["budget"])
    assert result["status"] == "completed", result
    observations = [
        json.loads(line) for line in (Path(result["path"]) / "predictions.jsonl").read_text().splitlines()
    ]
    assert captures[0] == captures[-1] and "EVALUATOR_SECRET" not in json.dumps(captures)
    assert original["config"]["output_instruction"] in captures[0][0]["content"]
    assert len(read(Path(result["path"]) / "warmup.json")) == 1
    assert (
        observations[0]["identities"]["condition"]["prompt_policy"]
        == read(Path(result["path"]) / "config.json")["prompt_policy"]
    )


def test_two_tier_analysis_is_order_invariant_and_rejects_duplicates():
    from ncm_lab.evaluation.tables import analyze_records

    rows = []
    for tier, a, b in [(250000, 1, 2), (1000000, 10, 5)]:
        for request in range(3):
            for method, seconds, checkpoint in [("A", a, "checkpoint-A"), ("B", b, "checkpoint-B")]:
                rows.append(
                    {
                        "method": method,
                        "training_seed": 17,
                        "data_tier": tier,
                        "split": "test",
                        "original_id": str(request),
                        "group_id": str(request),
                        "repetition_id": 0,
                        "request_seconds": seconds,
                        "checkpoint_sha256": checkpoint,
                        "codec_sha256": "codec-" + method,
                        "correct": True,
                    }
                )
    forward = analyze_records(rows)
    reverse = analyze_records(list(reversed(rows)))
    assert forward == reverse
    assert {r["condition"]["data_tier"]: r["mean"] for r in forward["paired_latency_differences"]} == {
        250000: 1,
        1000000: -5,
    }
    with pytest.raises(ValueError, match="duplicate observation"):
        analyze_records(rows + [{**rows[0], "request_seconds": 99}])
    assert (
        len(
            analyze_records(rows + [{**rows[0], "checkpoint_sha256": "other-selected-checkpoint"}])[
                "per_seed"
            ]
        )
        == 5
    )
    pilot = [{**row, "split": "train", "result_scope": "pilot"} for row in rows]
    assert len(analyze_records(rows + pilot)["paired_latency_differences"]) == 4
    missing = analyze_records(rows[:-1])
    assert missing["unmatched_observations"]


def test_independent_encoder_nonlinearity_precedes_pooling():
    states = torch.zeros(1, 15, 1)
    altered = states.clone()
    altered[0, 0, 0] = -1
    altered[0, 1, 0] = 1
    valid = torch.ones(1, 15, dtype=torch.bool)
    codec = MultiscaleCodec(1, 1, "independent")
    old = MultiscaleCodec(1, 1, "independent", architecture_version=2)
    shapes = []
    codec.independent["4"].register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    for model in (codec, old):
        for projection in model.independent.values():
            for layer in projection:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.ones_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    assert torch.equal(pool(states, 4, valid)[0], pool(altered, 4, valid)[0])
    assert not torch.equal(codec.encode(states, valid)[4], codec.encode(altered, valid)[4])
    assert torch.equal(old.encode(states, valid)[4], old.encode(altered, valid)[4])
    before = codec.encode(altered, valid)[4].clone()
    with torch.no_grad():
        for parameter in codec.independent["8"].parameters():
            parameter.add_(2)
    assert torch.equal(before, codec.encode(altered, valid)[4])
    assert all(shape == (1, 15, 1) for shape in shapes)
    assert codec.spec["architecture_version"] == 3 and old.spec["architecture_version"] == 2


@pytest.mark.parametrize(
    "boundary", ["weights_finished", "before_receipt", "receipt_written", "before_commit", "after_commit"]
)
@pytest.mark.parametrize("prior", [False, True])
def test_checkpoint_publication_boundary_is_previous_or_new(tmp_path, monkeypatch, boundary, prior):
    import ncm_lab.training.checkpoints as checkpoints

    path = tmp_path / "latest.pt"
    if prior:
        atomic_checkpoint(path, {"identity": "same", "model": {"weight": torch.tensor([0.0])}})

    def fail(stage):
        if stage == boundary:
            raise KeyboardInterrupt("injected publication boundary")

    monkeypatch.setattr(checkpoints, "publication_boundary", fail)
    with pytest.raises(KeyboardInterrupt):
        atomic_checkpoint(path, {"identity": "same", "model": {"weight": torch.tensor([1.0])}})
    if not prior and boundary != "after_commit":
        with pytest.raises(ValueError, match="no committed checkpoint"):
            read_checkpoint(path)
    else:
        loaded = read_checkpoint(path)
        assert loaded["model"]["weight"].item() == (1 if boundary == "after_commit" else 0)
    assert not list(tmp_path.glob("*.partial"))


def test_typed_official_endpoints_preserve_missing_zero_and_continuous():
    from ncm_lab.evaluation.endpoints import endpoint_result
    from ncm_lab.evaluation.analysis import summarize
    from ncm_lab.evaluation.tables import analyze_records

    strict = endpoint_result("ifeval", {"prompt_strict_accuracy": 1.0})
    loose = endpoint_result("ifeval", {"prompt_strict_accuracy": 0.0, "prompt_loose_accuracy": 1.0})
    missing = endpoint_result("ifeval", {})
    continuous = endpoint_result("longbench", {"rouge_l": 0.4})
    assert strict["correct"] is True and loose["correct"] is False and missing["status"] == "missing"
    assert continuous["correct"] is None and continuous["value"] == 0.4
    assert (
        summarize([{"correct": None, "request_seconds": 1, "endpoint": continuous}])[
            "correct_requests_per_hour"
        ]
        is None
    )
    rows = []
    for i, count, correct in [(0, 1, 1), (1, 9, 0)]:
        rows.append(
            {
                "original_id": str(i),
                "group_id": str(i),
                "method": "m",
                "request_seconds": 1,
                "correct": False,
                "official_scores": {"strict_instruction_correct": correct, "instruction_count": count},
            }
        )
    assert analyze_records(rows)["per_seed"][0]["instruction_strict_accuracy"] == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("learned_expansion", False),
        ("runtime_latent", False),
        ("context_policy", "mean_confirmed_features"),
        ("latent_supervision", False),
        ("runtime_latentt", True),
        ("runtime_latent", 1),
        ("skip", 1),
        ("initial_span", "true"),
        ("skip_rank", True),
    ],
)
def test_scientific_contradictions_fail(field, value):
    with pytest.raises(ValueError, match=field):
        resolve_method({"method": "HIER_MULTI_AE", "codec_sha256": "selected", field: value})
    matching = resolve_method(
        {
            "method": "HIER_MULTI_AE",
            "codec_sha256": "selected",
            "learned_expansion": True,
            "runtime_latent": True,
            "latent_supervision": True,
            "context_policy": "all_confirmed_features",
        }
    )
    assert matching.runtime_latent and matching.latent_supervision


def test_declared_data_semantics_cannot_be_silently_ignored():
    config = {
        "experiment_id": "ACL-005",
        "executor": "train",
        "result_origin": "test_fixture",
        "device": "cpu",
        "precision": "float32",
        "method": "HIER_SKIP",
        "training_data": "train.jsonl",
        "development_data": "dev.jsonl",
    }
    with pytest.raises(ValueError, match="data_ladder_version"):
        resolve_config({**config, "data_ladder_version": 1}, planning=True)
    with pytest.raises(ValueError, match="fixed_codec_transfer"):
        resolve_config({**config, "fixed_codec_transfer": True}, planning=True)
    with pytest.raises(ValueError, match="skip_rank"):
        resolve_method({"method": "HIER_NO_SKIP", "skip_rank": 7})
    assert resolve_config({**config, "data_ladder_version": 3}, planning=True)


def test_profile_resolves_actual_width_and_each_selected_codec(
    teacher_artifacts, fitted_paths, tmp_path, monkeypatch
):
    import ncm_lab.experiments.executors as executors

    captured = {}
    builder = executors.build_drafter

    def observe(target, backbone, method, *args, **kwargs):
        model = builder(target, backbone, method, *args, **kwargs)
        captured[method.name] = model
        return model

    monkeypatch.setattr(executors, "build_drafter", observe)
    methods = ["HIER_LATENT_CE_ONLY", "HIER_MULTI_AE", "HIER_LINEAR_CODEC"]
    overrides = {"HIER_LATENT_CE_ONLY": {"latent_width": 7}}
    for method, label in [("HIER_MULTI_AE", "prediction"), ("HIER_LINEAR_CODEC", "linear")]:
        parent = fitted_paths[label]
        overrides[method] = {
            "codec_selection": str(parent / "codec-selection.json"),
            "codec_sha256": digest(parent / "codec-selected.pt"),
        }
    config = {
        **teacher_artifacts["common"],
        "experiment_id": "ACL-001",
        "executor": "profile",
        "methods": methods,
        "method_overrides": overrides,
        "prefix_lengths": [4],
        "concurrency": [1],
        "repetitions": 1,
    }
    path = tmp_path / "profile.json"
    write_json(path, config)
    result = run("ACL-001", path, teacher_artifacts["budget"])
    assert result["status"] == "completed", result
    assert captured["HIER_LATENT_CE_ONLY"].latent.predictors["4"].out_features == 7
    metadata = read(Path(result["path"]) / "resolved_methods.json")
    assert metadata["HIER_LATENT_CE_ONLY"]["method"]["latent_width"] == 7
    assert metadata["HIER_MULTI_AE"]["codec_dependencies"]["spec"]["kind"] == "nested"
    assert metadata["HIER_LINEAR_CODEC"]["codec_dependencies"]["spec"]["kind"] == "linear"
    invalid = copy.deepcopy(config)
    invalid["method_overrides"]["HIER_MULTI_AE"]["latent_width"] = 7
    with pytest.raises(ValueError, match="latent_width"):
        resolve_config(invalid, planning=True, job_id="profile-invalid")
    assert not read(Path(result["path"]) / "metrics.json")["performance_claim_eligible"]


@pytest.mark.parametrize(
    "corruption", ["teacher_revision", "tokenizer_revision", "template", "policy_digest", "semantic_policy"]
)
def test_teacher_admission_rejects_incompatible_source_before_model(
    teacher_artifacts, tmp_path, monkeypatch, corruption
):
    from ncm_lab.data.artifacts import lock_input
    from ncm_lab.training.source_contracts import policy_record
    import ncm_lab.experiments.executors as executors

    rows = [
        json.loads(line) for line in (teacher_artifacts["train"] / "teacher.jsonl").read_text().splitlines()
    ]
    if corruption in {"teacher_revision", "tokenizer_revision"}:
        rows[0][corruption] = "wrong-same-width-source"
    elif corruption == "policy_digest":
        rows[0]["policy_hash"] = "invalid"
    else:
        if corruption == "template":
            rows[0]["generation_policy"]["template_hash"] = "different-template"
        else:
            rows[0]["generation_policy"]["eos_ids"] = [31]
        rows[0]["policy_hash"] = policy_record(rows[0]["generation_policy"])[1]
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract = read(teacher_artifacts["train"] / "teacher.contract.json")
    contract["data_sha256"] = digest(path)
    write_json(path.with_suffix(".contract.json"), contract)
    receipt = tmp_path / "input.json"
    lock_input(path, receipt, "teacher", "train", "fixture", fixture=True)
    config = {
        **teacher_artifacts["common"],
        **teacher_artifacts["data"],
        "experiment_id": "ACL-005",
        "executor": "train",
        "method": "HIER_SKIP",
        "training_data": str(path),
        "input_receipts": [str(receipt), teacher_artifacts["data"]["input_receipts"][1]],
        "steps": 1,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("model loaded before incompatible source was rejected")

    monkeypatch.setattr(executors, "construct", forbidden)
    config_path = tmp_path / "train.json"
    write_json(config_path, config)
    result = run("ACL-005", config_path, teacher_artifacts["budget"])
    assert result["status"] == "failed" and "model loaded" not in result["error"]
    assert any(term in result["error"] for term in ["source mismatch", "policy hash", "template/policy"])


def test_codec_contract_relocation_conversion_and_wrong_target_rejection(
    teacher_artifacts, fitted_paths, tmp_path
):
    from ncm_lab.training.codec import load_selected_codec

    source = fitted_paths["prediction"]
    for name in ("codec-selection.json", "codec-selected.pt"):
        shutil.copyfile(source / name, tmp_path / name)
    codec = load_selected_codec(
        tmp_path / "codec-selection.json",
        dtype=torch.bfloat16,
        expected_model={
            **teacher_artifacts["common"]["model"],
            "target_path": "relocated/unchanged/artifact",
        },
        diagnostic=True,
    )
    assert next(codec.parameters()).dtype == torch.bfloat16 and codec.objective == "prediction_aware"
    wrong = {**teacher_artifacts["common"]["model"], "target_revision": "1" * 40}
    with pytest.raises(ValueError, match="source identity"):
        load_selected_codec(tmp_path / "codec-selection.json", expected_model=wrong, diagnostic=True)


def test_codec_objective_zero_cannot_be_positive_prediction_aware(teacher_artifacts, fitted_paths):
    from ncm_lab.training.codec import load_selected_codec
    from ncm_lab.models.upstream import load_pair

    codec = load_selected_codec(fitted_paths["reconstruction"] / "codec-selection.json")
    target, backbone, _ = load_pair(teacher_artifacts["common"]["model"])
    before = {key: parameter.detach().clone() for key, parameter in target.named_parameters()}
    method = resolve_method(
        {
            "method": "HIER_MULTI_AE",
            "codec_sha256": digest(fitted_paths["reconstruction"] / "codec-selected.pt"),
            "codec_objective": "prediction_aware",
        }
    )
    with pytest.raises(ValueError, match="objective"):
        build_drafter(target, backbone, method, codec)
    assert all(torch.equal(parameter, before[key]) for key, parameter in target.named_parameters())


def test_two_tiers_objectives_equal_budget_and_locked_final_analysis(teacher_artifacts, tmp_path):
    from ncm_lab.experiments.matrix import resolve_plan, launch_plan
    from ncm_lab.data.artifacts import lock_input
    from ncm_lab.reporting.analysis_runs import analyze

    corpus = tmp_path / "test.jsonl"
    corpus.write_text(
        "".join(json.dumps({"question": "count " + str(i), "answer": "#### 1"}) + "\n" for i in range(2))
    )
    receipt = tmp_path / "test.input.json"
    lock_input(corpus, receipt, "gsm8k", "test", "fixture", fixture=True)
    common = teacher_artifacts["common"]
    data = teacher_artifacts["data"]
    budget = str(teacher_artifacts["budget"])
    jobs = []
    contrasts = []
    for tier in (11, 22):
        roles = {regime: {} for regime in ("equal_data", "equal_total_training_wall")}
        for objective, weight in [("reconstruction_only", 0), ("prediction_aware", 0.1)]:
            label = f"t{tier}-{objective}"
            jobs.append(
                {
                    "id": "codec-" + label,
                    "config": {
                        **common,
                        **data,
                        "experiment_id": "ACL-005",
                        "executor": "codec",
                        "data_tier": tier,
                        "codec_seed": 17,
                        "steps": 1,
                        "latent_width": 4,
                        "codec_objective": objective,
                        "prediction_weights": [weight],
                    },
                    "budget": budget,
                    "depends_on": [],
                }
            )
            for regime in roles:
                trained = "student-" + label + "-" + regime
                extra = {"comparison": regime}
                if regime == "equal_total_training_wall":
                    extra.update(
                        total_new_training_wall_seconds=20,
                        teacher_cost_source="consumed_teacher_artifacts",
                        budget_endpoint_policy="best_completed_dev_before_budget",
                    )
                transfer = {
                    **common,
                    **data,
                    **extra,
                    "experiment_id": "ACL-005",
                    "executor": "train",
                    "method": "HIER_MULTI_AE",
                    "data_tier": tier,
                    "codec_objective": objective,
                    "steps": 1,
                    "codec_selection": "@codec-" + label + "/codec-selection.json",
                    "codec_sha256": "@codec-" + label + "/codec-selected.pt#sha256",
                }
                jobs.append(
                    {"id": trained, "config": transfer, "budget": budget, "depends_on": ["codec-" + label]}
                )
                checked = {
                    k: v
                    for k, v in transfer.items()
                    if k not in {"training_data", "development_data", "input_receipts", "steps"}
                }
                checked.update(
                    experiment_id="SHR-002",
                    executor="verification",
                    dataset="gsm8k",
                    split="test",
                    data_path=str(corpus),
                    input_receipts=[str(receipt)],
                    max_requests=2,
                    max_new_tokens=12,
                    output_instruction="Finish with #### and the number.",
                    checkpoint="@" + trained + "/selected.pt",
                    checkpoint_sha256="@" + trained + "/selected.pt#sha256",
                    checkpoint_selection="@" + trained + "/selection.json",
                )
                parity = "parity-" + label + "-" + regime
                jobs.append({"id": parity, "config": checked, "budget": budget, "depends_on": [trained]})
                final = "final-" + label + "-" + regime
                jobs.append(
                    {
                        "id": final,
                        "config": {
                            **checked,
                            "experiment_id": "ACL-003",
                            "executor": "evaluate",
                            "full_official_split": True,
                            "warmup_requests": 1,
                        },
                        "budget": budget,
                        "depends_on": [parity],
                    }
                )
                roles[regime]["control" if weight == 0 else "treatment"] = [final]
        for regime, role_map in roles.items():
            contrasts.append(
                {
                    "id": f"objective-t{tier}-{regime}",
                    "roles": role_map,
                    "seed_pair": {"codec": 17, "student": 17},
                    "condition": {"data_tier": tier, "comparison": regime},
                    "held_constant": ["target", "data_sha256", "split", "policy_hash", "prompt_policy"],
                    "shared_reference_roles": [],
                }
            )
    source = tmp_path / "matrix.json"
    plan = tmp_path / "plan.json"
    write_json(
        source,
        {
            "jobs": jobs,
            "contrast_instances": contrasts,
            "timing_seed": 17,
            "aggregate_limits": {
                "steps": 2000,
                "wall_seconds": 5000,
                "source_positions": 1000000,
                "storage_bytes": 3000000000,
                "download_bytes": 0,
            },
        },
    )
    resolved = resolve_plan(source, plan)
    result = launch_plan(plan)
    assert all(r["status"] == "completed" for r in result["results"].values()), result["results"]
    paths = result["results"]
    for tier in (11, 22):
        a = read(Path(paths[f"codec-t{tier}-reconstruction_only"]["path"]) / "codec-selection.json")
        b = read(Path(paths[f"codec-t{tier}-prediction_aware"]["path"]) / "codec-selection.json")
        assert a["objective"] == "reconstruction_only" and b["objective"] == "prediction_aware"
        assert a["selected"]["prediction_weight"] == 0 and b["selected"]["prediction_weight"] > 0
        for objective in ("reconstruction_only", "prediction_aware"):
            training = Path(paths[f"student-t{tier}-{objective}-equal_total_training_wall"]["path"])
            charged = read(training / "resolved_training_cost.json")
            assert len(charged["charged_runs"]) == 3 and charged["actual_pre_student_expenditure"] > 0
    analysis = analyze("ACL-003", bindings=Path(result["path"]) / "analysis-bindings.json")
    records = read(analysis["path"])
    assert len(records["analysis"]["paired_latency_differences"]) == 4
    assert {c["condition"]["data_tier"] for c in records["analysis"]["paired_latency_differences"]} == {
        11,
        22,
    }
    assert read(analysis["analysis_manifest"])["binding"]["plan_sha256"] == resolved["sha256"]


def test_dedicated_selected_codec_metrics_are_analyzed(teacher_artifacts, fitted_paths, tmp_path):
    from ncm_lab.evaluation.codec_analysis import analyze_codec_runs

    codec = fitted_paths["prediction"]
    config = {
        **teacher_artifacts["common"],
        "experiment_id": "ACL-005",
        "executor": "codec_evaluate",
        "codec_kind": "nested",
        "codec_objective": "prediction_aware",
        "prediction_weights": [0.1],
        "latent_width": 4,
        "data_tier": 22,
        "split": "dev",
        "evaluation_data": teacher_artifacts["data"]["development_data"],
        "input_receipts": [teacher_artifacts["data"]["input_receipts"][1]],
        "codec_selection": str(codec / "codec-selection.json"),
        "codec_sha256": digest(codec / "codec-selected.pt"),
    }
    path = tmp_path / "diagnostic.json"
    write_json(path, config)
    result = run("ACL-005", path, teacher_artifacts["budget"])
    assert result["status"] == "completed", result
    out = tmp_path / "derived"
    analyzed = analyze_codec_runs([Path(result["path"])], out)
    table = read(out / "tables.json")["rate_distortion"]
    assert analyzed["rows"]["rate_distortion"] == 3
    assert all(
        r["purpose"] == "post_selection_diagnostic" and r["valid_positions"] > 0 and r["split"] == "dev"
        for r in table
    )
    assert {r["standalone_scalars"] for r in table} == {16, 32, 60}
    with pytest.raises(FileExistsError):
        analyze_codec_runs([Path(result["path"])], out)


def test_all_declared_plans_validate_without_loading_models(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ncm_lab.experiments.matrix import resolve_plan
    from ncm_lab.experiments.registry import registry

    def forbidden(*args, **kwargs):
        raise AssertionError("static planning loaded a model/tokenizer")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    full = resolve_plan(
        root() / "experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml",
        tmp_path / "full.json",
    )
    assert full["coverage"]["software_paths_complete"] and len(full["selected_tiers"]) == 2
    assert not full["coverage"]["terminal_training_jobs"]
    controls = {instance["id"].split("-t250000")[0] for instance in full["contrast_instances"]}
    assert {
        "runtime-capacity",
        "runtime-codec-ce",
        "runtime-aux",
        "runtime-linear",
        "runtime-single",
        "codec-design",
        "runtime-reconstruction",
    } <= controls
    for eid, protocol in registry().items():
        if not eid.startswith("IS-"):
            plan = resolve_plan(
                root() / protocol["directory"] / "configs/matrix.yaml", tmp_path / (eid + ".json")
            )
            assert plan["jobs"], eid
    assert sum(j["config"]["executor"] == "robustness" for j in full["jobs"]) == 1
    assert not any(
        j["config"]["experiment_id"] == "ACL-004" and j["config"]["executor"] == "evaluate"
        for j in full["jobs"]
    )


@pytest.mark.parametrize("stop", ["budget", "interrupt"])
def test_budget_endpoint_retains_completed_selection_without_promoting_interrupt(tmp_path, monkeypatch, stop):
    from ncm_lab.models.fixtures import tiny_pair
    from ncm_lab.training.loop import train
    from ncm_lab.training.records import normalize_record, admitted_hash
    from ncm_lab.runtime.budget import Budget, Meter, BudgetExceeded

    torch.set_num_threads(1)
    target, backbone = tiny_pair()
    drafter = build_drafter(target, backbone, resolve_method({"method": "HIER_SKIP"}))

    def row(split):
        return normalize_record(
            {
                "input_ids": [1, 4, 5, 6, 7, 2],
                "prompt_length": 2,
                "group_id": split,
                "original_id": split,
                "split": split,
                "origin": "software_fixture",
            },
            split,
            "software_fixture",
        )

    meter = Meter(Budget(**read(root() / "configs/budgets/acceptance.yaml")), tmp_path)
    check = meter.check

    def bounded(*args, **kwargs):
        if meter.steps >= 1 and kwargs.get("step", True):
            if stop == "interrupt":
                raise KeyboardInterrupt()
            raise BudgetExceeded("bounded fixture stopped after completed development endpoint")
        return check(*args, **kwargs)

    monkeypatch.setattr(meter, "check", bounded)
    result = train(
        target,
        drafter,
        [row("train")],
        meter,
        admitted_hash([row("train")]),
        steps=3,
        dev_records=[row("dev")],
        endpoint_policy="best_completed_dev_before_budget",
    )
    assert result["status"] == "cancelled" and result["optimizer_steps"] == 1
    assert result["endpoint_eligible"] is (stop == "budget")
    assert result["stop_reason"] == ("budget" if stop == "budget" else "user_interrupt")
    assert read(tmp_path / "selection.json")["selected"]["step"] == 1
    assert result["distinct_eligible_response_positions"] == 2
    assert result["available_eligible_label_positions"] == 3


def test_infeasible_total_budget_stops_before_model(teacher_artifacts, tmp_path, monkeypatch):
    import ncm_lab.experiments.executors as executors

    config = {
        **teacher_artifacts["common"],
        **teacher_artifacts["data"],
        "experiment_id": "ACL-005",
        "executor": "train",
        "method": "HIER_SKIP",
        "total_new_training_wall_seconds": 1e-12,
        "budget_endpoint_policy": "best_completed_dev_before_budget",
        "steps": 1,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("infeasible budget loaded a model")

    monkeypatch.setattr(executors, "construct", forbidden)
    path = tmp_path / "config.json"
    write_json(path, config)
    result = run("ACL-005", path, teacher_artifacts["budget"])
    metrics = read(Path(result["path"]) / "metrics.json")
    assert result["status"] == "infeasible" and metrics["optimizer_steps"] == 0
    assert metrics["endpoint_eligible"] is False


def test_total_data_rejects_larger_codec_and_fixed_transfer_is_explicit(
    teacher_artifacts, fitted_paths, tmp_path
):
    base = {
        **teacher_artifacts["common"],
        **teacher_artifacts["data"],
        "experiment_id": "ACL-005",
        "executor": "train",
        "method": "HIER_MULTI_AE",
        "data_tier": 11,
        "steps": 1,
        "codec_selection": str(fitted_paths["prediction"] / "codec-selection.json"),
        "codec_sha256": digest(fitted_paths["prediction"] / "codec-selected.pt"),
    }
    path = tmp_path / "config.json"
    write_json(path, base)
    rejected = run("ACL-005", path, teacher_artifacts["budget"])
    assert rejected["status"] == "failed" and "same admitted corpus" in rejected["error"]
    write_json(path, {**base, "fixed_codec_transfer": True, "comparison": "fixed_codec_transfer"})
    accepted = run("ACL-005", path, teacher_artifacts["budget"])
    assert accepted["status"] == "completed", accepted
    payload = read_checkpoint(Path(accepted["path"]) / "selected.pt")
    assert payload["training_contract"]["data_curve"] == "fixed_codec_transfer"
    assert payload["architecture"]["codec_dependencies"]["data_tier"] == 22
    assert read(Path(accepted["path"]) / "dependencies.json")


def test_legacy_migrations_require_original_provenance_and_preserve_bytes(
    teacher_artifacts, fitted_paths, tmp_path
):
    from ncm_lab.training.source_contracts import migrate_teacher, admit
    from ncm_lab.training.codec import migrate_codec, load_selected_codec
    from ncm_lab.runtime.provenance import Run
    from ncm_lab.runtime.budget import Budget
    from ncm_lab.experiments.registry import get

    teacher = teacher_artifacts["train"] / "teacher.jsonl"
    original_sha = digest(teacher)
    migrated = tmp_path / "teacher-migration.json"
    with pytest.raises(ValueError, match="original measured"):
        migrate_teacher(teacher, migrated)
    migrate_teacher(teacher, migrated, diagnostic_fixture=True)
    config = {**teacher_artifacts["common"], "teacher_contracts": {"train": str(migrated)}}
    rows, _ = admit(teacher, "train", config)
    assert rows and rows[0]["legacy_source_migrated"] and digest(teacher) == original_sha

    # A deliberately uncontracted fixture bundle models the old file format. It is
    # a new diagnostic run, never a fabricated historical research observation.
    source = fitted_paths["prediction"]
    original = read(source / "codec-selection.json")
    payload = read_checkpoint(source / original["weights"])
    payload.pop("source_contract")
    payload.pop("objective")
    legacy_config = {
        **read(source / "config.json"),
        "artifact_root": str(tmp_path / "legacy-fixture"),
        "data_ladder_version": 3,
    }
    record = Run(
        get("ACL-005"),
        legacy_config,
        Budget(**read(teacher_artifacts["budget"])),
        {"scope": "diagnostic fixture serialization of uncontracted legacy format"},
    )
    atomic_checkpoint(record.path / original["weights"], payload)
    original["weights_sha256"] = digest(record.path / original["weights"])
    write_json(record.path / "codec-selection.json", original)
    record.close("completed", {"fixture_only": True})
    weights_sha = digest(record.path / original["weights"])
    derived = tmp_path / "codec-migration.json"
    migrate_codec(record.path / "codec-selection.json", migrated, derived, diagnostic_fixture=True)
    loaded = load_selected_codec(
        derived, expected_model=teacher_artifacts["common"]["model"], diagnostic=True
    )
    assert (
        loaded.source_contract["model"]["target_revision"]
        == teacher_artifacts["common"]["model"]["target_revision"]
    )
    assert digest(record.path / original["weights"]) == weights_sha
    with pytest.raises(ValueError, match="diagnostic-only"):
        load_selected_codec(derived)
    insufficient = tmp_path / "unproven.jsonl"
    shutil.copyfile(teacher, insufficient)
    with pytest.raises((FileNotFoundError, ValueError)):
        migrate_teacher(insufficient, tmp_path / "refused.json", diagnostic_fixture=True)


def test_generated_pilot_validates_resume_and_selection_paths(tmp_path):
    from ncm_lab.runtime.prepare import selected_assets, write_launch_configs
    from ncm_lab.experiments.matrix import resolve_plan

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    prompts = {"outputs": {s: {"path": str(tmp_path / (s + ".jsonl"))} for s in ("train", "dev")}}
    write_launch_configs(attempt, tmp_path / "assets", selected_assets(), prompts, "cpu", None)
    plan = resolve_plan(attempt / "launch/codec_pilot_matrix.yaml", tmp_path / "pilot-plan.json")
    configs = {j["id"]: j["config"] for j in plan["jobs"]}
    assert configs["codec"]["prediction_weights"] == [0.1]
    assert configs["transfer-resume"]["resume_sha256"] == "@transfer/final.pt#sha256"
    assert configs["adapted-parity"]["checkpoint_selection"] == "@transfer-resume/selection.json"


def test_cost_coordinates_and_oracle_prediction_units(teacher_artifacts, fitted_paths, tmp_path):
    from ncm_lab.evaluation.codec_analysis import analyze_codec_runs
    from ncm_lab.reporting.costs import cost_closure

    student = fitted_paths["student"]
    codec = fitted_paths["prediction"]
    config = {
        **teacher_artifacts["common"],
        "experiment_id": "ACL-005",
        "executor": "latent_diagnostics",
        "method": "HIER_MULTI_AE",
        "split": "dev",
        "max_requests": 1,
        "evaluation_data": teacher_artifacts["data"]["development_data"],
        "input_receipts": [teacher_artifacts["data"]["input_receipts"][1]],
        "checkpoint": str(student / "selected.pt"),
        "checkpoint_sha256": digest(student / "selected.pt"),
        "checkpoint_selection": str(student / "selection.json"),
        "codec_selection": str(codec / "codec-selection.json"),
        "codec_sha256": digest(codec / "codec-selected.pt"),
    }
    path = tmp_path / "diagnostic.json"
    write_json(path, config)
    result = run("ACL-005", path, teacher_artifacts["budget"])
    assert result["status"] == "completed", result
    output = tmp_path / "tables"
    analyze_codec_runs([student, Path(result["path"])], output)
    tables = read(output / "tables.json")
    closure = cost_closure(student)
    prep = closure["total_once_per_dependency"]["source_preprocessing_seconds"]
    point = tables["learning_curves"][0]
    parents = [
        r
        for r in closure["runs"]
        if r["run_id"] != student.name
        and r["kind"] in {"teacher", "codec", "latent_cache", "train", "acceptance"}
    ]
    assert prep > 0
    assert point["cumulative_adaptation_wall_seconds"] == pytest.approx(
        prep + sum(r["executor_wall_seconds"] for r in parents) + point["cumulative_student_wall_seconds"]
    )
    assert point["selected_total_procedure_cost"] is None
    assert (
        tables["selected_procedure_cost"][0]["full_procedure_adaptation_wall_seconds"]
        > point["cumulative_adaptation_wall_seconds"]
    )
    diagnostic = read(next(Path(result["path"]).glob("diagnostic-*/latent-diagnostics.json")))
    for row in tables["oracle_predicted"]:
        level = str(row["level"])
        assert row["oracle_normalized_mse"] == diagnostic["oracle_codec"][level]["normalized_mse"]
        assert row["predicted_normalized_mse"] == diagnostic["causal_predictions"][level]["normalized_mse"]
        assert row["positions"]["valid"] == diagnostic["evaluated_positions"]["valid"]
        assert row["normalization"] > 0


def test_shared_robustness_strata_have_exact_denominators_and_no_new_model_run(teacher_artifacts, tmp_path):
    from ncm_lab.runtime.provenance import Run
    from ncm_lab.runtime.budget import Budget
    from ncm_lab.experiments.registry import get

    config = {
        **teacher_artifacts["common"],
        "experiment_id": "ACL-002",
        "executor": "evaluate",
        "dataset": "longbench",
    }
    record = Run(
        get("ACL-002"),
        config,
        Budget(**read(teacher_artifacts["budget"])),
        {"scope": "typed observation fixture; no benchmark result"},
    )
    rows = []
    for i, length in enumerate((100, 2048, 9000, 33000)):
        rows.append(
            {
                "original_id": str(i),
                "group_id": str(i),
                "method": "AR",
                "dataset": "longbench",
                "input_tokens": length,
                "request_seconds": 1,
                "correct": None,
                "official_scores": {"rouge_l": 0.5} if i != 2 else {},
                "error": "scoring fixture failure" if i == 2 else None,
            }
        )
    (record.path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record.close("completed", {"fixture_only": True})
    path = tmp_path / "robustness.json"
    write_json(
        path,
        {
            **teacher_artifacts["common"],
            "experiment_id": "ACL-004",
            "executor": "robustness",
            "context_length_strata": [0, 2048, 8192, 32768, None],
            "shared_evaluation_runs": [str(record.path)],
        },
    )
    result = run("ACL-004", path, teacher_artifacts["budget"])
    assert result["status"] == "completed", result
    strata = read(Path(result["path"]) / "strata.json")
    assert strata["shared_evidence"] and not strata["new_independent_generalization_runs"]
    assert [s["requests"] for s in strata["strata"]] == [1, 1, 1, 1]
    assert [s["missing_or_failed"] for s in strata["strata"]] == [0, 0, 1, 0]
    assert all(s["analysis"]["per_seed"][0]["accuracy"] is None for s in strata["strata"])
    assert strata["sources"][0]["run_id"] == record.id
    write_json(path, {**read(path), "result_origin": "measured"})
    rejected = run("ACL-004", path, teacher_artifacts["budget"])
    assert rejected["status"] == "failed" and "dependency lineage" in rejected["error"]
