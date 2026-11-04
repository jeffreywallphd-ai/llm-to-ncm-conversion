"""Actual tiny Stage 1 optimization and manifest-bound recovery/selection."""

import json
from pathlib import Path
import copy
import pytest
import torch
from ncm_lab.stage1.config import resolve
from ncm_lab.stage1.workflow import plan, run, load, checked, verify_record, configuration
from ncm_lab.runtime.io import read, digest, reference, atomic_json, write_json


def finish(experiment):
    result = run(experiment)
    if result["status"] == "review_required":
        reviewed = run(
            experiment,
            stage="codec",
            allow_codec_review=True,
            reason="Explicit tiny software fixture exercises finite diagnostic override; not a research result",
        )
        assert reviewed["status"] == "ready", reviewed
        result = run(experiment)
    assert result["status"] == "completed", result
    return result


def test_tiny_end_to_end_independent_precision_qualification_and_idempotency(tmp_path, stage1_fixture):
    settings = resolve(stage1_fixture["config"], canvas_size="xsmall")
    experiment = plan(settings, tmp_path / "actual experiment with spaces")["experiment"]
    finish(experiment)
    path, state, _ = load(experiment)
    codec = read(checked(state["stages"]["codec"]["receipt"]))
    generator = read(checked(state["stages"]["generator"]["receipt"]))
    assert codec["result"]["optimizer_steps"] == 12 and generator["result"]["optimizer_steps"] == 3
    assert codec["result"]["stage1_diagnostics"]["example_presentations"] == 48
    assert generator["result"]["stage1_diagnostics"]["example_presentations"] == 6
    assert codec["result"]["stage1_diagnostics"]["selection_policy"]["complete_coarse_coverage"]
    assert (
        generator["result"]["stage1_diagnostics"]["selection_policy"]["policy"]
        == "stage1_mean_example_text_ce_v1"
    )
    targets = read(checked(state["stages"]["targets"]["receipt"]))["result"]["artifact"]
    from ncm_lab.data.artifacts import Dataset

    dataset = Dataset(targets)
    assert dataset[0][1].dtype == torch.float32
    assert dataset.manifest["identity"]["codec_computation_dtype"] == "torch.float32"
    assert generator["result"]["stage1_diagnostics"]["precision_boundary"]["cached_source_dtype"] == "float32"
    report = read(path.parent / "reports/stage1_summary.json")
    assert report["workflow_completed"] and report["origin"] == "software_fixture"
    assert report["evaluation"]["splits"]["test"]["examples"] == 4
    assert report["qualification"]["selected"]["train"]["examples"] == 8
    assert report["evaluation"]["teacher_or_codec_loaded"] is False
    rows = [
        json.loads(line)
        for line in checked(report["evaluation"]["raw_outputs"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all(r["computed_answer_positions"] == r["allocated_answer_positions"] == 8 for r in rows)
    assert all(r["generation_backbone_traversals"] == 1 for r in rows)
    before = {
        p: digest(p)
        for p in (
            path,
            path.parent / "reports/stage1_summary.json",
            path.parent / "reports/stage1_summary.md",
        )
    }
    repeated = run(experiment)
    assert repeated["status"] == "already_completed", repeated
    assert all(digest(p) == sha for p, sha in before.items())
    assert run(experiment, stage="generator")["status"] == "already_completed"
    for name, entry in state["stages"].items():
        verify_record(name, read(checked(entry["receipt"])), path)


def prepared_targets(tmp_path, fixture, *, through_targets=True):
    experiment = plan(
        resolve(fixture["config"], canvas_size="xsmall"), tmp_path / "training recovery experiment"
    )["experiment"]
    for name in ("prepare", "teacher", "codec"):
        result = run(experiment, stage=name)
        if result["status"] == "review_required":
            result = run(
                experiment,
                stage="codec",
                allow_codec_review=True,
                reason="explicit offline reduced fixture diagnostic",
            )
        assert result["status"] == "ready", result
    if through_targets:
        result = run(experiment, stage="targets")
        assert result["status"] == "ready", result
    return load(experiment)


def test_interrupted_targets_resume_preserves_completed_codec_and_charges(
    tmp_path, stage1_fixture, monkeypatch
):
    from ncm_lab.runtime.runner import execute
    from ncm_lab.data.artifacts import ArtifactWriter

    path, state, settings = prepared_targets(tmp_path, stage1_fixture, through_targets=False)
    upstream = {name: copy.deepcopy(state["stages"][name]) for name in ("prepare", "teacher", "codec")}
    value = configuration("targets", path, state, settings)
    attempt = path.parent / "stages/targets/interrupted-first-target-shard"
    attempt.mkdir(parents=True)
    write_json(attempt / "resolved_config.json", value, exclusive=True)
    original = ArtifactWriter.append

    def fail_after_first_target(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.kind == "native_targets_v1" and self.count == 1:
            raise OSError("injected target interruption after a real FP32 shard write")

    with monkeypatch.context() as patch:
        patch.setenv("NCM_CHILD_LAUNCH_RECORD", str(attempt / "launch.json"))
        patch.setattr(ArtifactWriter, "append", fail_after_first_target)
        failed = execute(value)
    assert failed["status"] == "failed" and "injected target interruption" in failed["error"]
    assert not Path(value["output_ref"]).exists()
    ledger = path.parent / "ledgers/targets-writes.json"
    before = read(ledger)["written_bytes"]
    assert before > 0
    state["stages"]["targets"].update(status="failed", attempts=[{"path": str(attempt), "resume": None}])
    atomic_json(path, state)
    finish(path)
    _, resumed, _ = load(path)
    assert all(resumed["stages"][name] == entry for name, entry in upstream.items())
    assert read(ledger)["written_bytes"] > before
    assert len(read(ledger)["attempts"]) == 2
    record = read(checked(resumed["stages"]["targets"]["receipt"]))
    verify_record("targets", record, path)
    assert record["result"]["stats"]["examples"] == 12


def test_mid_stage_and_at_target_recovery_retain_state_and_do_only_remaining_updates(
    tmp_path, stage1_fixture, monkeypatch
):
    from ncm_lab.runtime.runner import execute
    from ncm_lab.training.checkpoints import CheckpointReader

    path, state, settings = prepared_targets(tmp_path, stage1_fixture)
    value = configuration("generator", path, state, settings)
    interrupted_attempt = path.parent / "stages/generator/controller-interrupted-before-finally"
    interrupted_attempt.mkdir(parents=True)
    write_json(interrupted_attempt / "resolved_config.json", value, exclusive=True)
    with monkeypatch.context() as patch:
        patch.setenv("NCM_CHILD_LAUNCH_RECORD", str(interrupted_attempt / "launch.json"))
        partial = execute(value, stop_after=1)
    assert partial["status"] == "incomplete" and partial["optimizer_steps"] == 1, partial
    reader = CheckpointReader(partial["checkpoint"])
    original = reader.materialize(reader.field("training"))
    # Simulate losing the controller BEFORE its finally block copies the pointer
    # into manifest state. Resume must inspect this exact owned launch, not restart.
    state["stages"]["generator"].update(
        status="incomplete", attempts=[{"path": str(interrupted_attempt), "resume": None}]
    )
    assert "last_checkpoint" not in state["stages"]["generator"]
    atomic_json(path, state)
    resumed = run(path, stage="generator")
    assert resumed["status"] == "ready", resumed
    _, state, _ = load(path)
    assert any(
        row["event"] == "recovered_owned_launch_checkpoint" and row["stage"] == "generator"
        for row in state["history"]
    )
    resumed_result = read(checked(state["stages"]["generator"]["receipt"]))["result"]
    assert resumed_result["optimizer_steps"] == 3 and resumed_result["charged_work"]["steps"] == 2
    assert resumed_result["stage1_diagnostics"]["initial"] == original["initial_validation"]
    full = execute(value)
    a = CheckpointReader(full["checkpoint"])
    b = CheckpointReader(resumed_result["checkpoint"])
    for key in ("model", "optimizer", "rng"):
        left, right = a.materialize(a.field(key)), b.materialize(b.field(key))

        def equal(x, y):
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y)
            elif isinstance(x, dict):
                assert set(x) == set(y)
                for k in x:
                    equal(x[k], y[k])
            elif isinstance(x, (list, tuple)):
                for first, second in zip(x, y, strict=True):
                    equal(first, second)
            else:
                assert x == y

        equal(left, right)
    # Fail final export after a COMPLETE target-boundary checkpoint; recovery must
    # finish sealing/export rather than perform a fourth optimizer update.
    import ncm_lab.inference.export as exports

    original_export = exports.export_generator

    def fail_export(*args, **kwargs):
        raise OSError("injected export failure after final checkpoint")

    monkeypatch.setattr(exports, "export_generator", fail_export)
    failed = execute(value)
    assert failed["status"] == "failed"
    current = Path(failed["path"]) / "checkpoints/current.json"
    reader = CheckpointReader(current)
    assert reader.materialize(reader.field("training"))["optimizer_steps"] == 3
    monkeypatch.setattr(exports, "export_generator", original_export)
    _, state, _ = load(path)
    entry = state["stages"]["generator"]
    entry.setdefault("prior_receipts", []).append(entry.pop("receipt"))
    entry.update(status="incomplete", last_checkpoint=reference(reader.path), attempts=[])
    atomic_json(path, state)
    result = run(path, stage="generator")
    assert result["status"] == "ready", result
    _, state, _ = load(path)
    final = read(checked(state["stages"]["generator"]["receipt"]))["result"]
    assert final["optimizer_steps"] == 3 and final["charged_work"]["steps"] == 0


def test_stage1_checkpoint_cadence_and_selected_step_distinct_from_terminal(
    tmp_path, stage1_fixture, monkeypatch
):
    from ncm_lab.runtime.runner import execute
    import ncm_lab.training.engine as engine

    path, state, settings = prepared_targets(tmp_path, stage1_fixture)
    value = configuration("generator", path, state, settings)
    value["training"].update(steps=5, validation_interval=1, checkpoint_interval=2)
    original = engine.chunked_ce
    calls = []

    def controlled_validation(*args, **kwargs):
        if torch.is_grad_enabled():
            return original(*args, **kwargs)
        boundary = len(calls) // 4
        calls.append(boundary)
        # Authored software validation fixture: initialization, then trained
        # improvement at step 1 followed by regressions. Never a model-quality claim.
        return torch.tensor(([5.0, 4.0, 4.1, 4.2, 4.3, 4.4, 4.0][min(boundary, 6)]))

    monkeypatch.setattr(engine, "chunked_ce", controlled_validation)
    result = execute(value)
    assert result["status"] == "completed", result
    assert result["optimizer_steps"] == 5 and result["selected_optimizer_step"] == 1
    rows = [
        json.loads(line) for line in (Path(result["path"]) / "monitor_events.jsonl").read_text().splitlines()
    ]
    checkpoints = [r["step"] for r in rows if r["status"] == "checkpoint_completed"]
    assert checkpoints == [1, 2, 4, 5]
    assert result["stage1_diagnostics"]["initial"]["selection_score"] == 5
    assert result["stage1_diagnostics"]["selected"]["selection_score"] == 4


@pytest.mark.parametrize("failure_stage", ["prepare", "teacher", "codec", "targets", "generator", "evaluate"])
def test_child_failure_stops_dependencies_and_saves_partial_report(
    tmp_path, stage1_fixture, monkeypatch, failure_stage
):
    import ncm_lab.stage1.workflow as flow

    path = Path(plan(resolve(stage1_fixture["config"]), tmp_path / "injected experiment")["experiment"])
    calls = []

    def fake_child(value, attempt, path, state, settings, resume):
        name = value["stage1_context"]["stage"]
        calls.append(name)
        attempt.mkdir(parents=True)
        return {
            "result": {"status": "failed", "error": "authored child-failure software fixture"},
            "exit_status": 2,
        }

    # Preceding verified completion is a state-machine stub; actual scientific
    # stage/export behavior is covered by the real tiny end-to-end test above.
    _, state, settings = load(path)
    names = list(flow.STAGES)
    for previous in names[: names.index(failure_stage)]:
        state["stages"][previous]["status"] = "completed"
        ref = path.parent / f"{previous}-fixture-receipt.json"
        write_json(ref, {"result": {}}, exclusive=True)
        state["stages"][previous]["receipt"] = reference(ref)
    atomic_json(path, state)
    monkeypatch.setattr(flow, "verify_record", lambda *a: {})
    monkeypatch.setattr(flow, "ensure_resources", lambda *a: None)
    monkeypatch.setattr(flow, "configuration", lambda name, *a: {"stage1_context": {"stage": name}})
    monkeypatch.setattr(flow, "execute_child", fake_child)
    monkeypatch.setattr(flow, "reports", lambda path, state, settings: {})
    result = run(path)
    assert result["status"] == "failed" and calls == [failure_stage]
    assert result["stages"][failure_stage] == "failed"


def test_locked_rows_rejected_by_training_artifact_and_capacity_blocks_before_teacher(
    tmp_path, stage1_fixture
):
    from ncm_lab.data.artifacts import Dataset, ArtifactWriter

    settings = resolve(stage1_fixture["config"])
    settings["qualification"].update(train_examples=10000)
    for cfg in settings["stages"].values():
        cfg["stage1_data"] = copy.deepcopy(settings["qualification"])
    from ncm_lab.runtime.io import canonical_hash

    settings["contract_sha256"] = canonical_hash(
        {k: v for k, v in settings.items() if k not in {"config_path", "contract_sha256"}}
    )
    experiment = plan(settings, tmp_path / "insufficient source")["experiment"]
    result = run(experiment)
    assert result["status"] == "blocked" and result["stages"]["teacher"] == "pending", result
    assert "capacity" in str(result["failure"])
    # A successful separate preparation seals test records under a distinct schema.
    settings = resolve(stage1_fixture["config"])
    path = Path(plan(settings, tmp_path / "locked source")["experiment"])
    assert run(path, stage="prepare")["status"] == "ready"
    _, state, _ = load(path)
    record = read(checked(state["stages"]["prepare"]["receipt"]))["result"]
    locked = Dataset(checked(record["evaluation_data"]))
    example, _ = locked[0]
    writer = ArtifactWriter(tmp_path / "must not train", "native_data_v1", locked.manifest["identity"])
    try:
        with pytest.raises(ValueError, match="locked evaluation"):
            writer.append(example)
    finally:
        writer.index.close()


def test_actual_partial_accumulation_matches_manual_microbatch_mean_and_one_update(tmp_path, stage1_fixture):
    import random
    from ncm_lab.data.artifacts import Dataset, Cursor
    from ncm_lab.training.engine import padded_tensors, make_optimizer
    from ncm_lab.training.losses import chunked_ce, auxiliary_loss
    from ncm_lab.teacher.model import load_generator
    from ncm_lab.runtime.runner import execute
    from ncm_lab.training.checkpoints import CheckpointReader

    rows = [json.loads(line) for line in stage1_fixture["source"].read_text().splitlines()]
    for i, row in enumerate(rows):
        row["answer"] = "token8 " * (1 + i % 5)
    stage1_fixture["source"].write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    source_config = Path(stage1_fixture["experiment"]["stages"]["prepare"])
    source_value = read(source_config)
    source_value["data"]["source_sha256"] = digest(stage1_fixture["source"])
    write_json(source_config, source_value)
    path, state, settings = prepared_targets(tmp_path, stage1_fixture)
    value = configuration("generator", path, state, settings)
    # Explicit internal software diagnostic: one epoch gives 3/3/2 examples,
    # hence THREE actual microbatches despite accumulation's maximum of four.
    value["training"].update(steps=1, epochs=1, batch_size=3, accumulation=4)
    data = Dataset(value["inputs"]["targets"])
    assert len({len(data.metadata(i)["example"]["answer_ids"]) for i in data.indices("train")}) > 1
    sampler = random.Random(value["training"]["seed"])
    cursor = Cursor(data.indices("train"), sampler)
    groups = [cursor.take(3, 1) for _ in range(3)]
    assert [len(g) for g in groups] == [3, 3, 2]
    torch.manual_seed(value["training"]["seed"])
    model = load_generator(value["model"], value["architecture"], "cpu", torch.bfloat16)
    model.set_phase("full")
    model.train()
    optimizer = make_optimizer(model, value["training"], True)
    for indices in groups:
        _, ids, mask, labels, valid, targets = padded_tensors(data, indices, 8, "cpu", torch.bfloat16)
        out = model(ids, mask, return_aux=True)
        text = chunked_ce(model.backbone.lm_head, out.answer_states, labels, valid, 8)
        latent, _, _ = auxiliary_loss(
            out.cumulative_codes,
            targets,
            valid,
            model.architecture.resolutions,
            data.manifest["identity"]["scales"],
        )
        ((text + 0.1 * latent) / 3).backward()
    gradients = optimizer.check_gradients()
    gradients.clip_(1.0)
    optimizer.step(gradients)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = execute(value)
    assert result["status"] == "completed" and result["optimizer_steps"] == 1, result
    reader = CheckpointReader(result["checkpoint"])
    actual = reader.materialize(reader.field("model"))
    assert all(torch.equal(actual[k], v) for k, v in expected.items())
    assert result["stage1_diagnostics"]["example_presentations"] == 8
    rows = [json.loads(line) for line in (Path(result["path"]) / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["actual_accumulation"] == 3 and rows[0]["microbatches"] == 3


def test_separate_and_all_routes_equivalent_then_explicit_extension_and_ar_reference(
    tmp_path, stage1_fixture
):
    from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE
    from ncm_lab.stage1.workflow import run_ar_reference

    settings = resolve(stage1_fixture["config"])
    first = Path(plan(settings, tmp_path / "all route")["experiment"])
    finish(first)
    second = Path(plan(settings, tmp_path / "separate route")["experiment"])
    for name in ("prepare", "teacher", "codec", "targets", "generator"):
        result = run(second, stage=name)
        if result["status"] == "review_required":
            result = run(second, stage=name, allow_codec_review=True, reason="explicit tiny fixture review")
        assert result["status"] == "ready", result
    _, a, sa = load(first)
    _, b, sb = load(second)
    assert sa == sb
    for name in ("prepare", "codec", "generator"):
        left = read(checked(a["stages"][name]["receipt"]))["result"]
        right = read(checked(b["stages"][name]["receipt"]))["result"]
        if name == "prepare":
            assert read(checked(left["qualification"])) == read(checked(right["qualification"]))
            continue
        assert left["stage1_diagnostics"]["selected"] == right["stage1_diagnostics"]["selected"]
        assert left["selected_optimizer_step"] == right["selected_optimizer_step"]
        manifests = [read(result["artifact"]) for result in (left, right)]
        readers = [
            CheckpointReader(
                Path(result["artifact"]).parent / info["weights_manifest"], expected_kind=INFERENCE
            )
            for result, info in zip((left, right), manifests, strict=True)
        ]
        weights = [r.materialize(r.field("model")) for r in readers]
        assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    # Extension is explicitly TOTAL, only before locked test, with prior state and
    # accumulated charges retained. The immutable initial proposal stays unchanged.
    proposal_sha = digest(checked(b["contract"]))
    extended = run(
        second, stage="generator", extend_generator_steps=5, reason="tiny fixture total-step extension"
    )
    assert extended["status"] == "ready", extended
    _, b, _ = load(second)
    result = read(checked(b["stages"]["generator"]["receipt"]))["result"]
    assert result["optimizer_steps"] == 5 and result["charged_work"]["steps"] == 2
    assert digest(checked(b["contract"])) == proposal_sha
    assert run(second, stage="evaluate")["status"] == "completed"
    with pytest.raises(ValueError, match="before locked evaluation"):
        run(second, stage="generator", extend_generator_steps=6, reason="must not extend after test")
    reference_result = run_ar_reference(second, max_new_tokens=4)
    assert reference_result["status"] == "completed", reference_result
    reference_report = read(reference_result["reference"])
    assert reference_report["used_to_repair_ncm"] is False
    assert reference_report["likelihood_comparable_to_parallel_ce"] is False
    assert len(reference_report["rows"]) == 4
    assert all(row["cap"] == 4 and row["returned_positions"] <= 4 for row in reference_report["rows"])


def test_recovery_contract_rejected_before_allocation_and_no_clock_training_stop(
    tmp_path, stage1_fixture, monkeypatch
):
    from ncm_lab.runtime.runner import execute
    from ncm_lab.runtime.budget import Meter
    import ncm_lab.training.engine as engine

    path, state, settings = prepared_targets(tmp_path, stage1_fixture)
    value = configuration("generator", path, state, settings)
    # Observations beyond both historical clock limits do not stop Stage 1.
    original_init = Meter.__init__

    def old_elapsed(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.started -= 20000

    with monkeypatch.context() as patch:
        patch.setattr(Meter, "__init__", old_elapsed)
        partial = execute(value, stop_after=1)
    assert partial["status"] == "incomplete" and partial["optimizer_steps"] == 1, partial
    assert partial["wall_policy"]["attempt_observed_seconds"] > 20000
    assert partial["wall_policy"]["enforce_wall_time_limits"] is False

    def forbidden(*args, **kwargs):
        pytest.fail("conflicting recovery reached model allocation")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "load_generator", forbidden)
        mutations = []
        batch = copy.deepcopy(value)
        batch["training"]["batch_size"] = 2
        mutations.append(batch)
        subset = copy.deepcopy(value)
        subset["training"]["max_validation_examples"] = 3
        mutations.append(subset)
        dtype = copy.deepcopy(value)
        dtype["dtype"] = "float32"
        mutations.append(dtype)
        policy = copy.deepcopy(value)
        policy["stage1"]["selection"] = "stage1_codec_fixed_components_v1"
        mutations.append(policy)
        codec = copy.deepcopy(value)
        from ncm_lab.runtime.io import resolve_reference
        import shutil

        original_targets = resolve_reference(value["inputs"]["targets"])
        different = tmp_path / "different codec targets"
        shutil.copytree(original_targets.parent, different)
        manifest = read(different / "manifest.json")
        manifest["identity"]["codec_sha256"] = "0" * 64
        write_json(different / "manifest.json", manifest)
        codec["inputs"]["targets"] = str(different / "manifest.json")
        mutations.append(codec)
        for changed in mutations:
            result = execute(changed, resume=partial["checkpoint"])
            assert result["status"] == "failed" and "before allocation" in result["error"].replace(
                "before model allocation", "before allocation"
            ), result


@pytest.mark.parametrize("valid", [False, True])
def test_codec_review_override_respects_hard_validity_boundary(tmp_path, stage1_fixture, monkeypatch, valid):
    import ncm_lab.stage1.workflow as flow

    path = Path(plan(resolve(stage1_fixture["config"]), tmp_path / "gate state fixture")["experiment"])
    _, state, _ = load(path)
    receipt = path.parent / "authored-gate-receipt.json"
    write_json(
        receipt, {"result": {"stage1_diagnostics": {"codec_qualification": {"artifacts_valid": valid}}}}
    )
    state["stages"]["codec"].update(status="review_required", receipt=reference(receipt))
    atomic_json(path, state)
    # Gate-control fixture only; real sealed codec artifacts are exercised above.
    monkeypatch.setattr(flow, "verify_record", lambda *args: {})
    monkeypatch.setattr(flow, "reports", lambda *args: {})
    result = run(path, stage="codec", allow_codec_review=True, reason="explicit gate-control fixture")
    _, final, _ = load(path)
    assert result["status"] == ("ready" if valid else "review_required")
    assert bool(final["codec_review_override"]) is valid
    assert final["stages"]["targets"]["status"] == "pending"


def test_controller_cancellation_stops_only_owned_child_and_saves_partial_report(
    tmp_path, stage1_fixture, monkeypatch
):
    import ncm_lab.stage1.workflow as flow
    import ncm_lab.runtime.pilot as pilot

    path = Path(plan(resolve(stage1_fixture["config"]), tmp_path / "cancelled controller")["experiment"])
    owned, stopped = [], []

    class InterruptedChild:
        pid = 123456
        returncode = None

        def __init__(self, *args, **kwargs):
            assert kwargs["shell"] is False and kwargs["env"]["PYTHONUNBUFFERED"] == "1"
            owned.append(self)

        def poll(self):
            raise KeyboardInterrupt

    def stop(child):
        assert child in owned
        stopped.append(child)
        child.returncode = 130

    monkeypatch.setattr(flow.subprocess, "Popen", InterruptedChild)
    monkeypatch.setattr(pilot, "stop_child", stop)
    result = run(path, stage="prepare")
    assert result["status"] == "cancelled" and owned == stopped
    _, state, _ = load(path)
    assert state["stages"]["prepare"]["status"] == "cancelled"
    assert read(checked(state["reports"]["json"]))["workflow_completed"] is False
    attempt = Path(state["stages"]["prepare"]["attempts"][0]["path"])
    assert read(attempt / "exit.json")["exit_status"] == 130
