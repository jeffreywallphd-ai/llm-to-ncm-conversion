from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import pytest
import torch
from ncm_lab.block8.cli import main
from ncm_lab.block8.config import resolve
from ncm_lab.block8.data import BlockDataset
from ncm_lab.block8.runtime import checked
from ncm_lab.block8.training import load_export
from ncm_lab.block8.workflow import run_workflow
from ncm_lab.data.artifacts import ArtifactWriter, Dataset
from ncm_lab.runtime.io import digest, read, write_json
from ncm_lab.training.checkpoints import CheckpointReader


def prepare_fixture(fixture, path):
    return run_workflow(
        fixture["settings"],
        experiment_dir=path,
        execute=True,
        prepare_only=True,
        data_manifest=fixture["data"],
        evaluation_manifest=fixture["evaluation"],
    )


def test_parent_offsets_preserve_whole_answer_eos_and_do_not_duplicate_context(block8_fixture):
    f = block8_fixture
    result = prepare_fixture(f, f["folder"] / "prepared")
    assert result["status"] == "completed", result
    with closing(BlockDataset(checked(result["dataset"]), f["settings"])) as data:
        assert data.manifest["parents"] == dict(train=4, dev=4, test=4)
        assert data.manifest["views"] == dict(train=8, dev=8, test=8)
        assert data.manifest["unique_answer_positions"] == dict(train=50, dev=50, test=50)
        for split in ("train", "dev", "test"):
            for key in data.ids[split]:
                parent = data.parent(key)
                recovered = []
                for start in range(0, len(parent.answer_ids), 8):
                    context, labels = data.view(key, start)
                    assert context == parent.prompt_ids + parent.answer_ids[:start]
                    assert 2 not in labels[:-1]
                    if start + 8 < len(parent.answer_ids):
                        assert 2 not in labels
                    recovered.extend(labels)
                assert recovered == parent.answer_ids
                assert parent.provenance["reference_text"].startswith("Original complete")
        assert {r[1] for r in data.db.execute("PRAGMA table_info(views)")} == {"parent", "start", "split"}
        assert not (Path(checked(result["dataset"])).parent / "prefixes.jsonl").exists()
    state = read(result["experiment"])
    assert set(state["stages"]) == {"prepare"}
    assert state["recoveries"] == {"codec": None, "generator": None}


def test_complete_workflow_export_generation_and_at_target_resume(block8_fixture, capsys):
    f = block8_fixture
    result = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "complete",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert result["status"] == "completed", result
    assert result["completed_step"] == 4
    state = read(result["experiment"])
    training = read(checked(state["stages"]["generator"]["receipt"]))["result"]
    assert training["full_backbone_trainable"] and 0 <= training["selected_step"] <= 4
    evaluated = read(checked(result["evaluation"]["report"]))
    assert evaluated["selection_used_test"] is False and evaluated["teacher_or_verifier_used"] is False
    assert len(evaluated["rows"]) == 2
    before = {p.name: digest(p) for p in (Path(result["experiment"]).parent / "ledgers").glob("*.json")}
    resumed = run_workflow(resume=result["experiment"], execute=True)
    assert resumed["status"] == "completed" and resumed["selected"] == result["selected"]
    assert before == {
        p.name: digest(p) for p in (Path(result["experiment"]).parent / "ledgers").glob("*.json")
    }
    model, tokenizer, metadata = load_export(checked(result["selected"]), "cpu")
    assert model.architecture.canvas == 8 and metadata["teacher_codec_or_verifier_required"] is False
    assert len(tokenizer) == 32
    generated = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "generation",
        execute=True,
        checkpoint=checked(result["selected"]),
        prompt="token4 token7",
    )
    assert generated["status"] == "completed", generated
    assert generated["result"]["verification_or_repair"] is False
    assert generated["result"]["selected_step"] == metadata["selected_step"]
    assert generated["result"]["blocks"] <= 3
    assert "block8_generator" in capsys.readouterr().out


def model_state(recovery):
    reader = CheckpointReader(checked(recovery))
    return {reader.materialize(k): reader.materialize(v) for k, v in reader.field("model")["items"]}


def test_step_console_and_persisted_estimates_match_at_ten_updates(block8_fixture, capsys):
    f = block8_fixture
    cfg = read(f["config"])
    for name, steps in (("codec", 10), ("training", 12)):
        cfg[name].update(steps=steps, validation_interval=10, checkpoint_interval=10)
    write_json(f["config"], cfg)
    result = run_workflow(
        resolve(f["config"]),
        experiment_dir=f["folder"] / "step reporting",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert result["status"] == "completed", result
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith("block8_generator step")]
    assert len(lines) == 12
    assert "avg_step=pending eta=pending" in lines[8]
    assert "estimate_step=10" in lines[9] and "estimate_step=10" in lines[10]
    assert "eta=0d 0h 0m" in lines[11]
    folder = Path(result["experiment"]).parent
    metrics = next(folder.glob("attempts/generator/*/metrics.jsonl"))
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert rows[9]["progress"] == rows[10]["progress"]
    assert rows[9]["progress"]["sampled_steps"] == 10
    assert rows[9]["progress"]["average_seconds_per_step"] > 0
    assert rows[11]["progress"]["remaining_seconds"] == 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_sharded_interrupted_resume_matches_uninterrupted_update_and_preserves_failed_attempt(
    block8_fixture, dtype
):
    f = block8_fixture
    cfg = read(f["config"])
    cfg["dtype"] = dtype
    write_json(f["config"], cfg)
    f["settings"] = resolve(f["config"])
    full = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "uninterrupted",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert full["status"] == "completed", full
    interrupted = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "interrupted",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
        stop_after=2,
    )
    assert interrupted["status"] == "incomplete", interrupted
    state = read(interrupted["experiment"])
    old_receipt = checked(state["stages"]["generator"]["receipt"])
    old_recovery = state["recoveries"]["generator"]
    old_sha = digest(old_receipt)
    resumed = run_workflow(resume=interrupted["experiment"], execute=True)
    assert resumed["status"] == "completed", resumed
    assert digest(old_receipt) == old_sha and checked(old_recovery).is_file()
    full_state = read(full["experiment"])
    new_state = read(resumed["experiment"])
    a, b = (
        model_state(full_state["recoveries"]["generator"]),
        model_state(new_state["recoveries"]["generator"]),
    )
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key
    for ref in (full_state["recoveries"]["generator"], new_state["recoveries"]["generator"]):
        reader = CheckpointReader(checked(ref))
        assert len(reader.manifest["files"]) > 1
        assert reader.materialize(reader.field("training"))["step"] == 4
    before = read(old_receipt)["charged_work"]
    phase = next((Path(resumed["experiment"]).parent / "ledgers").glob("*generator.json"))
    assert read(phase)["charges"]["steps"] > before["steps"]


def test_complete_pinned_parquet_adapter_to_block_dataset_without_teacher_or_codec(
    block8_fixture, tulu_fixture
):
    f = block8_fixture
    source = f["folder"] / "pinned source.json"
    write_json(source, tulu_fixture["data"])
    cfg = read(f["config"])
    cfg["source"].update(
        config=str(source), max_candidates=40, max_source_index_bytes=2000000, max_candidate_bytes=1000000
    )
    write_json(f["config"], cfg)
    settings = resolve(f["config"])
    result = run_workflow(
        settings, experiment_dir=f["folder"] / "parquet preparation", execute=True, prepare_only=True
    )
    assert result["status"] == "completed", result
    state = read(result["experiment"])
    assert set(state["stages"]) == {"source", "prepare"}
    imported = read(checked(state["stages"]["source"]["receipt"]))["result"]
    storage = read(checked(imported["source_storage"]))
    assert storage["scan"]["complete"] and storage["scan"]["rows_read"] == 50
    assert storage["model_loaded"] is False
    with closing(BlockDataset(checked(result["dataset"]), settings)) as data:
        assert all(len(data.ids[s]) >= 2 for s in ("train", "dev", "test"))
        assert data.manifest["origins"] == {"software_fixture": sum(len(x) for x in data.ids.values())}
        assert data.parent(data.ids["train"][0]).provenance["reference_text"]


def test_corrupt_export_shard_rejected_before_model_allocation(block8_fixture, monkeypatch):
    f = block8_fixture
    result = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "corrupt export",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert result["status"] == "completed", result
    exported = read(checked(result["selected"]))
    weights = checked(exported["weights"])
    manifest = read(weights)
    shard = weights.parent / next(iter(manifest["files"]))
    with shard.open("r+b") as stream:
        stream.write(b"bad")
    monkeypatch.setattr(
        "transformers.Qwen3ForCausalLM.__init__",
        lambda *a, **k: pytest.fail("allocated before verifying shards"),
    )
    with pytest.raises(ValueError, match="corrupt/missing"):
        load_export(checked(result["selected"]), "cpu")


def test_changed_parent_index_and_changed_resume_configuration_are_rejected(block8_fixture):
    f = block8_fixture
    result = prepare_fixture(f, f["folder"] / "changed")
    assert result["status"] == "completed", result
    with pytest.raises(ValueError, match="overrides"):
        run_workflow(f["settings"], resume=result["experiment"], execute=True)
    file = checked(result["dataset"]).parent / "parents.sqlite"
    with sqlite3.connect(file) as db:
        db.execute("UPDATE views SET start=99 WHERE parent=1 AND start=0")
    with pytest.raises(ValueError, match="changed"):
        BlockDataset(checked(result["dataset"]), f["settings"])


def test_cross_split_group_and_duplicate_prompt_are_rejected_before_filtering(block8_fixture):
    f = block8_fixture
    original = Dataset(f["data"])
    for conflict in ("group", "prompt"):
        writer = ArtifactWriter(
            f["folder"] / ("conflict " + conflict), "native_data_v1", original.manifest["identity"]
        )
        first = f["parents"][0]
        for e in f["parents"]:
            if e.split == "test":
                continue
            if e.original_id == "dev-0":
                e = (
                    replace(e, group_id=first.group_id)
                    if conflict == "group"
                    else replace(e, prompt_ids=first.prompt_ids)
                )
            # Native writer already protects group overlap, so use a differing prompt alias
            # for the group case and mutate only the sealed source after writing below.
            if conflict == "group" and e.original_id == "dev-0":
                e = replace(e, group_id="temporary")
            writer.append(e)
        path = writer.finish()
        if conflict == "group":
            lines = (path.parent / "records.jsonl").read_text().splitlines()
            from ncm_lab.data.records import Example

            for i, line in enumerate(lines):
                row = json.loads(line)
                if row["example"]["original_id"] == "dev-0":
                    row["example"]["group_id"] = first.group_id
                    row["example_sha256"] = Example(**row["example"]).identity
                    lines[i] = json.dumps(row)
            (path.parent / "records.jsonl").write_text("\n".join(lines) + "\n")
            manifest = read(path)
            manifest["index_sha256"] = digest(path.parent / "records.jsonl")
            write_json(path, manifest)
        result = run_workflow(
            f["settings"],
            experiment_dir=f["folder"] / ("blocked " + conflict),
            execute=True,
            prepare_only=True,
            data_manifest=path,
            evaluation_manifest=f["evaluation"],
        )
        assert result["status"] == "failed" and "crosses" in result["failure"]["error"], result
        assert read(result["experiment"])["recoveries"]["generator"] is None


def test_plan_and_cli_do_not_load_models_or_create_runs(block8_fixture, monkeypatch, capsys):
    from ncm_lab import cli

    f = block8_fixture
    monkeypatch.setattr(
        "ncm_lab.teacher.model.load_original", lambda *a, **k: pytest.fail("plan loaded model")
    )
    assert main(["--config", str(f["config"]), "--plan-only"]) == 0
    assert cli.main(["block8", "--config", str(f["config"]), "--plan-only"]) == 0
    output = capsys.readouterr().out
    assert '"research_work_started": false' in output and '"model_allocated": false' in output
    assert not (f["folder"] / "runs").exists()


def test_budget_failure_is_preserved_and_never_starts_model(block8_fixture, monkeypatch):
    f = block8_fixture
    cfg = read(f["config"])
    cfg["budget"]["max_steps"] = 1
    write_json(f["config"], cfg)
    settings = resolve(f["config"])
    monkeypatch.setattr(
        "ncm_lab.teacher.model.load_original", lambda *a, **k: pytest.fail("source failure loaded model")
    )
    result = run_workflow(
        settings,
        experiment_dir=f["folder"] / "budget",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert result["status"] == "budget_stopped", result
    state = read(result["experiment"])
    receipt = read(checked(state["stages"]["prepare"]["receipt"]))
    assert receipt["charged_work"]["steps"] == 1 and "generator" not in state["stages"]
    assert (Path(result["experiment"]).parent / "attempts/prepare").is_dir()


def test_hard_sqlite_parent_index_limit_preserves_diagnostics(block8_fixture):
    import random

    f = block8_fixture
    source = Dataset(f["data"])
    writer = ArtifactWriter(
        f["folder"] / "large parent provenance", "native_data_v1", source.manifest["identity"]
    )
    sampler = random.Random(5)
    for parent in f["parents"]:
        if parent.split != "test":
            provenance = {**parent.provenance, "reference_text": sampler.randbytes(32000).hex()}
            writer.append(replace(parent, provenance=provenance))
    artifact = writer.finish()
    cfg = read(f["config"])
    cfg["data"]["max_index_bytes"] = 65536
    write_json(f["config"], cfg)
    result = run_workflow(
        resolve(f["config"]),
        experiment_dir=f["folder"] / "hard index limit",
        execute=True,
        data_manifest=artifact,
        evaluation_manifest=f["evaluation"],
    )
    assert result["status"] == "budget_stopped", result
    state = read(result["experiment"])
    receipt = checked(state["stages"]["prepare"]["receipt"])
    failed = read(receipt.parent / "dataset/storage_failure.json")
    assert failed["index_bytes"] <= 65536 and failed["failed_index_preserved"]
    assert not (receipt.parent / "dataset/manifest.json").exists() and "generator" not in state["stages"]


def test_teacher_codec_and_targets_are_frozen_parent_aligned_and_generator_is_offline(
    block8_fixture, monkeypatch
):
    from ncm_lab.block8.targets import TensorStore, FEATURES, TARGETS
    from ncm_lab.block8.training import load_codec
    from ncm_lab.teacher.model import prediction_states
    import ncm_lab.teacher.model as teachers

    f = block8_fixture
    result = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "offline target pipeline",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
        stop_after=1,
    )
    assert result["status"] == "incomplete", result
    state = read(result["experiment"])
    assert all(
        state["stages"][s]["status"] == "completed" for s in ("prepare", "teacher", "codec", "targets")
    )

    def receipt(stage):
        return read(checked(state["stages"][stage]["receipt"]))["result"]

    data_ref = receipt("prepare")["dataset"]
    features_ref = receipt("teacher")["features"]
    codec_ref = receipt("codec")["codec"]
    targets_ref = receipt("targets")["targets"]
    codec_sha = digest(checked(codec_ref))
    teacher = teachers.load_original(f["settings"]["model"], frozen=True)
    with (
        closing(BlockDataset(checked(data_ref), f["settings"])) as data,
        closing(TensorStore(checked(features_ref), f["settings"], data, FEATURES)) as features,
        closing(TensorStore(checked(targets_ref), f["settings"], data, TARGETS)) as targets,
    ):
        codec, meta = load_codec(checked(codec_ref), f["settings"], data, "cpu")
        assert meta["train_only_statistics"] and meta["teacher_head_frozen"]
        assert not any(p.requires_grad for p in teacher.parameters())
        assert not any(p.requires_grad for p in codec.parameters())
        assert set(targets.ids["train"]) == set(data.ids["train"])
        assert set(targets.ids["dev"]) == set(data.ids["dev"])
        assert not set(data.ids["test"]) & set(features.ids["train"] + features.ids["dev"])
        assert not set(data.ids["test"]) & set(targets.ids["train"] + targets.ids["dev"])
        key = features.ids["train"][0]
        parent = data.parent(key)
        actual = features.get(key, parent)
        expected = prediction_states(teacher, parent.prompt_ids, parent.answer_ids)
        torch.testing.assert_close(actual, expected)
        with torch.no_grad():
            code = codec.fine_codes(expected.float())
        torch.testing.assert_close(targets.get(key, parent), code)
        assert len(meta["scales"]) == 16 and all(len(v) == 8 for v in meta["scales"].values())
    real_load = teachers.load_original

    def guard(*args, **kwargs):
        assert kwargs.get("frozen") is False, "generator attempted an online teacher load"
        return real_load(*args, **kwargs)

    monkeypatch.setattr(teachers, "load_original", guard)
    monkeypatch.setattr(
        "ncm_lab.block8.training.load_codec", lambda *a, **k: pytest.fail("generator loaded codec")
    )
    resumed = run_workflow(resume=result["experiment"], execute=True)
    assert resumed["status"] == "completed", resumed
    assert digest(checked(codec_ref)) == codec_sha


def test_codec_exact_sharded_resume_and_independent_generator_recovery(block8_fixture):
    f = block8_fixture
    full = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "full codec reference",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
    )
    assert full["status"] == "completed", full
    stopped = run_workflow(
        f["settings"],
        experiment_dir=f["folder"] / "interrupted codec",
        execute=True,
        data_manifest=f["data"],
        evaluation_manifest=f["evaluation"],
        stop_after_codec=1,
    )
    assert stopped["status"] == "incomplete", stopped
    old = read(stopped["experiment"])
    codec_recovery = old["recoveries"]["codec"]
    assert old["recoveries"]["generator"] is None and "targets" not in old["stages"]
    resumed = run_workflow(resume=stopped["experiment"], execute=True)
    assert resumed["status"] == "completed", resumed
    a, b = read(full["experiment"]), read(resumed["experiment"])
    for phase in ("codec", "generator"):
        x, y = model_state(a["recoveries"][phase]), model_state(b["recoveries"][phase])
        for name in x:
            assert torch.equal(x[name], y[name]), (phase, name)
        left = CheckpointReader(checked(a["recoveries"][phase]))
        right = CheckpointReader(checked(b["recoveries"][phase]))
        ox = left.materialize(left.field("optimizer"))
        oy = right.materialize(right.field("optimizer"))
        assert ox["rounding_draws"] == oy["rounding_draws"]
        assert torch.equal(ox["rounding_rng"], oy["rounding_rng"])
        for key in ox["state"]:
            for stat in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(ox["state"][key][stat], oy["state"][key][stat])
    assert checked(codec_recovery).is_file()


def test_removed_custom_baseline_contracts_are_rejected(block8_fixture):
    f = block8_fixture
    cfg = read(f["config"])
    cfg["schema"] = "block8_workflow_v1"
    write_json(f["folder"] / "old config.json", cfg)
    with pytest.raises(ValueError, match="configuration"):
        resolve(f["folder"] / "old config.json")
    old = f["folder"] / "old custom export.json"
    write_json(old, dict(schema="block8_export_v1"))
    with pytest.raises(ValueError, match="family"):
        load_export(old, "cpu")
    manifest = f["folder"] / "old custom experiment.json"
    write_json(manifest, dict(schema="block8_experiment_v1"))
    with pytest.raises(ValueError, match="custom baseline"):
        run_workflow(resume=manifest)
