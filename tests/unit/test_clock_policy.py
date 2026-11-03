"""Fake-clock accounting and explicit source-transition checks, no live capacity inputs."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import ncm_lab.runtime.budget as budgets
from ncm_lab.runtime.budget import Budget, Meter, BudgetStop, phase_wall_report
from ncm_lab.runtime.io import read, atomic_json, implementation_hash
from ncm_lab.config.stages import config
from ncm_lab.training.checkpoints import save_checkpoint, CheckpointReader, native_identity
from ncm_lab.training.recovery_compatibility import verify_recovery, PREDECESSOR, POLICY


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budgets, "time", SimpleNamespace(perf_counter=lambda: state.now))
    return state


def meter(tmp_path, budget, name="run"):
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)
    return Meter(budget, folder, tmp_path / "ledger")


def test_omitted_policy_keeps_both_wall_limits(tmp_path, clock):
    budget = Budget("bounded", max_wall_seconds=10, max_phase_wall_seconds=20)
    first = meter(tmp_path, budget)
    assert first.wall_reservation == 10
    clock.now = 10
    with pytest.raises(BudgetStop, match="per-run wall"):
        first.check()
    meter(tmp_path, budget, "second")  # Two unresolved finite reservations fill the cap.
    before = read(first.path)
    with pytest.raises(BudgetStop, match="aggregate wall reservation"):
        meter(tmp_path, budget, "third")
    assert read(first.path) == before


@pytest.mark.parametrize("value", [0, 1, "false", "true", None, [], 0.0])
def test_switch_is_strict_boolean(value):
    with pytest.raises(ValueError, match="must be boolean"):
        Budget("invalid", enforce_wall_time_limits=value)


@pytest.mark.parametrize("value", [0, -1, None, float("inf"), float("nan"), True])
def test_disabled_mode_still_validates_numeric_wall_settings(value):
    with pytest.raises(ValueError, match="positive finite budget"):
        Budget("invalid", enforce_wall_time_limits=False, max_wall_seconds=value)


@pytest.mark.parametrize("enforced", [False, True])
def test_actual_reservation_settlement_and_idempotent_close(tmp_path, clock, enforced):
    item = meter(tmp_path, Budget("settlement", max_wall_seconds=10, enforce_wall_time_limits=enforced))
    assert read(item.path)["wall"] == (10 if enforced else 0)
    clock.now = 7
    item.close()
    closed = read(item.path)
    assert closed["wall"] == 7
    assert closed["wall_attempts"][item.attempt]["elapsed_seconds"] == 7
    clock.now = 999
    item.close()
    assert read(item.path) == closed
    assert item.wall_report()["attempt_observed_seconds"] == 7


def test_disabled_mode_preserves_legacy_spending_and_caps_after_huge_clock_advance(tmp_path, clock):
    budget = Budget("legacy", max_wall_seconds=10, max_phase_wall_seconds=30, max_steps=2, max_phase_steps=3)
    folder = tmp_path / "ledger"
    folder.mkdir()
    caps = {k: v for k, v in asdict(budget).items() if k.startswith("max_phase_")}
    old = {
        "caps": caps,
        "wall": 1000.0,
        "charges": {"steps": 1, "source_tokens": 9, "answer_positions": 0, "canvas_positions": 0},
        "runs": [str(tmp_path / "historical")],
    }
    atomic_json(folder / "legacy.json", old)
    item = meter(tmp_path, replace(budget, enforce_wall_time_limits=False))
    clock.now = 10**8
    item.check()
    item.charge(steps=2)
    item.check_disk()
    state = read(item.path)
    assert state["wall"] == 1000 and state["caps"] == old["caps"]
    assert state["runs"][0] == old["runs"][0] and state["charges"]["source_tokens"] == 9
    assert item.wall_report()["numeric_wall_limits_active"] is False
    item.close()
    state = read(item.path)
    assert state["wall"] == 1000 + 10**8 and state["charges"]["steps"] == 3
    report = phase_wall_report(state)
    assert report["new_attempts_closed_observed_seconds"] == 10**8
    assert report["new_attempts_active_reserved_seconds"] == 0
    assert (
        report["legacy_charges_and_reservations_seconds"] == 1000
        and report["legacy_split_status"] == "unknown"
    )
    with pytest.raises(BudgetStop, match="aggregate wall"):
        meter(tmp_path, budget, "finite")


def test_unclosed_disabled_attempt_is_unknown_and_blocks_finite_transition(tmp_path, clock):
    budget = Budget("crash", max_wall_seconds=10, max_phase_wall_seconds=30, enforce_wall_time_limits=False)
    crashed = meter(tmp_path, budget)
    crashed.charge(source=7)
    next_meter = meter(tmp_path, budget, "next")
    next_meter.close()
    state = read(crashed.path)
    assert state["wall"] == 0 and state["charges"]["source_tokens"] == 7
    assert state["wall_attempts"][crashed.attempt]["elapsed_seconds"] is None
    assert phase_wall_report(state)["incomplete_unbounded_attempts"] == [crashed.attempt]
    with pytest.raises(BudgetStop, match="accounting incomplete"):
        meter(tmp_path, replace(budget, enforce_wall_time_limits=True), "finite")
    assert read(crashed.path) == state
    # A genuinely observed close resolves only this attempt, never a guessed duration.
    clock.now = 5
    crashed.close()
    finite = meter(tmp_path, replace(budget, enforce_wall_time_limits=True), "finite")
    assert read(finite.path)["wall"] == 15


@pytest.mark.parametrize("scope", ["run", "phase"])
def test_non_time_work_and_storage_still_enforced(tmp_path, clock, scope):
    budget = Budget(
        "work",
        max_steps=1,
        max_phase_steps=2,
        max_storage_bytes=10,
        max_phase_storage_bytes=20,
        enforce_wall_time_limits=False,
    )
    first = meter(tmp_path, budget)
    clock.now = 10**7
    first.charge(steps=1)
    if scope == "run":
        with pytest.raises(BudgetStop, match="steps budget"):
            first.charge(steps=1)
        (first.run_dir / "file").write_bytes(b"x" * 11)
        with pytest.raises(BudgetStop, match="per-run storage"):
            first.check_disk()
    else:
        (first.run_dir / "file").write_bytes(b"x" * 10)
        first.close()
        second = meter(tmp_path, budget, "second")
        second.charge(steps=1)
        second.close()
        third = meter(tmp_path, budget, "third")
        with pytest.raises(BudgetStop, match="steps budget"):
            third.charge(steps=1)
        (second.run_dir / "file").write_bytes(b"x" * 10)
        with pytest.raises(BudgetStop, match="aggregate retained storage"):
            third.check_disk(additional=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        first.charge(source=True)


def test_only_real_shared_budget_opts_out_and_step_target_is_preserved():
    for name in ("generator_pilot", "codec_pilot", "teacher_features_pilot", "latent_targets_pilot"):
        cfg = config("configs/validation/" + name + ".yaml")
        assert cfg["budget"]["enforce_wall_time_limits"] is False
        assert cfg["budget"]["phase_id"] == "native-real-pilot-v1"
    generator = config("configs/validation/generator_pilot.yaml")
    assert generator["training"]["steps"] == 2
    assert generator["budget"]["max_wall_seconds"] == 1200
    assert generator["budget"]["max_phase_wall_seconds"] == 10800
    assert config("configs/validation/spark_8b_profile.yaml")["budget"]["enforce_wall_time_limits"] is True


def test_config_and_cli_preserve_the_explicit_switch_and_resume_option(tmp_path, monkeypatch, capsys):
    from ncm_lab.cli import main
    import ncm_lab.runtime.runner as runner

    value = config("configs/validation/generator_pilot.yaml")
    value["budget"]["enforce_wall_time_limits"] = "false"
    file = tmp_path / "invalid.json"
    atomic_json(file, value)
    with pytest.raises(ValueError, match="must be boolean"):
        config(file)
    args = [
        "generator",
        "train",
        "--config",
        "configs/validation/generator_pilot.yaml",
        "--recovery-compatibility",
        POLICY,
    ]
    assert main(args) == 2
    assert "requires an explicit --resume" in capsys.readouterr().err
    seen = {}

    def capture(value, resume, *, recovery_compatibility):
        seen.update(
            resume=str(resume),
            compatibility=recovery_compatibility,
            enforce=value["budget"]["enforce_wall_time_limits"],
        )
        return {"status": "completed", "scope": "CLI dispatch fixture; no training"}

    monkeypatch.setattr(runner, "execute", capture)
    assert main(args + ["--resume", str(tmp_path / "unread-fixture-pointer.json")]) == 0
    assert seen["compatibility"] == POLICY and seen["enforce"] is False


def recovery(tmp_path, metadata, inner_identity=None):
    identity = native_identity(metadata)
    saved = save_checkpoint(
        tmp_path,
        {
            "identity": inner_identity or identity,
            "metadata": metadata,
            "scheduler": {"schema": "constant_separate_rates_v1"},
            "scaler": None,
        },
    )
    return CheckpointReader(saved["manifest"])


@pytest.fixture
def transition_metadata():
    return {
        "implementation_sha256": PREDECESSOR,
        "phase_id": "native-real-pilot-v1",
        "model": "same",
        "source": {"codec": "same", "data": "same"},
        "training_contract": {"objective": "steps", "optimizer": "stochastic"},
        "dtype": "bfloat16",
    }


def test_only_reviewed_transition_is_allowed_with_explicit_option(tmp_path, transition_metadata):
    old = recovery(tmp_path, transition_metadata)
    current = {**transition_metadata, "implementation_sha256": implementation_hash()}
    with pytest.raises(ValueError, match="unreviewed implementation"):
        verify_recovery(old, current)
    receipt = verify_recovery(old, current, POLICY)
    assert receipt["predecessor_implementation_sha256"] == PREDECESSOR
    assert receipt["current_implementation_sha256"] == implementation_hash()
    assert Path(receipt["parent_checkpoint"]["manifest"]) == old.path


@pytest.mark.parametrize("field", ["phase_id", "model", "source", "training_contract", "dtype"])
def test_transition_never_ignores_other_identity_fields(tmp_path, transition_metadata, field):
    old = recovery(tmp_path, transition_metadata)
    current = {**transition_metadata, "implementation_sha256": implementation_hash(), field: "changed"}
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_recovery(old, current, POLICY)


def test_unknown_predecessor_and_changed_successor_fail(tmp_path, transition_metadata, monkeypatch):
    unknown = {**transition_metadata, "implementation_sha256": "f" * 64}
    old = recovery(tmp_path / "unknown", unknown)
    current = {**transition_metadata, "implementation_sha256": implementation_hash()}
    with pytest.raises(ValueError, match="unreviewed implementation"):
        verify_recovery(old, current, POLICY)
    old = recovery(tmp_path / "known", transition_metadata)
    import ncm_lab.training.recovery_compatibility as compatibility

    monkeypatch.setattr(compatibility, "implementation_hash", lambda: "b" * 64)
    with pytest.raises(ValueError, match="receipt does not approve"):
        verify_recovery(old, current, POLICY)


def test_old_metadata_must_match_its_own_identity(tmp_path, transition_metadata):
    reader = recovery(tmp_path, transition_metadata, inner_identity="inconsistent")
    with pytest.raises(ValueError, match="metadata/identity inconsistency"):
        verify_recovery(
            reader, {**transition_metadata, "implementation_sha256": implementation_hash()}, POLICY
        )


def test_wave_rejects_disabled_child_instead_of_hiding_outer_deadline(monkeypatch):
    import ncm_lab.config.waves as waves

    stages = {name: name for name in waves.STAGES if name != "preflight"}
    limits = dict(
        max_wave_wall_seconds=100,
        max_controller_seconds=10,
        max_phase_controller_seconds=20,
        max_log_bytes=1000,
        reserve_free_bytes=1,
    )
    monkeypatch.setattr(
        waves,
        "read",
        lambda path: dict(
            schema="native_wave_v1",
            phase_id="wave",
            run_root="runs",
            notes="fixture",
            limits=limits,
            stages=stages,
        ),
    )
    monkeypatch.setattr(
        waves,
        "config",
        lambda path: dict(
            stage="prepare" if str(path).startswith("prepare_") else path,
            budget=asdict(Budget("wave", enforce_wall_time_limits=False)),
            model={"fixture": True},
        ),
    )
    with pytest.raises(ValueError, match="outer execution deadlines"):
        waves.wave_config("unused.json")
