import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ncm_lab.config.stages import config
from ncm_lab.runtime.host_memory import memory_snapshot
from ncm_lab.runtime import doctor as diagnostic

GIB = 1024**3
BASELINE = 99007205760
RECLAIM_AWARE = 119936961536
FREE_ONLY = 55796248576


@pytest.fixture
def assessment(monkeypatch, memory_files, tmp_path):
    state = {
        "sizes": {0: 256 * GIB, 1: 128 * GIB},
        "selected": 1,
        "cuda": True,
        "disk": 2_000_000_000_000,
        "cuda_free_calls": 0,
    }

    def sample():
        return memory_snapshot(
            system="linux", reader=memory_files, sysconf=lambda key: pytest.fail("unexpected sysconf")
        )

    monkeypatch.setattr(diagnostic, "memory_snapshot", sample)
    monkeypatch.setattr(diagnostic.torch.cuda, "is_available", lambda: state["cuda"])
    monkeypatch.setattr(diagnostic.torch.cuda, "device_count", lambda: len(state["sizes"]))
    monkeypatch.setattr(diagnostic.torch.cuda, "current_device", lambda: state["selected"])
    monkeypatch.setattr(
        diagnostic.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="fixture GPU " + str(index), total_memory=state["sizes"][index]),
    )

    def low_cuda_free(*args):
        state["cuda_free_calls"] += 1
        return FREE_ONLY, 128 * GIB

    monkeypatch.setattr(diagnostic.torch.cuda, "mem_get_info", low_cuda_free)
    monkeypatch.setattr(
        diagnostic,
        "parameter_inventory",
        lambda *args: {"original": 8190735360, "generator": 8250600480, "codec": 6901504, "head": 622329856},
    )
    monkeypatch.setattr(diagnostic.shutil, "disk_usage", lambda path: SimpleNamespace(free=state["disk"]))
    value = config("configs/validation/spark_8b_profile.yaml")
    return state, value, sample, tmp_path


def test_synthetic_free_vs_available_reproduction(assessment, memory_files):
    state, value, _, path = assessment
    result = diagnostic.preflight(value, path, training=True)
    assert FREE_ONLY < BASELINE < RECLAIM_AWARE
    assert not result["blockers"], result
    assert result["minimum_weights_gradients_moments_bytes"] == BASELINE
    assert result["effective_capacity_bytes"] == RECLAIM_AWARE
    assert result["capacity_limiting_source"] == "proc_meminfo.MemAvailable"
    assert result["remaining_after_baseline_bytes"] == RECLAIM_AWARE - BASELINE
    assert result["memory_reserve_bytes"] == 0 and result["fp32_master_bytes"] == 0
    assert result["one_recovery_bytes_estimate"] == 82506004800
    assert result["bounded_optimizer_scratch_estimate_bytes"] == 167837696
    assert result["hardware"]["host_free_bytes"] == FREE_ONLY and state["cuda_free_calls"] == 0
    assert memory_files.calls["/proc/meminfo"] == 1
    assert "baseline" in result["assessment_scope"] and "unmeasured" in result["assessment_scope"]
    # Isolated old-policy control: same snapshot/device, but admission sees free pages.
    legacy = copy.deepcopy(result["hardware"])
    legacy.update(host_available_bytes=FREE_ONLY, host_available_source="legacy.free_pages")
    old = diagnostic.memory_capacity(legacy, "cuda", "spark")
    assert old["effective_capacity_bytes"] == FREE_ONLY < result["minimum_weights_gradients_moments_bytes"]
    assert old["capacity_limiting_source"] == "legacy.free_pages"
    json.dumps(result, allow_nan=False)


def test_genuine_low_available_blocks_despite_cache_and_swap(assessment, memory_files):
    _, value, _, path = assessment
    memory_files["/proc/meminfo"] = (
        f"MemTotal: {128 * 1024**2} kB\nMemFree: {1024**2} kB\nMemAvailable: {50 * 1024**2} kB\nCached: {80 * 1024**2} kB\nBuffers: {20 * 1024**2} kB\nSReclaimable: {30 * 1024**2} kB\nSwapFree: {500 * 1024**2} kB\n"
    )
    result = diagnostic.preflight(value, path, training=True)
    assert result["effective_capacity_bytes"] == 50 * GIB and result["remaining_after_baseline_bytes"] < 0
    detail = result["blocker_details"][0]
    assert detail["category"] == "insufficient_memory" and detail["required_bytes"] == BASELINE
    assert detail["available_bytes"] == 50 * GIB and detail["limiting_source"] == "proc_meminfo.MemAvailable"
    assert any(
        "weights, gradients and FP32 moments" in b and "required_bytes=" in b for b in result["blockers"]
    )


def test_selected_device_total_limits_spark_never_adds_pools(assessment):
    state, value, _, path = assessment
    state["sizes"][1] = 64 * GIB
    result = diagnostic.preflight(value, path, training=True)
    assert result["effective_capacity_bytes"] == result["device_total_bytes"] == 64 * GIB
    assert result["capacity_limiting_source"] == "cuda.device_properties[1].total_memory"
    assert result["hardware"]["selected_device_index"] == 1 and result["blockers"]
    assert result["effective_capacity_bytes"] != RECLAIM_AWARE + 64 * GIB
    assert not state["cuda_free_calls"]


def test_cgroup_headroom_and_indeterminate_usage(assessment, memory_files):
    _, value, _, path = assessment
    memory_files["/sys/fs/cgroup/memory.max"] = str(64 * GIB)
    memory_files["/sys/fs/cgroup/memory.current"] = str(60 * GIB)
    result = diagnostic.preflight(value, path, training=True)
    assert result["hardware"]["host_available_bytes"] == RECLAIM_AWARE
    assert result["effective_capacity_bytes"] == 4 * GIB and result["blockers"]
    assert result["capacity_limiting_source"] == "cgroup_v2.remaining_headroom:/sys/fs/cgroup"
    memory_files["/sys/fs/cgroup/memory.current"] = PermissionError()
    unknown = diagnostic.preflight(value, path, training=True)
    assert unknown["effective_capacity_bytes"] is None and unknown["remaining_after_baseline_bytes"] is None
    assert unknown["blocker_details"][0]["category"] == "memory_assessment_indeterminate"
    assert any("indeterminate process" in item for item in unknown["blockers"])


@pytest.mark.parametrize("kind", ["host", "device", "selection"])
def test_unknown_required_capacity_blocks_without_fabricated_zero(assessment, memory_files, kind):
    state, value, _, path = assessment
    if kind == "host":
        memory_files["/proc/meminfo"] = "MemAvailable: bad kB\n"
    elif kind == "device":
        state["sizes"][1] = None
    else:
        state["selected"] = 5
    result = diagnostic.preflight(value, path, training=True)
    assert result["effective_capacity_bytes"] is None and result["capacity_limiting_source"] is None
    assert result["blockers"] and result["blocker_details"][0]["available_bytes"] is None
    json.dumps(result, allow_nan=False)


def test_zero_host_is_known_zero(assessment, memory_files):
    _, value, _, path = assessment
    memory_files["/proc/meminfo"] = "MemAvailable: 0 kB\n"
    result = diagnostic.preflight(value, path, training=True)
    assert result["effective_capacity_bytes"] == 0
    assert result["blocker_details"][0]["category"] == "insufficient_memory"


def test_cuda_and_disk_block_independently(assessment):
    state, value, _, path = assessment
    state["disk"] = 1
    disk = diagnostic.preflight(value, path, training=True)
    assert disk["effective_capacity_bytes"] == RECLAIM_AWARE
    assert any("free disk" in b for b in disk["blockers"])
    state["disk"] = 2_000_000_000_000
    state["cuda"] = False
    unavailable = diagnostic.preflight(value, path, training=True)
    assert any("CUDA unavailable" in b for b in unavailable["blockers"])
    assert unavailable["effective_capacity_bytes"] is None


def test_discrete_gpu_and_cpu_policy_are_preserved(assessment, memory_files):
    _, value, _, path = assessment
    value["hardware_profile"] = "cuda"
    memory_files["/proc/meminfo"] = "MemAvailable: broken\n"
    gpu = diagnostic.preflight(value, path, training=True)
    assert not gpu["blockers"] and gpu["effective_capacity_bytes"] == 128 * GIB
    assert set(gpu["capacity_operands"]) == {"device_total_bytes"}
    memory_files["/proc/meminfo"] = "MemAvailable: 117125939 kB\n"
    value.update(device="cpu", hardware_profile="cpu")
    cpu = diagnostic.preflight(value, path, training=True)
    assert any("full 8B CPU training not authorized" in b for b in cpu["blockers"])
    value["stage"] = "generation"
    inference = diagnostic.preflight(value, path, training=False)
    assert not inference["blockers"] and inference["effective_capacity_bytes"] == RECLAIM_AWARE
    value["stage"] = "codec"
    codec = diagnostic.preflight(value, path)
    assert codec["minimum_weights_gradients_moments_bytes"] == (622329856 + 6901504) * 2 + 6901504 * 10


def test_doctor_and_observe_share_definitions_not_rss_capacity(assessment, memory_files):
    import ncm_lab.runtime.memory as observed

    _, _, sample, _ = assessment
    snap = sample()
    doctor = diagnostic.doctor("spark", memory=snap)
    report = observed.observe(memory=snap, process_memory=lambda: (123, 456))
    for key in snap:
        assert doctor[key] == report[key]
    assert report["host_available_bytes"] == RECLAIM_AWARE
    assert report["process_rss_bytes"] == 123 and report["process_lifetime_peak_rss_bytes"] == 456
    assert "process lifetime" in observed.__doc__
    assert not any("psutil" in p for p in Path("pyproject.toml").read_text().splitlines())


@pytest.mark.parametrize("blocked", [False, True])
def test_cli_json_status_without_training_or_download(assessment, memory_files, monkeypatch, capsys, blocked):
    import urllib.request
    import ncm_lab.teacher.model as models
    import ncm_lab.training.engine as engine
    import ncm_lab.runtime.prepare as preparation
    from ncm_lab.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("non-training CLI executed model/training/acquisition")

    monkeypatch.setattr(models, "load_original", forbidden)
    monkeypatch.setattr(engine, "fit", forbidden)
    monkeypatch.setattr(preparation, "prepare", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    if blocked:
        memory_files["/proc/meminfo"] = "MemAvailable: 4194304 kB\n"
    code = main(["preflight", "--config", "configs/validation/spark_8b_profile.yaml"])
    report = json.loads(capsys.readouterr().out)
    assert code == (2 if blocked else 0) and report["status"] == ("blocked" if blocked else "completed")
    assert report["capacity_guard_applied"] and "baseline" in report["assessment_scope"]
    assert main(["doctor", "--profile", "spark"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["host_available_bytes"] == report["hardware"]["host_available_bytes"]


def test_persisted_preflight_snapshot_is_not_replaced_by_later_close_observation(
    assessment, memory_files, monkeypatch
):
    import ncm_lab.runtime.runner as runner
    import ncm_lab.runtime.memory as observed
    import ncm_lab.training.engine as engine
    from ncm_lab.runtime.io import read, digest

    _, original, sample, path = assessment
    value = copy.deepcopy(original)
    value.update(inputs={}, output_ref=str(path / "unpublished.json"), run_root=str(path / "runs"))
    memory_files["/proc/meminfo"] = "MemAvailable: 4194304 kB\n"
    monkeypatch.setattr(observed, "memory_snapshot", sample)
    monkeypatch.setattr(observed, "_process_memory", lambda: (123, 456))
    real = diagnostic.preflight

    def assessment_then_change(*args, **kwargs):
        report = real(*args, **kwargs)
        memory_files["/proc/meminfo"] = "MemAvailable: 117125939 kB\n"
        return report

    monkeypatch.setattr(runner, "preflight", assessment_then_change)
    monkeypatch.setattr(engine, "fit", lambda *a, **kw: pytest.fail("blocked runner trained"))
    result = runner.execute(value)
    assert (
        result["status"] == "blocked"
        and result["model_loaded"] is False
        and result["charged_work"]["steps"] == 0
    )
    report_path = Path(result["path"]) / "preflight.json"
    report = read(report_path)
    assert report["hardware"]["host_available_bytes"] == report["effective_capacity_bytes"] == 4 * GIB
    assert result["memory"]["host_available_bytes"] == RECLAIM_AWARE
    assert not Path(value["output_ref"]).exists()
    before = digest(report_path)
    observed.observe(memory=sample(), process_memory=lambda: (999, 999))
    assert digest(report_path) == before


def test_historical_monitoring_records_need_no_new_memory_fields(tmp_path, monkeypatch):
    from dataclasses import asdict
    from ncm_lab.runtime.provenance import Run
    from ncm_lab.runtime.budget import Budget
    from ncm_lab.config.architecture import tiny_architecture
    from ncm_lab.validation.fixtures import model_spec
    from ncm_lab.monitoring.records import Measure
    from ncm_lab.monitoring.projector import Projector
    import ncm_lab.monitoring.projector as projector_module
    import ncm_lab.runtime.provenance as provenance
    import ncm_lab.runtime.memory as observed

    legacy = {
        "host_available_bytes": 123,
        "process_rss_bytes": 10,
        "process_lifetime_peak_rss_bytes": 20,
        "scope": "observed process memory",
    }
    monkeypatch.setattr(provenance, "doctor", lambda *a: {"host_available_bytes": 123})
    monkeypatch.setattr(observed, "observe", lambda: legacy)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(projector_module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1000000000))
    value = {
        "stage": "generator",
        "model": model_spec(),
        "architecture": tiny_architecture().to_dict(),
        "device": "cpu",
        "dtype": "float32",
        "run_root": str(tmp_path / "runs"),
        "budget": asdict(Budget("legacy-memory-fixture")),
    }
    run = Run(value)
    run.recorder.metric({"resources/host_available_bytes": Measure(123, unit="bytes")}, terminal=True)
    result = run.close("completed", {})
    before = (run.path / "summary.json").read_bytes()
    view = Projector(run.path.parents[1], tmp_path / "view", attempt_dir=run.path, headroom_bytes=0)
    try:
        assert view.poll() > 0
        assert result["memory"] == legacy and "host_available_source" not in result["memory"]
    finally:
        view.close()
    assert (run.path / "summary.json").read_bytes() == before
