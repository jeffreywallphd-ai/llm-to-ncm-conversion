import json
import pytest
from ncm_lab.runtime.host_memory import memory_snapshot, parse_meminfo, MAX_ANCESTORS


def snapshot(files, **kwargs):
    return memory_snapshot(
        system="linux", reader=files, sysconf=lambda key: pytest.fail("unexpected fallback"), **kwargs
    )


def test_rounded_reproduction_and_fresh_observations(memory_files):
    first = snapshot(memory_files)
    assert first["host_free_bytes"] == 55796248576
    assert first["host_available_bytes"] == first["host_effective_available_bytes"] == 119936961536
    assert first["host_available_source"] == "proc_meminfo.MemAvailable"
    assert memory_files.calls["/proc/meminfo"] == 1
    memory_files["/proc/meminfo"] = "MemAvailable: 0 kB\nMemTotal: 10 kB\nMemFree: 3 kB\n"
    second = snapshot(memory_files)
    assert (
        second["host_available_bytes"] == 0 and second["host_available_source"] == "proc_meminfo.MemAvailable"
    )
    assert memory_files.calls["/proc/meminfo"] == 2
    json.dumps(second, allow_nan=False)


def test_units_whitespace_reordering_and_optional_fields(memory_files):
    memory_files["/proc/meminfo"] = (
        "\tMemAvailable :   7   kB \nUnknownCounter: nope\nMemAvailable-extra: invalid\nHugePages_Total: 5\nCached: bad MB\n"
    )
    value = snapshot(memory_files)
    assert value["host_available_bytes"] == 7168 and type(value["host_available_bytes"]) is int
    assert value["host_total_bytes"] is None and value["host_free_bytes"] is None
    assert value["host_cache_observations"]["cached_bytes"] is None
    assert value["memory_warnings"]


@pytest.mark.parametrize(
    "line",
    [
        "MemAvailable: -1 kB",
        "MemAvailable: nan kB",
        "MemAvailable: 5 MB",
        "MemAvailable: 5 KB",
        "MemAvailable: 5 kB trailing",
        "MemAvailable 5 kB",
        "MemAvailable: 11 kB",
        "MemAvailable: 5 kB\nMemAvailable: 5 kB",
        "MemAvailable: 18446744073709551616 kB",
    ],
)
def test_invalid_required_value_is_unknown_not_free_fallback(memory_files, line):
    memory_files["/proc/meminfo"] = "MemTotal: 10 kB\nMemFree: 2 kB\n" + line + "\n"
    value = snapshot(memory_files)
    assert value["host_available_bytes"] is None and value["host_available_source"] is None
    assert value["memory_warnings"]
    json.dumps(value, allow_nan=False)


def test_contradictory_core_values_and_swap_are_not_capacity(memory_files):
    memory_files["/proc/meminfo"] = "MemTotal: 10 kB\nMemFree: 11 kB\nMemAvailable: 8 kB\n"
    assert snapshot(memory_files)["host_available_bytes"] is None
    memory_files["/proc/meminfo"] = (
        "MemTotal: 10 kB\nMemFree: 1 kB\nMemAvailable: 8 kB\nCached: 9 kB\nSwapTotal: 2 kB\nSwapFree: 5 kB\n"
    )
    value = snapshot(memory_files)
    assert value["host_available_bytes"] == 8192
    assert value["host_cache_observations"]["swap_free_bytes"] is None


def test_missing_memavailable_uses_labeled_memfree_from_same_read(memory_files):
    memory_files["/proc/meminfo"] = "MemTotal: 10 kB\nMemFree: 2 kB\n"
    value = snapshot(memory_files)
    assert value["host_available_bytes"] == 2048
    assert value["host_available_source"] == "proc_meminfo.MemFree.conservative_fallback"
    assert "conservative" in " ".join(value["memory_warnings"])
    assert memory_files.calls["/proc/meminfo"] == 1


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError(), OSError()])
def test_procfs_failure_uses_validated_free_pages(memory_files, error):
    memory_files["/proc/meminfo"] = error
    value = memory_snapshot(
        system="linux",
        reader=memory_files,
        sysconf=lambda key: {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 123}[key],
    )
    assert value["host_available_bytes"] == 4096 * 123
    assert value["host_available_source"] == "sysconf.SC_AVPHYS_PAGES.conservative_fallback"


@pytest.mark.parametrize("error", [ValueError(), OSError(), KeyError(), NotImplementedError()])
def test_failed_sysconf_is_unknown(memory_files, error):
    memory_files["/proc/meminfo"] = FileNotFoundError()

    def unsupported(key):
        raise error

    value = memory_snapshot(system="linux", reader=memory_files, sysconf=unsupported)
    assert value["host_available_bytes"] is None and value["host_effective_available_bytes"] is None
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("page,free", [(0, 2), (4096, -1), (True, 2), (4096, 1.5)])
def test_invalid_sysconf_values_do_not_invent_capacity(page, free):
    value = memory_snapshot(
        system="darwin", sysconf=lambda key: {"SC_PAGE_SIZE": page, "SC_AVPHYS_PAGES": free}[key]
    )
    assert value["host_available_bytes"] is None


def test_nonlinux_mocked_branches_and_public_wrappers(memory_files, monkeypatch):
    zero = memory_snapshot(
        system="windows",
        windows_memory=lambda: (10000, 0),
        reader=lambda p: pytest.fail("Windows read procfs"),
    )
    assert (
        zero["host_available_bytes"] == 0
        and zero["host_available_source"] == "GlobalMemoryStatusEx.ullAvailPhys"
    )

    def failed():
        raise OSError("OS failure")

    assert memory_snapshot(system="windows", windows_memory=failed)["host_available_bytes"] is None
    assert memory_snapshot(system="win32", windows_memory=lambda: (1, 2))["host_available_bytes"] is None
    posix = memory_snapshot(
        system="darwin", sysconf=lambda key: {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 0}[key]
    )
    assert posix["host_available_bytes"] == 0 and "conservative_fallback" in posix["host_available_source"]
    import ncm_lab.runtime.host_memory as host
    from ncm_lab.runtime.doctor import host_available as doctor_available
    from ncm_lab.runtime.memory import host_available as observe_available

    value = snapshot(memory_files)
    value["host_effective_available_bytes"] = 123
    monkeypatch.setattr(host, "memory_snapshot", lambda: value)
    assert host.host_available() == doctor_available() == observe_available() == 119936961536


def group(files, path, maximum, current=None):
    files[path + "/memory.max"] = str(maximum)
    if current is not None:
        files[path + "/memory.current"] = str(current)


@pytest.mark.parametrize(
    "leaf_max,leaf_current,parent_max,parent_current,expected,limiter",
    [
        (100, 20, "max", None, 80, "/sys/fs/cgroup/parent/leaf"),
        (100, 10, 80, 50, 30, "/sys/fs/cgroup/parent"),
        ("max", None, 80, 50, 30, "/sys/fs/cgroup/parent"),
        (0, 0, "max", None, 0, "/sys/fs/cgroup/parent/leaf"),
        (10, 20, "max", None, 0, "/sys/fs/cgroup/parent/leaf"),
    ],
)
def test_paired_child_and_ancestor_headroom(
    memory_files, leaf_max, leaf_current, parent_max, parent_current, expected, limiter
):
    memory_files["/proc/self/cgroup"] = "0::/parent/leaf\n"
    group(memory_files, "/sys/fs/cgroup/parent/leaf", leaf_max, leaf_current)
    group(memory_files, "/sys/fs/cgroup/parent", parent_max, parent_current)
    memory_files["/sys/fs/cgroup/parent/leaf/memory.stat"] = "file 99999999999\n"
    value = snapshot(memory_files)
    constraint = value["process_memory_constraint"]
    assert constraint["status"] == "finite" and constraint["remaining_headroom_bytes"] == expected
    assert constraint["limiting_group"] == limiter and value["host_effective_available_bytes"] == expected
    assert value["host_available_bytes"] == 119936961536
    assert "without reclaim credit" in constraint["headroom_policy"]
    assert not memory_files.calls["/sys/fs/cgroup/parent/leaf/memory.stat"]


@pytest.mark.parametrize(
    "maximum,current", [("bad", None), ("-1", None), (100, None), (100, "bad"), (100, -1)]
)
def test_indeterminate_limit_or_current_blocks_host_dependent_assessment(memory_files, maximum, current):
    memory_files["/proc/self/cgroup"] = "0::/leaf\n"
    group(memory_files, "/sys/fs/cgroup/leaf", maximum, current)
    value = snapshot(memory_files)
    assert value["process_memory_constraint"]["status"] == "indeterminate"
    assert value["process_memory_constraint"]["blocking"] and value["host_effective_available_bytes"] is None
    assert value["host_available_bytes"] == 119936961536


def test_root_absence_v1_and_discovery_failures_are_distinct(memory_files):
    value = snapshot(memory_files)
    assert value["process_memory_constraint"]["status"] == "no_finite_observed"
    assert value["process_memory_constraint"]["hierarchy"] == "cgroup_v2"
    assert not value["process_memory_constraint"]["blocking"]
    memory_files["/proc/self/cgroup"] = "4:memory:/legacy\n"
    assert snapshot(memory_files)["process_memory_constraint"]["status"] == "unsupported"
    memory_files["/proc/self/cgroup"] = PermissionError()
    unknown = snapshot(memory_files)
    assert unknown["process_memory_constraint"]["status"] == "indeterminate"
    assert unknown["host_available_bytes"] == 119936961536 and unknown["memory_warnings"]


def test_mount_roots_escapes_and_bounded_ancestry(memory_files):
    memory_files["/proc/self/cgroup"] = "0::/tenant work/job/leaf\n"
    memory_files["/proc/self/mountinfo"] = (
        "1 0 0:1 /tenant\\040work /cg\\040mount\\134name rw - cgroup2 cgroup rw\n"
    )
    mount = "/cg mount\\name"
    group(memory_files, mount + "/job/leaf", "max")
    group(memory_files, mount + "/job", 100, 60)
    group(memory_files, mount, "max")
    value = snapshot(memory_files)
    constraint = value["process_memory_constraint"]
    assert (
        constraint["resolved_group"] == mount + "/job/leaf" and constraint["remaining_headroom_bytes"] == 40
    )
    assert [r["path"] for r in constraint["inspected"]] == [mount + "/job/leaf", mount + "/job", mount]
    assert "hide higher" in " ".join(value["memory_warnings"])


def test_namespace_relative_mapping_requires_current_process_proof(memory_files):
    memory_files["/proc/self/cgroup"] = "0::/leaf\n"
    memory_files["/proc/self/mountinfo"] = "1 0 0:1 /hidden/container /delegated rw - cgroup2 cgroup rw\n"
    memory_files["/delegated/leaf/cgroup.procs"] = "41\n42\n"
    group(memory_files, "/delegated/leaf", 100, 25)
    group(memory_files, "/delegated", "max")
    result = snapshot(memory_files, pid=42)
    assert result["process_memory_constraint"]["remaining_headroom_bytes"] == 75
    assert result["process_memory_constraint"]["mapping"] == "namespace_relative_requires_membership"
    memory_files.calls.clear()
    result = snapshot(memory_files, pid=99)
    assert result["process_memory_constraint"]["status"] == "indeterminate"
    assert not memory_files.calls["/delegated/leaf/memory.max"]  # No unrelated-group limit probing.


def test_widest_mount_keeps_visible_parent_limits(memory_files):
    memory_files["/proc/self/cgroup"] = "0::/parent/leaf\n"
    memory_files["/proc/self/mountinfo"] = (
        "1 0 0:1 /parent /narrow rw - cgroup2 cgroup rw\n2 0 0:1 / /wide rw - cgroup2 cgroup rw\n"
    )
    group(memory_files, "/wide/parent/leaf", "max")
    group(memory_files, "/wide/parent", 100, 90)
    group(memory_files, "/wide", "max")
    value = snapshot(memory_files)
    assert value["process_memory_constraint"]["remaining_headroom_bytes"] == 10
    assert not any(path.startswith("/narrow/") for path in memory_files.calls)


@pytest.mark.parametrize("failure", [PermissionError(), ValueError("bad text")])
def test_unreadable_known_constraint_is_explicit(memory_files, failure):
    memory_files["/sys/fs/cgroup/memory.max"] = "100"
    memory_files["/sys/fs/cgroup/memory.current"] = failure
    assert snapshot(memory_files)["process_memory_constraint"]["blocking"]


def test_external_paths_and_probe_bounds_are_not_traversed(memory_files):
    memory_files["/proc/self/cgroup"] = "0::/../outside\n"
    assert snapshot(memory_files)["process_memory_constraint"]["status"] == "indeterminate"
    assert not memory_files.calls["/sys/fs/outside/memory.max"]
    parts = ["level"] * (MAX_ANCESTORS + 1)
    memory_files["/proc/self/cgroup"] = "0::/" + "/".join(parts) + "\n"
    for count in range(1, len(parts) + 1):
        group(memory_files, "/sys/fs/cgroup/" + "/".join(parts[:count]), "max")
    value = snapshot(memory_files)["process_memory_constraint"]
    assert len(value["inspected"]) == MAX_ANCESTORS and value["blocking"]


def test_parser_does_not_add_cache_or_swap():
    value = parse_meminfo(
        "MemAvailable: 1 kB\nCached: 9000 kB\nBuffers: 9000 kB\nSReclaimable: 9000 kB\nSwapFree: 9000 kB\n"
    )
    assert value["values"]["host_available_bytes"] == 1024


@pytest.mark.parametrize(
    "file,body",
    [
        ("/proc/self/cgroup", "0:/malformed\n"),
        ("/proc/self/mountinfo", "1 0 - cgroup2 x rw\n"),
        ("/proc/self/mountinfo", "1 0 0:1 / /bad\\999mount rw - cgroup2 x rw\n"),
        ("/proc/self/mountinfo", PermissionError()),
        ("/proc/self/cgroup", "0::/" + "x" * 5000 + "\n"),
    ],
)
def test_malformed_or_unreadable_hierarchy_is_not_reported_unlimited(memory_files, file, body):
    memory_files[file] = body
    value = snapshot(memory_files)
    assert value["host_available_bytes"] == 119936961536
    assert value["process_memory_constraint"]["status"] == "indeterminate"
    assert "discovery incomplete" in " ".join(value["memory_warnings"])


def test_missing_nonroot_limit_and_root_permission_are_indeterminate(memory_files):
    memory_files["/proc/self/cgroup"] = "0::/missing\n"
    first = snapshot(memory_files)
    assert first["process_memory_constraint"]["blocking"]
    memory_files["/proc/self/cgroup"] = "0::/\n"
    memory_files["/sys/fs/cgroup/memory.max"] = PermissionError()
    second = snapshot(memory_files)
    assert second["process_memory_constraint"]["blocking"]


def test_snapshot_does_not_import_psutil(memory_files, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "psutil" or name.startswith("psutil."):
            pytest.fail("mandatory psutil dependency introduced")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert snapshot(memory_files)["host_available_bytes"] == 119936961536
