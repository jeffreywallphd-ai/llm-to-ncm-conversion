"""Read-only host availability and visible cgroup-v2 headroom; no Torch dependency."""

import ctypes
import os
from pathlib import PurePosixPath
import re
import sys

MAX_BYTES = (1 << 64) - 1
MAX_ANCESTORS = 64
MAX_MOUNTS = 16
MEMINFO_FIELDS = {
    "MemTotal": "host_total_bytes",
    "MemFree": "host_free_bytes",
    "MemAvailable": "host_available_bytes",
    "Buffers": "buffers_bytes",
    "Cached": "cached_bytes",
    "SReclaimable": "sreclaimable_bytes",
    "SwapTotal": "swap_total_bytes",
    "SwapFree": "swap_free_bytes",
}


def unsigned_bytes(text):
    if not re.fullmatch(r"[0-9]+", text.strip()):
        raise ValueError("expected an unsigned integer byte count")
    value = int(text)
    if value > MAX_BYTES:
        raise ValueError("byte count exceeds unsigned 64-bit range")
    return value


def parse_meminfo(text):
    """One kernel snapshot; optional cache/swap observations never add capacity."""
    values, present, warnings = {}, set(), []
    for line in text.splitlines():
        key, separator, raw = line.partition(":")
        key = key.strip()
        if key not in MEMINFO_FIELDS:
            # A malformed required line is not an older kernel omitting the field.
            match = re.match(r"^\s*(MemAvailable|MemTotal|MemFree)(?:\s|$)", line)
            if not match:
                continue
            key = match[1]
        field = MEMINFO_FIELDS[key]
        if key in present:
            values[field] = None
            warnings.append("duplicate proc_meminfo." + key)
            continue
        present.add(key)
        match = re.fullmatch(r"\s*([0-9]+)\s+kB\s*", raw) if separator else None
        try:
            if not match:
                raise ValueError("expected nonnegative integer kB")
            values[field] = unsigned_bytes(str(int(match[1]) * 1024))
        except ValueError:
            values[field] = None
            warnings.append("invalid proc_meminfo." + key)
    total = values.get("host_total_bytes")
    for field in ("host_free_bytes", "host_available_bytes"):
        if total is not None and values.get(field) is not None and values[field] > total:
            values[field] = None
            values["host_available_bytes"] = None
            warnings.append("contradictory proc_meminfo: " + field + " exceeds MemTotal")
    swap = values.get("swap_total_bytes")
    if swap is not None and values.get("swap_free_bytes") is not None and values["swap_free_bytes"] > swap:
        values["swap_free_bytes"] = None
        warnings.append("contradictory optional SwapFree exceeds SwapTotal; no swap credited")
    return {"values": values, "present": present, "warnings": warnings}


def _read(reader, path, limit):
    if reader is None:
        with open(path, encoding="utf-8") as stream:
            text = stream.read(limit + 1)
    else:
        text = reader(path)
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError("memory observation exceeds bounded text size")
    return text


def _path(text):
    if not text.startswith("/") or "\0" in text or any(p in {".", ".."} for p in text.split("/")):
        raise ValueError("unsafe or namespace-external cgroup path")
    if len(text) > 4096:
        raise ValueError("cgroup path too long")
    return PurePosixPath(text)


def _mount_path(text):
    def decode(match):
        if match[1] not in {"040", "011", "012", "134"}:
            raise ValueError("unsupported mountinfo path escape")
        return chr(int(match[1], 8))

    if re.search(r"\\(?![0-7]{3})", text):
        raise ValueError("invalid mountinfo path escape")
    return _path(re.sub(r"\\([0-7]{3})", decode, text))


def parse_cgroup2_paths(cgroup_text, mountinfo_text):
    """Resolve mount-root-relative candidates; namespace-relative candidates need PID proof."""
    unified = []
    for line in cgroup_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise ValueError("malformed /proc/self/cgroup")
        if parts[:2] == ["0", ""]:
            unified.append(_path(parts[2]))
    if not unified:
        return None, []  # Legacy v1 is explicitly unsupported, not guessed from root files.
    if len(unified) != 1:
        raise ValueError("ambiguous unified process membership")
    group = unified[0]
    candidates = []
    for line in mountinfo_text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or not after.split() or after.split()[0] != "cgroup2":
            continue
        fields = before.split()
        if len(fields) < 6:
            raise ValueError("malformed cgroup2 mountinfo")
        boundary, mount = _mount_path(fields[3]), _mount_path(fields[4])
        if group.is_relative_to(boundary):
            relative = group.relative_to(boundary)
            mapping = "mount_root_relative"
        else:
            relative = group.relative_to("/")
            mapping = "namespace_relative_requires_membership"
        candidates.append({"mount": mount, "root": boundary, "leaf": mount / relative, "mapping": mapping})
        if len(candidates) > MAX_MOUNTS:
            raise ValueError("cgroup2 mount probe limit exceeded")
    # Prefer the widest visible matching ancestry, not a bind mount hiding parents.
    candidates.sort(
        key=lambda c: (c["mapping"] != "mount_root_relative", len(c["root"].parts), str(c["mount"]))
    )
    return group, candidates


def cgroup_constraint(reader=None, pid=None):
    result = {
        "hierarchy": "cgroup_v2",
        "status": "indeterminate",
        "blocking": False,
        "scope": "visible process ancestry only; namespace-hidden higher limits are not verified",
        "headroom_policy": "remaining headroom without reclaim credit",
        "remaining_headroom_bytes": None,
        "limiting_group": None,
        "inspected": [],
        "warnings": [],
    }
    try:
        group, candidates = parse_cgroup2_paths(
            _read(reader, "/proc/self/cgroup", 65536), _read(reader, "/proc/self/mountinfo", 1048576)
        )
        if group is None:
            result.update(hierarchy="unsupported_or_legacy", status="unsupported")
            result["warnings"].append("no cgroup-v2 membership; cgroup-v1 limits are not implemented")
            return result
        if not candidates:
            raise ValueError("no visible cgroup2 mount resolves this process")
        selected = None
        for candidate in candidates:
            if candidate["mapping"] == "mount_root_relative":
                selected = candidate
                break
            try:
                membership = _read(reader, str(candidate["leaf"] / "cgroup.procs"), 1048576)
                if str(os.getpid() if pid is None else pid) in membership.split():
                    selected = candidate
                    break
            except (OSError, ValueError):
                continue
        if selected is None:
            raise ValueError(
                "namespace-relative cgroup mount could not be verified against process membership"
            )
        result.update(
            process_cgroup=str(group),
            mount_point=str(selected["mount"]),
            mount_root=str(selected["root"]),
            resolved_group=str(selected["leaf"]),
            mapping=selected["mapping"],
        )
        result["warnings"].append("cgroup namespace or mount boundaries may hide higher ancestor constraints")
        path = selected["leaf"]
        headrooms = []
        for _ in range(MAX_ANCESTORS):
            entry = {
                "path": str(path),
                "memory_max_bytes": None,
                "memory_current_bytes": None,
                "remaining_headroom_bytes": None,
            }
            result["inspected"].append(entry)
            try:
                raw = _read(reader, str(path / "memory.max"), 4096).strip()
                if raw == "max":
                    entry["state"] = "no_finite_limit_at_this_level"
                else:
                    maximum = unsigned_bytes(raw)
                    entry["memory_max_bytes"] = maximum
                    current = unsigned_bytes(_read(reader, str(path / "memory.current"), 4096))
                    entry.update(
                        memory_current_bytes=current,
                        remaining_headroom_bytes=max(0, maximum - current),
                        state="finite",
                    )
                    headrooms.append((entry["remaining_headroom_bytes"], str(path)))
            except FileNotFoundError:
                if path == selected["mount"] and entry["memory_max_bytes"] is None:
                    entry["state"] = "visible_root_interface_absent"
                else:
                    entry["state"] = "indeterminate"
                    result["blocking"] = True
                    result["warnings"].append("required cgroup limit/current file missing at " + str(path))
            except (OSError, ValueError) as exc:
                entry["state"] = "indeterminate"
                result["blocking"] = True
                result["warnings"].append(
                    "unreadable/invalid cgroup limit or current at " + str(path) + ": " + type(exc).__name__
                )
            if path == selected["mount"]:
                break
            path = path.parent
        else:
            result["blocking"] = True
            result["warnings"].append("cgroup ancestry probe limit reached before mount boundary")
        if headrooms:
            result["remaining_headroom_bytes"], result["limiting_group"] = min(headrooms)
        result["status"] = (
            "indeterminate" if result["blocking"] else "finite" if headrooms else "no_finite_observed"
        )
    except (OSError, ValueError) as exc:
        result["warnings"].append("cgroup discovery incomplete: " + str(exc))
    return result


def _windows_memory():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "total",
                "available",
                "page_total",
                "page_available",
                "virtual_total",
                "virtual_available",
                "extended",
            )
        ]

    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("GlobalMemoryStatusEx failed")
    return value.total, value.available


def memory_snapshot(*, system=None, reader=None, sysconf=None, windows_memory=None, pid=None):
    """A fresh report. Raw host availability stays separate from process-limited headroom."""
    system = (sys.platform if system is None else system).lower()
    result = {
        "host_total_bytes": None,
        "host_free_bytes": None,
        "host_available_bytes": None,
        "host_available_source": None,
        "host_effective_available_bytes": None,
        "host_cache_observations": {},
        "memory_warnings": [],
    }
    warnings = result["memory_warnings"]
    fallback = True
    if system.startswith("linux"):
        try:
            parsed = parse_meminfo(_read(reader, "/proc/meminfo", 1048576))
            values = parsed["values"]
            warnings.extend(parsed["warnings"])
            result.update(
                {
                    key: values.get(key)
                    for key in ("host_total_bytes", "host_free_bytes", "host_available_bytes")
                }
            )
            result["host_cache_observations"] = {k: v for k, v in values.items() if not k.startswith("host_")}
            if "MemAvailable" in parsed["present"]:
                fallback = (
                    False  # Corrupt/contradictory required data is not a missing-kernel-field fallback.
                )
                if result["host_available_bytes"] is not None:
                    result["host_available_source"] = "proc_meminfo.MemAvailable"
            elif result["host_free_bytes"] is not None:
                result["host_available_bytes"] = result["host_free_bytes"]
                result["host_available_source"] = "proc_meminfo.MemFree.conservative_fallback"
                warnings.append(
                    "MemAvailable missing; conservative immediately-free estimate, without reclaim credit"
                )
                fallback = False
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append("proc_meminfo unavailable: " + type(exc).__name__)
        constraint = cgroup_constraint(reader, pid)
    elif system in {"win32", "windows"}:
        fallback = False
        try:
            total, available = (windows_memory or _windows_memory)()
            if (
                any(type(v) is not int or not 0 <= v <= MAX_BYTES for v in (total, available))
                or available > total
            ):
                raise ValueError("invalid Windows physical memory counters")
            result.update(
                host_total_bytes=total,
                host_available_bytes=available,
                host_available_source="GlobalMemoryStatusEx.ullAvailPhys",
            )
        except (OSError, AttributeError, ValueError) as exc:
            warnings.append("Windows host memory unavailable: " + type(exc).__name__)
        constraint = {
            "hierarchy": None,
            "status": "not_applicable",
            "blocking": False,
            "remaining_headroom_bytes": None,
            "warnings": [],
        }
    else:
        constraint = {
            "hierarchy": None,
            "status": "unsupported",
            "blocking": False,
            "remaining_headroom_bytes": None,
            "warnings": ["cgroup-v2 probing is Linux-only"],
        }
    if fallback:
        conf = sysconf if sysconf is not None else getattr(os, "sysconf", None)
        try:
            page, free = conf("SC_PAGE_SIZE"), conf("SC_AVPHYS_PAGES")
            if type(page) is not int or page <= 0 or type(free) is not int or free < 0:
                raise ValueError("invalid free-pages fallback")
            available = unsigned_bytes(str(page * free))
            total = result["host_total_bytes"]
            if total is not None and available > total:
                raise ValueError("fallback free memory exceeds observed total")
            result.update(
                host_free_bytes=available,
                host_available_bytes=available,
                host_available_source="sysconf.SC_AVPHYS_PAGES.conservative_fallback",
            )
            warnings.append("conservative sysconf free-pages fallback; not reclaim-aware MemAvailable")
        except (OSError, ValueError, TypeError, AttributeError, OverflowError, KeyError, NotImplementedError):
            warnings.append("host availability unknown; no usable free-pages fallback")
    result["process_memory_constraint"] = constraint
    warnings.extend(constraint["warnings"])
    available, headroom = result["host_available_bytes"], constraint["remaining_headroom_bytes"]
    if available is not None and not constraint["blocking"]:
        result["host_effective_available_bytes"] = (
            min(available, headroom) if headroom is not None else available
        )
    if available is None:
        warnings.append(
            "host availability measurement unavailable; real host-dependent admission cannot be assessed"
        )
    return result


def host_available():
    """Compatibility API: raw host estimate as integer bytes or None, never process-limited."""
    return memory_snapshot()["host_available_bytes"]
