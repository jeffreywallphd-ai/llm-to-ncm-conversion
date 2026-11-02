"""Observed process/host memory; a process lifetime peak is not an isolated step peak."""

import ctypes
import os
import sys
from ncm_lab.runtime.host_memory import memory_snapshot, host_available as host_available


def _process_memory():
    current = peak = None
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t)
                for n in (
                    "peak_ws",
                    "ws",
                    "peak_page_pool",
                    "page_pool",
                    "peak_nonpage_pool",
                    "nonpage_pool",
                    "pagefile",
                    "peak_pagefile",
                    "private",
                )
            ]

        value = Counters()
        value.cb = ctypes.sizeof(value)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()), ctypes.byref(value), value.cb
        ):
            current = value.ws
            peak = value.peak_ws
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm") as stream:
                current = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return current, peak


def observe(*, memory=None, process_memory=None):
    snapshot = memory_snapshot() if memory is None else memory
    warnings = list(snapshot["memory_warnings"])
    try:
        current, peak = (process_memory or _process_memory)()
    except (OSError, ValueError, AttributeError, IndexError, OverflowError, ImportError, NotImplementedError):
        current = peak = None
        warnings.append("process RSS observation unavailable; not used as host capacity")
    return {
        **snapshot,
        "memory_warnings": warnings,
        "process_rss_bytes": current,
        "process_lifetime_peak_rss_bytes": peak,
        "scope": "observed process memory; Spark host/device capacities must not be summed",
    }
