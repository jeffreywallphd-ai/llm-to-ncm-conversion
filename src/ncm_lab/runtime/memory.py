"""Observed process/host memory; a process lifetime peak is not an isolated step peak."""

import ctypes
import os
import sys
from ncm_lab.runtime.doctor import host_available


def observe():
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
    return {
        "host_available_bytes": host_available(),
        "process_rss_bytes": current,
        "process_lifetime_peak_rss_bytes": peak,
        "scope": "observed process memory; Spark host/device capacities must not be summed",
    }
