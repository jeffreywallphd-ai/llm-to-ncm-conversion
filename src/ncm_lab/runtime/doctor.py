import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess


def host_memory():
    try:
        if os.name == "nt":
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in ("total", "available", "page_total",
                              "page_available", "virtual_total", "virtual_available", "extended")]]
            status = Status()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            return {"total_bytes": status.total, "available_bytes": status.available, "reason": None}
        return {"total_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
                "available_bytes": os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), "reason": None}
    except (OSError, ValueError, AttributeError) as exc:
        return {"total_bytes": None, "available_bytes": None, "reason": str(exc)}


def doctor(profile="cpu"):
    packages = {}
    for p in ("torch", "transformers", "PyYAML", "sglang", "vllm"):
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    gpus, gpu_error = [], None
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=15, check=True)
            gpus = [line.strip() for line in proc.stdout.splitlines()]
        except (OSError, subprocess.SubprocessError) as exc:
            gpu_error = str(exc)
    torch_info = {"cuda_available": False, "precision": ["float32"], "cuda_runtime": None,
                  "bf16_supported": None, "kernels": "unprobed"}
    if packages["torch"]:
        import torch
        torch_info.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                          kernels="CPU SDPA; GPU kernels require parity/timing validation")
        if torch.cuda.is_available():
            torch_info["bf16_supported"] = torch.cuda.is_bf16_supported()
            torch_info["precision"] += ["float16"] + (["bfloat16"] if torch_info["bf16_supported"] else [])
            torch_info["free_device_bytes"] = torch.cuda.mem_get_info()[0]
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    cached = sorted(p.name for p in cache.glob("models--*") if p.is_dir())
    unified = any("GB10" in x or "Spark" in x for x in gpus)
    disk = shutil.disk_usage(Path.cwd())
    return {"profile": profile, "platform": platform.platform(), "cpu_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "python": platform.python_version(), "packages": packages,
            "gpus": gpus, "gpu_probe_error": gpu_error, "gpu_count": len(gpus), "unified_memory": unified,
            "memory_note": "Do not add device and host totals on unified memory systems; usable memory is measured at runtime.",
            "host_memory": host_memory(),
            "torch": torch_info, "disk_free_bytes": disk.free, "cached_models": cached,
            "code_isolation": {"docker_cli_available": bool(shutil.which("docker")), "verified": False,
                               "reason": "CLI presence is not an audited execution boundary"},
            "spark_ready": profile == "spark" and unified and torch_info["cuda_available"],
            "performance_validated": False}
