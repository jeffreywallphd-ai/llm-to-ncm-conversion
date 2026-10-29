"""Detection, kernel execution, full training, and throughput are separate evidence."""

import ctypes
import os
import platform
import shutil
import sys
import torch


def host_available():
    if os.name == "nt":

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong)
                for n in (
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
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return value.available
    if hasattr(os, "sysconf"):
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return None


def doctor(profile="cpu"):
    import transformers

    devices = [
        {"name": torch.cuda.get_device_name(i), "bytes": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ]
    return {
        "profile": profile,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_detected": torch.cuda.is_available(),
        "devices": devices,
        "host_available_bytes": host_available(),
        "attention": "explicit-mask SDPA; actual kernel fallback requires measurement",
        "optimizer": "fp32_moments_bf16_stochastic_v1: tiled FP32 arithmetic/moments, private RNG, one stochastic BF16 write-back, no persistent master",
        "kernel_validation": "not executed by detection",
        "full_8b_training_validation": "not executed",
        "measured_throughput": None,
        "shared_memory_policy": "Spark host/device capacity is one physical pool; never add them",
    }


def parameter_inventory(model, architecture):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from ncm_lab.embedded.model import EmbeddedGenerator
    from ncm_lab.codec.model import ContinuousCodec
    from ncm_lab.config.architecture import Architecture

    configuration = (
        model["config"]
        if model.get("fixture")
        else dict(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            tie_word_embeddings=False,
        )
    )
    with torch.device("meta"):
        backbone = Qwen3ForCausalLM(Qwen3Config(**configuration))
        original = sum(p.numel() for p in backbone.parameters())
        generated = EmbeddedGenerator(backbone, Architecture(**architecture))
        codec = ContinuousCodec(backbone.config.hidden_size, Architecture(**architecture))
    return {
        "original": original,
        "generator": sum(p.numel() for p in generated.parameters()),
        "codec": sum(p.numel() for p in codec.parameters()),
        "head": backbone.lm_head.weight.numel(),
    }


def estimate_parameters(model, architecture):
    return parameter_inventory(model, architecture)["generator"]


def preflight(value, run_path, training=False):
    device = value["device"]
    profile = value.get("hardware_profile", "cpu" if device == "cpu" else "cuda")
    info = doctor(profile)
    inventory = parameter_inventory(value["model"], value["architecture"])
    params = inventory["generator"]
    item = 2 if value["dtype"] == "bfloat16" else 4
    weights = params * item
    trained = params if training else 0
    stage = value.get("stage")
    if not value["model"].get("fixture") and stage == "codec":
        trained = inventory["codec"]
        weights = (inventory["head"] + trained) * item
    if not value["model"].get("fixture") and stage == "targets" and value.get("inputs", {}).get("features"):
        weights = inventory["codec"] * item
    minimum = weights + trained * (item + 8)
    checkpoint = trained * (item + 8) if trained else weights
    result = {
        "parameter_inventory": inventory,
        "numerical_policy": value.get("training", {}).get(
            "optimizer_policy", "fp32_moments_bf16_stochastic_v1"
        ),
        "estimated_parameters": params,
        "weights_bytes": weights,
        "minimum_weights_gradients_moments_bytes": minimum,
        "fp32_master_bytes": 0,
        "one_recovery_bytes_estimate": checkpoint,
        "peak_checkpoint_disk_policy": "last two plus selected, new version before retirement; up to four recovery versions",
        "peak_disk_estimate": 4 * checkpoint + 64 * 1024 * 1024 + 1048576,
        "bounded_optimizer_scratch_estimate_bytes": value.get("training", {}).get(
            "optimizer_tile_elements", 1048576
        )
        * 160
        + 65536,
        "activation_bytes": "not included; requires actual kernel/optimizer/recovery probe",
        "hardware": info,
        "blockers": [],
    }
    if device == "cuda" and not torch.cuda.is_available():
        result["blockers"].append("CUDA unavailable in this PyTorch build")
    if not value["model"].get("fixture"):
        if device == "cpu" and training:
            result["blockers"].append("full 8B CPU training not authorized by this initial hardware profile")
        capacity = info["host_available_bytes"] or 0
        if device == "cuda":
            gpu = info["devices"][0]["bytes"] if info["devices"] else 0
            capacity = min(capacity, gpu) if profile == "spark" else gpu
        if minimum > capacity:
            result["blockers"].append(
                "weights, gradients and FP32 moments exceed available physical/device memory before activations"
            )
        if (training or stage == "codec") and shutil.disk_usage(run_path).free < result["peak_disk_estimate"]:
            result["blockers"].append("free disk cannot cover retained/new full recovery shards")
    return result
