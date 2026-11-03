"""Detection, kernel execution, full training, and throughput are separate evidence."""

import platform
import shutil
import sys
import torch
from ncm_lab.runtime.host_memory import memory_snapshot, host_available as host_available


def doctor(profile="cpu", *, memory=None):
    import transformers

    snapshot = memory_snapshot() if memory is None else memory
    devices, warnings, selected = [], [], None
    try:
        detected = bool(torch.cuda.is_available())
        if detected:
            count = torch.cuda.device_count()
            if type(count) is not int or not 0 <= count <= 1024:
                raise ValueError("invalid CUDA device count")
            for index in range(count):
                try:
                    properties = torch.cuda.get_device_properties(index)
                    size = properties.total_memory
                    if type(size) is not int or size < 0:
                        raise ValueError("invalid device total memory")
                    devices.append({"index": index, "name": properties.name, "bytes": size})
                except (RuntimeError, OSError, ValueError) as exc:
                    devices.append({"index": index, "name": None, "bytes": None})
                    warnings.append("CUDA device properties unavailable: " + str(exc))
            selected = torch.cuda.current_device()
            if type(selected) is not int or not 0 <= selected < count:
                selected = None
                warnings.append("invalid current CUDA device selection")
    except (RuntimeError, OSError, ValueError, AssertionError) as exc:
        detected = False if not devices else True
        warnings.append("CUDA detection/selection unavailable: " + str(exc))
    return {
        "profile": profile,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_detected": detected,
        "devices": devices,
        "selected_device_index": selected,
        "device_warnings": warnings,
        **snapshot,
        "attention": "explicit-mask SDPA; actual kernel fallback requires measurement",
        "optimizer": "fp32_moments_bf16_stochastic_v1: tiled FP32 arithmetic/moments, private RNG, one stochastic BF16 write-back, no persistent master",
        "kernel_validation": "not executed by detection",
        "full_8b_training_validation": "not executed",
        "measured_throughput": None,
        "shared_memory_policy": "Spark host/device capacity is one physical pool; never add them",
    }


def memory_capacity(info, device, profile):
    """Pure capacity decision over one doctor snapshot; no extra live measurements."""
    host_needed = device == "cpu" or profile == "spark"
    selected = info.get("selected_device_index")
    gpu = (
        next((d.get("bytes") for d in info["devices"] if d.get("index") == selected), None)
        if selected is not None
        else None
    )
    operands, sources, missing = {}, {}, []
    constraint = info["process_memory_constraint"]
    if host_needed:
        operands["host_available_bytes"] = info["host_available_bytes"]
        sources["host_available_bytes"] = info["host_available_source"]
        if constraint["remaining_headroom_bytes"] is not None:
            operands["cgroup_remaining_headroom_bytes"] = constraint["remaining_headroom_bytes"]
            sources["cgroup_remaining_headroom_bytes"] = (
                "cgroup_v2.remaining_headroom:" + constraint["limiting_group"]
            )
        if constraint["blocking"]:
            missing.append("indeterminate process memory constraint (limit/current could not be assessed)")
    if device == "cuda":
        operands["device_total_bytes"] = gpu
        sources["device_total_bytes"] = "cuda.device_properties[" + str(selected) + "].total_memory"
    missing.extend(key for key, value in operands.items() if value is None)
    capacity = min(operands.values()) if operands and not missing else None
    limiting = (
        next((sources[key] for key, value in operands.items() if value == capacity), None)
        if capacity is not None
        else None
    )
    return {
        "device_total_bytes": gpu,
        "effective_capacity_bytes": capacity,
        "capacity_limiting_source": limiting,
        "capacity_operands": operands,
        "capacity_operand_sources": sources,
        "memory_assessment_limitations": missing + constraint["warnings"] + info["device_warnings"],
        "memory_reserve_bytes": 0,
        "assessment_scope": "baseline state admission only; activations, allocator and recovery runtime peaks unmeasured",
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
    from ncm_lab.runtime.budget import Budget

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
    assessment = memory_capacity(info, device, profile)
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
        "wall_policy": Budget(**value["budget"]).wall_policy(),
        **assessment,
        "remaining_after_baseline_bytes": assessment["effective_capacity_bytes"] - minimum
        if assessment["effective_capacity_bytes"] is not None
        else None,
        "capacity_guard_applied": not value["model"].get("fixture"),
        "blocker_details": [],
        "blockers": [],
    }
    if device == "cuda" and not info["cuda_detected"]:
        result["blockers"].append("CUDA unavailable in this PyTorch build")
    if not value["model"].get("fixture"):
        if device == "cpu" and training:
            result["blockers"].append("full 8B CPU training not authorized by this initial hardware profile")
        capacity = assessment["effective_capacity_bytes"]
        if capacity is None:
            reason = "memory capacity measurement/constraint assessment unavailable: " + ", ".join(
                assessment["memory_assessment_limitations"]
            )
            result["blockers"].append(reason)
            result["blocker_details"].append(
                {
                    "category": "memory_assessment_indeterminate",
                    "required_bytes": minimum,
                    "available_bytes": None,
                    "limiting_source": None,
                    "warnings": info["memory_warnings"] + info["device_warnings"],
                }
            )
        elif minimum > capacity:
            result["blockers"].append(
                "weights, gradients and FP32 moments exceed available physical/device memory before activations"
                + f"; required_bytes={minimum}, available_bytes={capacity}, limiting_source={assessment['capacity_limiting_source']}"
            )
            result["blocker_details"].append(
                {
                    "category": "insufficient_memory",
                    "required_bytes": minimum,
                    "available_bytes": capacity,
                    "limiting_source": assessment["capacity_limiting_source"],
                    "warnings": info["memory_warnings"] + info["device_warnings"],
                }
            )
        if (training or stage == "codec") and shutil.disk_usage(run_path).free < result["peak_disk_estimate"]:
            result["blockers"].append("free disk cannot cover retained/new full recovery shards")
    return result
