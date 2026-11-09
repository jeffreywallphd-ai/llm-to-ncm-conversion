"""Finite attempts, immutable settings, persistent phase and write ledgers."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import uuid
from ncm_lab.runtime.budget import Budget, Meter, BudgetStop
from ncm_lab.runtime.io import canonical_hash, digest, read, reference, write_json
from ncm_lab.stage1.io import WriteMeter


def checked(ref):
    path = Path(ref["manifest"])
    if ref.get("schema") != "native_reference_v1" or digest(path) != ref["sha256"]:
        raise ValueError("changed block8 dependency receipt")
    return path


class Attempt:
    def __init__(self, experiment, action, settings):
        self.experiment, self.action, self.settings = Path(experiment), action, settings
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        self.path = self.experiment / "attempts" / action / self.id
        self.path.mkdir(parents=True)
        self.value = {**settings, "stage": "block8_" + action}
        write_json(self.path / "resolved_config.json", self.value, exclusive=True)
        budget = {**settings["budget"], "phase_id": "block8-" + settings["identity"][:24] + "-" + action}
        self.meter = Meter(Budget(**budget), self.path, self.experiment / "ledgers")
        disk_check = self.meter.check_disk

        def check_disk(additional=0):
            disk_check(additional)
            if shutil.disk_usage(self.path).free < settings["resources"]["reserve_free_bytes"] + additional:
                raise BudgetStop("block8 free-disk reserve exhausted")

        self.meter.check_disk = check_disk
        self.meter.write_meter = WriteMeter(
            dict(
                ledger=str(self.experiment / "ledgers" / (action + "-writes.json")),
                max_run_bytes=settings["resources"]["max_write_bytes"],
                max_phase_bytes=settings["resources"]["max_phase_write_bytes"],
            ),
            self.id,
        )
        self.started = time.perf_counter()

    def publish(self, path):
        pass  # The controller seals handoffs only after a completed attempt.

    def close(self, status, result):
        self.meter.close()
        value = dict(
            schema="block8_attempt_v1",
            action=self.action,
            status=status,
            result=result,
            elapsed_seconds=time.perf_counter() - self.started,
            charged_work=self.meter.used,
            wall_policy=self.meter.wall_report(),
            implementation_sha256=self.settings["implementation_sha256"],
            resolved_config_sha256=digest(self.path / "resolved_config.json"),
            origin="software_fixture" if self.settings["model"].get("fixture") else "real_model",
            research_claims=[],
        )
        write_json(self.path / "receipt.json", value, exclusive=True)
        return reference(self.path / "receipt.json")


def charge_write(run, size):
    # Conservative logical SQLite/page/journal charges survive failures and pruning.
    path = run.path / "logical-write-charge"
    writer = run.meter.write_meter
    with writer.transaction() as state:
        own = state["attempts"].get(run.id, 0)
        active = sum(v["bytes"] for v in state["reservations"].values())
        if (
            own + size > writer.settings["max_run_bytes"]
            or state["written_bytes"] + active + size > writer.settings["max_phase_bytes"]
        ):
            raise BudgetStop("block8 cumulative logical writes exhausted")
        state["attempts"][run.id] = own + size
        state["written_bytes"] += size
        state["logical_scope"] = (
            "conservative source/index logical bytes plus page allowance; not physical device wear"
        )
        state["last_logical_path"] = str(path)


def admission(run, settings, *, training, kind="generator", data=None):
    """Inventory the actual hierarchy/codec/frozen models; never sum Spark pools."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from ncm_lab.embedded.model import EmbeddedGenerator
    from ncm_lab.codec.model import ContinuousCodec
    from ncm_lab.block8.config import Architecture
    from ncm_lab.runtime.doctor import doctor, memory_capacity

    checked(settings["model_asset_policy"])
    config = (
        settings["model"]["config"]
        if settings["model"].get("fixture")
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
        backbone = Qwen3ForCausalLM(Qwen3Config(**config))
        original = sum(p.numel() for p in backbone.parameters())
        head = backbone.lm_head.weight.numel()
        codec = ContinuousCodec(backbone.config.hidden_size, Architecture(**settings["architecture"]))
        codec_params = sum(p.numel() for p in codec.parameters())
        generator = EmbeddedGenerator(backbone, Architecture(**settings["architecture"]))
        generator_params = sum(p.numel() for p in generator.parameters())
    item = 2 if settings["dtype"] == "bfloat16" else 4
    teacher_item = 2 if settings["features"]["teacher_dtype"] == "bfloat16" else 4
    if kind == "codec":
        params, weights = codec_params, codec_params * 4 + head * 4
        trained = codec_params if training else 0
        parameter_item = 4
        checkpoint_weights = codec_params * 4
    elif kind in {"teacher", "targets"}:
        params, weights = original, original * teacher_item + (codec_params * 4 if kind == "targets" else 0)
        trained = 0
        parameter_item = teacher_item
        checkpoint_weights = weights
    else:
        params, weights = generator_params, generator_params * item
        trained = generator_params if training else 0
        parameter_item = item
        checkpoint_weights = weights
    minimum = weights + trained * (parameter_item + 8) + settings["resources"]["activation_reserve_bytes"]
    recovery = trained * (parameter_item + 8) if trained else checkpoint_weights
    disk = 3 * recovery + 3 * checkpoint_weights if training else checkpoint_weights
    if kind == "teacher":
        positions = (
            settings["features"]["target_train_positions"]
            + settings["data"]["max_answer_tokens"]
            - 1
            + settings["features"]["max_dev_parents"] * settings["data"]["max_answer_tokens"]
        )
        disk = positions * backbone.config.hidden_size * teacher_item + settings["data"]["max_parents"] * 8192
    if kind == "targets" and data is not None:
        positions = sum(data.manifest["unique_answer_positions"].get(k, 0) for k in ("train", "dev"))
        disk = (
            positions * settings["architecture"]["latent_width"] * 4 + settings["data"]["max_parents"] * 8192
        )
    info = doctor(settings["hardware_profile"])
    assessment = memory_capacity(info, settings["device"], settings["hardware_profile"])
    blockers = []
    if settings["device"] == "cuda" and not info["cuda_detected"]:
        blockers.append("CUDA unavailable")
    if assessment["effective_capacity_bytes"] is None or minimum > assessment["effective_capacity_bytes"]:
        blockers.append(
            "declared weights/gradients/FP32 moments and activation reserve exceed known capacity"
        )
    if disk > run.meter.budget.max_storage_bytes:
        blockers.append("stage cache/checkpoint staging exceeds retained storage allowance")
    if shutil.disk_usage(run.path).free < disk + settings["resources"]["reserve_free_bytes"]:
        blockers.append("insufficient cache/checkpoint staging and free-disk reserve")
    value = dict(
        schema="hierarchical_block8_admission_v1",
        kind=kind,
        parameters=params,
        parameter_inventory=dict(
            original=original, generator=generator_params, codec=codec_params, frozen_head=head
        ),
        weights_bytes=weights,
        minimum_declared_memory_bytes=minimum,
        checkpoint_or_cache_staging_bytes=disk,
        memory_capacity=assessment,
        blockers=blockers,
        scope="meta inventory and declared reserves; actual full-8B activation/kernel/recovery peaks unmeasured",
    )
    write_json(run.path / "admission.json", value, exclusive=True)
    if blockers:
        raise ValueError("hierarchical block8 admission blocked: " + "; ".join(blockers))
    return value


def plan(settings):
    return dict(
        status="planned",
        schema="hierarchical_block8_plan_v1",
        configuration_identity=settings["identity"],
        architecture=settings["architecture"],
        data=settings["data"],
        training=settings["training"],
        decoding=settings["decoding"],
        resources=settings["resources"],
        budget=settings["budget"],
        source=settings["source"],
        model=settings["model"],
        stages=[
            "source (unless explicit sealed data supplied)",
            "prepare",
            "teacher",
            "codec",
            "targets",
            "generator",
            "evaluate",
        ],
        codec=settings["codec"],
        features=settings["features"],
        block_contract=settings["block_contract"],
        research_work_started=False,
        downloads_started=False,
        model_allocated=False,
        target_policy="released references; teacher-generated targets require an explicit sealed input artifact",
        limitations=[
            "prototype training horizon is not a validated adaptation recipe",
            "8B Spark feasibility, quality and speed require separate measurements",
        ],
    )


def seal_settings(folder, settings):
    write_json(Path(folder) / "resolved_config.json", settings, exclusive=True)
    return dict(
        schema="hierarchical_block8_experiment_v1",
        settings=reference(Path(folder) / "resolved_config.json"),
        configuration_identity=settings["identity"],
        status="created",
        stages={},
        attempts={},
        recoveries={"codec": None, "generator": None},
    )


def verify_settings(state):
    from ncm_lab.runtime.io import implementation_hash

    if state.get("schema") != "hierarchical_block8_experiment_v1":
        raise ValueError("custom baseline experiments cannot resume as hierarchical block8")
    settings = read(checked(state["settings"]))
    identity = settings.pop("identity")
    if canonical_hash(settings) != identity or identity != state["configuration_identity"]:
        raise ValueError("block8 resolved configuration identity changed")
    settings["identity"] = identity
    if implementation_hash() != settings["implementation_sha256"]:
        raise ValueError("block8 implementation changed; preserve this experiment and start a fresh one")
    return settings
