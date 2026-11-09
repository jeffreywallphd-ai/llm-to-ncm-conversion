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


def admission(run, settings, *, training):
    """Meta inventory and shared-pool admission; no measured feasibility claim."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
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
        params = sum(p.numel() for p in backbone.parameters()) + backbone.config.hidden_size
    item = 2 if settings["dtype"] == "bfloat16" else 4
    weights = params * item
    recovery = params * (item + 8) if training else weights
    minimum = (
        weights + (params * (item + 8) if training else 0) + settings["resources"]["activation_reserve_bytes"]
    )
    # Two retained recoveries, one in-progress version, two referenced selections
    # and one replacement selection before the next recovery is atomically committed.
    disk = 3 * recovery + 3 * weights if training else weights
    info = doctor(settings["hardware_profile"])
    assessment = memory_capacity(info, settings["device"], settings["hardware_profile"])
    blockers = []
    if settings["device"] == "cuda" and not info["cuda_available"]:
        blockers.append("CUDA unavailable")
    if assessment["effective_capacity_bytes"] is None or minimum > assessment["effective_capacity_bytes"]:
        blockers.append(
            "weights/gradients/FP32 moments plus declared activation allowance exceed known capacity"
        )
    if disk > run.meter.budget.max_storage_bytes:
        blockers.append("checkpoint staging exceeds retained run storage allowance")
    if shutil.disk_usage(run.path).free < disk + settings["resources"]["reserve_free_bytes"]:
        blockers.append("insufficient checkpoint staging/free-disk reserve")
    value = dict(
        schema="block8_admission_v1",
        parameters=params,
        weights_bytes=weights,
        minimum_declared_memory_bytes=minimum,
        checkpoint_staging_bytes=disk,
        memory_capacity=assessment,
        blockers=blockers,
        scope="meta arithmetic and declared reserves; actual activations/kernels/full-8B recovery unmeasured",
    )
    write_json(run.path / "admission.json", value, exclusive=True)
    if blockers:
        raise ValueError("block8 admission blocked: " + "; ".join(blockers))
    return value


def plan(settings):
    return dict(
        status="planned",
        schema="block8_plan_v1",
        configuration_identity=settings["identity"],
        architecture=settings["architecture"],
        data=settings["data"],
        training=settings["training"],
        decoding=settings["decoding"],
        resources=settings["resources"],
        budget=settings["budget"],
        source=settings["source"],
        model=settings["model"],
        stages=["source (unless explicit sealed data supplied)", "prepare", "train", "evaluate"],
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
        schema="block8_experiment_v1",
        settings=reference(Path(folder) / "resolved_config.json"),
        configuration_identity=settings["identity"],
        status="created",
        stages={},
        attempts={},
        recovery=None,
    )


def verify_settings(state):
    from ncm_lab.runtime.io import implementation_hash

    settings = read(checked(state["settings"]))
    identity = settings.pop("identity")
    if canonical_hash(settings) != identity or identity != state["configuration_identity"]:
        raise ValueError("block8 resolved configuration identity changed")
    settings["identity"] = identity
    if implementation_hash() != settings["implementation_sha256"]:
        raise ValueError("block8 implementation changed; preserve this experiment and start a fresh one")
    return settings
