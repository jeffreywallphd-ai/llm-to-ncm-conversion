"""Canvas-consistent local preparation, exact cache receipts and namespace policy.

No acquisition, historical scans, tensor resizing or training engine lives here.
"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from ncm_lab.config.architecture import Architecture
from ncm_lab.config.canvas import canvas_metadata
from ncm_lab.config.stages import config, bind_inputs
from ncm_lab.runtime.io import root, canonical_hash, implementation_hash, read, digest, resolve_reference

STAGES = {
    "data": ("data", "prepare"),
    "teacher": ("teacher", "extract"),
    "codec": ("codec", "train"),
    "targets": ("targets", "build"),
    "generator": ("generator", "train"),
}
TEMPLATES = {
    "data": "configs/data/real_pilot.yaml",
    "teacher": "configs/validation/teacher_features_pilot.yaml",
    "codec": "configs/validation/codec_pilot.yaml",
    "targets": "configs/validation/latent_targets_pilot.yaml",
}


@contextmanager
def exclusive_writer(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write("Owned experiment writer; inspect interrupted process before removing this lock.\n")
    except FileExistsError as exc:
        raise ValueError(f"experiment writer locked: {path}; inspect the owning/interrupted process") from exc
    try:
        yield
    finally:
        path.unlink()


def namespace(run_root, architecture, fingerprint):
    """All convenience pointers are qualified by size AND experiment contract."""
    label = canvas_metadata(architecture)["preset"]
    if label == "explicit":
        label = "explicit-" + str(architecture["canvas"])
    return Path(run_root) / "handoffs" / "canvas" / label / fingerprint


def pipeline_configs(generator, data_config=None, stage_overrides=None):
    generator = deepcopy(generator)
    values = {name: config(path) for name, path in TEMPLATES.items()}
    if data_config:
        values["data"] = config(data_config)
        if values["data"]["stage"] != "data":
            raise ValueError("--data-config must name a native data stage configuration")
    for name, value in values.items():
        value["architecture"] = deepcopy(generator["architecture"])
        value["model"] = deepcopy(generator["model"])
        # Data retains CPU/tokenizer-only execution and its own bounded source limits.
        if name != "data":
            for field in ("device", "dtype", "hardware_profile", "threads"):
                value[field] = generator[field]
        value["budget"]["enforce_wall_time_limits"] = False
    values["generator"] = generator
    for name, override in (stage_overrides or {}).items():
        values[name] = {**values[name], **deepcopy(override)}
    generator = values["generator"]
    run_root = Path(generator.get("run_root", root() / "runs")).resolve()
    # Generator step target/cadence are deliberately absent: 2 -> 100 is same experiment.
    contract = {
        name: {
            k: v
            for k, v in value.items()
            if k not in {"inputs", "output_ref", "run_root", "notes", "monitoring", "experiment_context"}
        }
        for name, value in values.items()
    }
    contract["generator"]["training"] = {
        k: v
        for k, v in generator["training"].items()
        if k not in {"steps", "validation_interval", "checkpoint_interval", "recovery_probe"}
    }
    for value in contract.values():
        value["budget"] = {k: v for k, v in value["budget"].items() if k != "phase_id"}
    fingerprint = canonical_hash(
        {"schema": "native_canvas_experiment_v1", "stages": contract, "implementation": implementation_hash()}
    )
    handoffs = namespace(run_root, generator["architecture"], fingerprint)
    context = {
        **canvas_metadata(generator["architecture"]),
        "fingerprint": fingerprint,
        "namespace": str(handoffs),
        "data_config": str(Path(data_config).resolve()) if data_config else TEMPLATES["data"],
    }
    paths = {name: str(handoffs / (name + ".json")) for name in STAGES}
    paths["generator"] = str(handoffs / f"generator_{generator['training']['steps']}_step.json")
    for name, value in values.items():
        value["run_root"] = str(run_root)
        value["output_ref"] = paths[name]
        value["experiment_context"] = context
        value["budget"] = {**value["budget"], "phase_id": "canvas-" + fingerprint + "-" + name}
        if name == "data":
            value["inputs"] = {}
        elif name == "teacher":
            value["inputs"] = {"data": paths["data"]}
        elif name == "codec":
            value["inputs"] = {"features": paths["teacher"]}
        elif name == "targets":
            value["inputs"] = {"data": paths["data"], "features": paths["teacher"], "codec": paths["codec"]}
        else:
            value["inputs"] = {"targets": paths["targets"]}
    return values


def bind_canvas_stage(value):
    """Direct stage override uses the same default experiment contract as named pilots."""
    if value["stage"] not in STAGES:
        return value
    generator = (
        deepcopy(value)
        if value["stage"] == "generator"
        else config("configs/validation/generator_100_step_pilot.yaml", _bind_canvas=False)
    )
    for field in ("architecture", "model", "run_root"):
        if field in value:
            generator[field] = deepcopy(value[field])
    values = pipeline_configs(generator, stage_overrides={value["stage"]: value})
    return values[value["stage"]]


def verify_stage(value):
    """A cache is a successful sealed run with this exact config and immutable inputs."""
    pointer = read(value["output_ref"])
    path = resolve_reference(value["output_ref"])
    run_ref = pointer.get("run")
    if not run_ref or digest(run_ref["manifest"]) != run_ref["sha256"]:
        raise ValueError("missing/changed sealed canvas stage receipt")
    run = Path(run_ref["manifest"]).parent
    from ncm_lab.runtime.provenance import validate_run

    sealed = validate_run(run)
    bound, missing = bind_inputs(value)
    if missing or read(run / "resolved_config.json") != {**value, "resolved_inputs": bound}:
        raise ValueError("prepared artifact configuration/source/input binding differs")
    if sealed["status"] != "completed" or sealed["implementation_sha256"] != implementation_hash():
        raise ValueError("prepared stage incomplete or implementation changed")
    summary = read(run / "summary.json")
    if Path(summary["artifact"]).resolve() != path.resolve():
        raise ValueError("stage receipt artifact differs")
    artifact = read(path)
    expected = {
        "data": "native_data_v1",
        "teacher": "native_features_v1",
        "codec": "native_codec_v1",
        "targets": "native_targets_v1",
    }[value["stage"]]
    if artifact.get("schema") != expected:
        raise ValueError("wrong prepared artifact kind")
    if value["stage"] == "codec":
        from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE

        if artifact["spec"]["architecture"] != value["architecture"]:
            raise ValueError("codec architecture mismatch")
        weights = (path.parent / artifact["weights_manifest"]).resolve()
        if (
            not weights.is_relative_to(path.parent.resolve())
            or digest(weights) != artifact["weights_manifest_sha256"]
        ):
            raise ValueError("codec weights changed")
        CheckpointReader(weights, expected_kind=INFERENCE, identity=artifact["identity"])
    else:
        from ncm_lab.data.artifacts import Dataset

        dataset = Dataset(path, expected)
        identity = dataset.manifest["identity"]
        if (
            identity.get("canvas", identity.get("architecture", {}).get("canvas"))
            != value["architecture"]["canvas"]
        ):
            raise ValueError("prepared canvas mismatch")
        if value["stage"] == "targets" and identity["architecture"] != value["architecture"]:
            raise ValueError("prepared target schedule mismatch")
        for _, row in dataset.iter_metadata():
            if "tensor" in row:
                target = (dataset.path / row["tensor"]["file"]).resolve()
                if (
                    not target.is_relative_to(dataset.path.resolve())
                    or digest(target) != row["tensor"]["sha256"]
                ):
                    raise ValueError("prepared tensor shard changed")
    return {"manifest": str(path), "sha256": digest(path)}


def prerequisite_plan(values):
    decisions = {}
    for name in TEMPLATES:
        try:
            decisions[name] = {"reuse": True, "artifact": verify_stage(values[name])}
        except (OSError, ValueError, KeyError) as exc:
            decisions[name] = {"reuse": False, "reason": str(exc)}
    return decisions


def verify_bound_inputs(value):
    """Direct stage commands require completed producers in this exact experiment."""
    for key, path in value["inputs"].items():
        artifact = resolve_reference(path)
        # Producers always store native artifacts under RUN/artifacts/NAME/manifest.json.
        run = artifact.parent.parent.parent
        producer = read(run / "resolved_config.json")
        producer.pop("resolved_inputs", None)
        if (
            producer.get("experiment_context", {}).get("fingerprint")
            != value["experiment_context"]["fingerprint"]
            or producer["architecture"] != value["architecture"]
        ):
            raise ValueError(f"{key}: wrong canvas/schedule/source experiment binding")
        from ncm_lab.runtime.provenance import validate_run

        sealed = validate_run(run)
        if sealed["status"] != "completed" or sealed["implementation_sha256"] != implementation_hash():
            raise ValueError(f"{key}: prerequisite was not completed by this implementation")
        resolved = read(run / "resolved_config.json")
        for binding in resolved["resolved_inputs"].values():
            if digest(binding["manifest"]) != binding["sha256"]:
                raise ValueError(f"{key}: immutable producer input changed")
        if Path(read(run / "summary.json")["artifact"]).resolve() != artifact.resolve():
            raise ValueError(f"{key}: producer artifact differs")
        # Check content without consulting a convenience pointer that may have advanced.
        info = read(artifact)
        if info["schema"] == "native_codec_v1":
            from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE

            weights = (artifact.parent / info["weights_manifest"]).resolve()
            if (
                not weights.is_relative_to(artifact.parent.resolve())
                or digest(weights) != info["weights_manifest_sha256"]
            ):
                raise ValueError("codec weights changed")
            CheckpointReader(weights, expected_kind=INFERENCE, identity=info["identity"])
        else:
            from ncm_lab.data.artifacts import Dataset

            dataset = Dataset(artifact)
            for _, row in dataset.iter_metadata():
                if "tensor" in row:
                    shard = (dataset.path / row["tensor"]["file"]).resolve()
                    if (
                        not shard.is_relative_to(dataset.path.resolve())
                        or digest(shard) != row["tensor"]["sha256"]
                    ):
                        raise ValueError("prerequisite tensor changed")


def resume_config(checkpoint, explicit=None, canvas_size=None, target_steps=None):
    """Recover immutable launch config/inputs. Never consult mutable convenience pointers."""
    from ncm_lab.training.checkpoints import CheckpointReader, native_identity

    reader = CheckpointReader(checkpoint)
    metadata = reader.materialize(reader.field("metadata"))
    if (
        target_steps is not None
        and reader.materialize(reader.field("training"))["optimizer_steps"] > target_steps
    ):
        raise ValueError(
            "resume target is below completed optimizer steps; cannot move a checkpoint backwards"
        )
    if (
        native_identity(metadata) != reader.manifest["identity"]
        or reader.materialize(reader.field("identity")) != reader.manifest["identity"]
    ):
        raise ValueError("recovery metadata identity mismatch")
    # A native checkpoint is under RUN/checkpoints/version-ID/manifest.json.
    run = reader.path.parent.parent.parent
    resolved = read(run / "resolved_config.json")
    launch = read(run / "launch.json")
    if digest(run / "resolved_config.json") != launch["resolved_config_sha256"]:
        raise ValueError("recovery launch configuration changed")
    if metadata["architecture"] != resolved["architecture"]:
        raise ValueError("recovery architecture differs from launch")
    if canvas_size is not None:
        from ncm_lab.config.canvas import resolve_canvas

        requested = (
            resolve_canvas(canvas_size).architecture(Architecture(**resolved["architecture"])).to_dict()
        )
        if requested != metadata["architecture"]:
            raise ValueError("resume canvas mismatch: changing canvas requires --fresh")
    if explicit and explicit["architecture"] != metadata["architecture"]:
        raise ValueError("resume configuration architecture mismatch; use --fresh for another canvas")
    if explicit:
        for field in ("model", "device", "dtype"):
            if explicit[field] != resolved[field]:
                raise ValueError(
                    "explicit resume configuration " + field + " mismatch before model allocation"
                )
        for key, item in explicit["training"].items():
            if (
                key not in {"steps", "validation_interval", "checkpoint_interval", "recovery_probe"}
                and item != resolved["training"][key]
            ):
                raise ValueError("explicit resume training configuration mismatch: " + key)
        if explicit["budget"]["phase_id"] != metadata["phase_id"]:
            raise ValueError("explicit resume phase mismatch; prior accounting must continue")
    if metadata["implementation_sha256"] != implementation_hash():
        raise ValueError(
            "unreviewed implementation transition for canvas pilots; use --fresh (old exports remain usable)"
        )
    value = deepcopy(resolved)
    bound = value.pop("resolved_inputs")
    value["inputs"] = {key: item["manifest"] for key, item in bound.items()}
    for item in bound.values():
        if digest(item["manifest"]) != item["sha256"]:
            raise ValueError("checkpoint-bound preparation artifact changed")
    if bound["targets"]["sha256"] != metadata["dataset_sha256"]:
        raise ValueError("checkpoint-bound targets differ from recovery metadata")
    if explicit:
        from ncm_lab.config.stages import bind_inputs

        requested, missing = bind_inputs(explicit)
        if missing or requested != bound:
            raise ValueError("explicit resume configuration inputs differ from checkpoint-bound inputs")
        # The training engine still checks the complete numerical/runtime/phase contract.
        value.update({k: v for k, v in explicit.items() if k not in {"inputs", "architecture"}})
    return value


def verify_generator_inputs(value):
    """Explicit legacy/fixture configs still need compatible offline targets before a child."""
    from ncm_lab.teacher.model import model_identity

    bound, missing = bind_inputs(value)
    if missing:
        raise ValueError("; ".join(missing))
    source = read(bound["targets"]["manifest"])
    if (
        source.get("schema") != "native_targets_v1"
        or source["identity"]["architecture"] != value["architecture"]
    ):
        raise ValueError("targets canvas/schedule mismatch")
    if source["identity"]["model"] != model_identity(value["model"]):
        raise ValueError("targets source/model mismatch")
