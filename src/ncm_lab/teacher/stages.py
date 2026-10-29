"""Offline teacher states and frozen-codec targets; generator training never calls these."""

from pathlib import Path
import torch
from ncm_lab.config.architecture import Architecture
from ncm_lab.data.artifacts import Dataset, ArtifactWriter
from ncm_lab.data.records import ALIGNMENT
from ncm_lab.teacher.model import load_original, prediction_states, model_identity
from ncm_lab.runtime.io import read, digest, canonical_hash


def extract(run, value, bound):
    data = Dataset(bound["data"]["manifest"], "native_data_v1")
    if data.manifest["identity"]["model"] != model_identity(value["model"]):
        raise ValueError("teacher source/model mismatch")
    if data.manifest["identity"]["canvas"] != value["architecture"]["canvas"]:
        raise ValueError("teacher source canvas mismatch")
    quota = value.get("data", {}).get("max_unique_answer_positions", 0)
    if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
        raise ValueError("nonnegative integer teacher subset quota required")
    indices = list(range(len(data)))
    if quota:
        train = data.indices("train")
        selected = []
        positions = 0
        for index in train:
            selected.append(index)
            positions += len(data.metadata(index)["example"]["answer_ids"])
            if positions >= quota:
                break
        indices = sorted(selected + data.indices("dev")[: value["data"].get("max_dev_examples", 256)])
    teacher = load_original(value["model"], value["device"], getattr(torch, value["dtype"]), frozen=True)
    identity = {
        "model": model_identity(value["model"]),
        "data_sha256": data.sha256,
        "state_alignment": ALIGNMENT,
        "canvas": value["architecture"]["canvas"],
        "dtype": value["dtype"],
        "template_sha256": data.manifest["identity"]["template_sha256"],
        "selected_indices_sha256": canonical_hash(indices),
    }
    writer = ArtifactWriter(run.path / "artifacts/features", "native_features_v1", identity, run.meter)
    for i in indices:
        example, _ = data[i]
        run.meter.charge(
            source=len(example.prompt_ids) + len(example.answer_ids) - 1,
            answer=len(example.answer_ids),
            steps=1,
        )
        states = prediction_states(teacher, example.prompt_ids, example.answer_ids)
        writer.append(example, states)
    manifest = writer.finish(
        {
            "hidden_size": teacher.config.hidden_size,
            "extraction": "backbone final normalized output only; no vocabulary logits or all-layer outputs",
            "requested_unique_training_answer_positions": quota,
            "whole_example_selection": True,
        }
    )
    run.publish(manifest)
    return {
        "artifact": str(manifest),
        "stats": writer.stats,
        "teacher_frozen": not any(p.requires_grad for p in teacher.parameters()),
        "hidden_size": teacher.config.hidden_size,
    }


def load_codec(path, device="cpu", dtype=torch.float32):
    from ncm_lab.codec.model import ContinuousCodec, SCHEMA
    from ncm_lab.training.checkpoints import CheckpointReader, INFERENCE

    path = Path(path)
    manifest = read(path)
    if manifest.get("schema") != "native_codec_v1" or manifest["spec"]["schema"] != SCHEMA:
        raise ValueError("native frozen codec schema required")
    weights = (path.parent / manifest["weights_manifest"]).resolve()
    if (
        not weights.is_relative_to(path.parent.resolve())
        or digest(weights) != manifest["weights_manifest_sha256"]
    ):
        raise ValueError("selected codec weights changed")
    codec = ContinuousCodec(
        manifest["spec"]["hidden_size"], Architecture(**manifest["spec"]["architecture"])
    ).to(device=device, dtype=dtype)
    reader = CheckpointReader(weights, expected_kind=INFERENCE, identity=manifest["identity"])
    reader.load_model(codec)
    codec.requires_grad_(False).eval()
    return codec, manifest


def targets(run, value, bound):
    data = Dataset(bound["data"]["manifest"], "native_data_v1")
    if data.manifest["identity"]["canvas"] != value["architecture"]["canvas"]:
        raise ValueError("target source canvas mismatch")
    codec, selected = load_codec(bound["codec"]["manifest"], value["device"], getattr(torch, value["dtype"]))
    if (
        data.manifest["identity"]["model"] != selected["source"]["model"]
        or selected["spec"]["architecture"] != Architecture(**value["architecture"]).to_dict()
    ):
        raise ValueError("native target source/codec/geometry mismatch")
    identity = {
        "model": model_identity(value["model"]),
        "data_sha256": data.sha256,
        "state_alignment": ALIGNMENT,
        "codec_sha256": bound["codec"]["sha256"],
        "normalization": selected["normalization"],
        "scales": selected["scales"],
        "architecture": value["architecture"],
        "dtype": value["dtype"],
        "template_sha256": data.manifest["identity"]["template_sha256"],
    }
    writer = ArtifactWriter(run.path / "artifacts/targets", "native_targets_v1", identity, run.meter)
    features = Dataset(bound["features"]["manifest"], "native_features_v1") if bound.get("features") else None
    if features and features.manifest["identity"]["data_sha256"] != data.sha256:
        raise ValueError("cached teacher source differs; no silent fallback")
    if features and (
        features.manifest["identity"]["model"] != model_identity(value["model"])
        or features.manifest["identity"]["state_alignment"] != ALIGNMENT
        or features.manifest["identity"]["canvas"] != value["architecture"]["canvas"]
    ):
        raise ValueError("cached teacher semantics or canvas changed")
    if features and len(features) != len(data):
        raise ValueError(
            "teacher subset cache does not cover target pool; omit features explicitly for streamed preprocessing"
        )
    teacher = (
        None
        if features
        else load_original(value["model"], value["device"], getattr(torch, value["dtype"]), frozen=True)
    )
    with torch.no_grad():
        for i in range(len(data)):
            example, _ = data[i]
            if features:
                source, states = features[i]
                if source.identity != example.identity:
                    raise ValueError("teacher cache ordering/content mismatch")
                run.meter.charge(answer=len(example.answer_ids), steps=1)
                states = states.to(value["device"], getattr(torch, value["dtype"]))
            else:
                run.meter.charge(
                    source=len(example.prompt_ids) + len(example.answer_ids) - 1,
                    answer=len(example.answer_ids),
                    steps=1,
                )
                states = prediction_states(teacher, example.prompt_ids, example.answer_ids)
            writer.append(example, codec.fine_codes(states))
    manifest = writer.finish(
        {
            "coordinate_system": "raw frozen encoder outputs; scales applied only in auxiliary loss",
            "full_teacher_states_stored": False,
            "cached_teacher_reused": features is not None,
        }
    )
    run.publish(manifest)
    return {
        "artifact": str(manifest),
        "stats": writer.stats,
        "codec_sha256": bound["codec"]["sha256"],
        "inference_dependency": False,
    }
