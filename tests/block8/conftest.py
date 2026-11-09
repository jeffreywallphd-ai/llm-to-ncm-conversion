"""Finite local block8 fixtures, with real Qwen modules and no downloaded assets."""

from dataclasses import asdict
import pytest
from ncm_lab.block8.config import resolve, DataPolicy, Training, Decoding, Resources
from ncm_lab.data.artifacts import ArtifactWriter, LOCKED
from ncm_lab.data.records import Example, LOCKED_SCHEMA
from ncm_lab.inference.export import fixture_tokenizer
from ncm_lab.runtime.io import canonical_hash, read, root, write_json
from ncm_lab.teacher.model import model_identity
from ncm_lab.validation.fixtures import model_spec


@pytest.fixture
def block8_fixture(tmp_path):
    cfg = read(root() / "configs/block8/experiment.yaml")
    spec = model_spec()
    spec["config"]["num_hidden_layers"] = 2
    cfg.update(model=spec, device="cpu", dtype="float32", hardware_profile="cpu")
    cfg["data"] = asdict(
        DataPolicy(
            max_prompt_tokens=64,
            max_answer_tokens=32,
            max_context_tokens=96,
            max_parents=64,
            max_views=256,
            max_index_bytes=2000000,
            minimum_train=2,
            minimum_dev=2,
            minimum_test=2,
        )
    )
    cfg["training"] = asdict(
        Training(
            steps=4,
            accumulation=1,
            learning_rate=0.001,
            mask_learning_rate=0.002,
            validation_interval=1,
            checkpoint_interval=1,
            max_validation_views=2,
            shard_bytes=8192,
            activation_checkpointing=False,
            curriculum_end_steps=(1, 2, 3, 4),
        )
    )
    cfg["decoding"] = asdict(
        Decoding(refinement_passes=2, max_new_tokens=24, max_blocks=3, max_evaluation_examples=2)
    )
    cfg["resources"] = asdict(
        Resources(
            reserve_free_bytes=1,
            activation_reserve_bytes=0,
            max_write_bytes=1000000000,
            max_phase_write_bytes=3000000000,
        )
    )
    cfg["budget"].update(
        max_wall_seconds=120,
        max_phase_wall_seconds=360,
        max_storage_bytes=200000000,
        max_phase_storage_bytes=600000000,
    )
    path = tmp_path / "block8 config.json"
    write_json(path, cfg)
    template = canonical_hash(fixture_tokenizer().chat_template)
    identity = dict(model=model_identity(spec), template_sha256=template, canvas=32, thinking=False)
    train = ArtifactWriter(tmp_path / "native parents", "native_data_v1", identity)
    locked = ArtifactWriter(tmp_path / "locked parents", LOCKED, identity)
    parents = []
    for split, marker in (("train", 4), ("dev", 5), ("test", 6)):
        for i, length in enumerate((8, 9, 16, 17)):
            e = Example(
                original_id=f"{split}-{i}",
                group_id=f"tree-{split}-{i}",
                split=split,
                source_type="software_fixture",
                prompt_ids=[1, marker, 7 + i],
                answer_ids=[8 + i] * (length - 1) + [2],
                eos_genuine=True,
                censored=False,
                provenance=dict(
                    template_sha256=template,
                    eos_ids=[2],
                    license="software fixture",
                    reference_text="Original complete reference " * length,
                ),
                schema=LOCKED_SCHEMA if split == "test" else "native_answer_example_v1",
            )
            (locked if split == "test" else train).append(e)
            parents.append(e)
    return dict(
        config=path,
        settings=resolve(path),
        data=train.finish(),
        evaluation=locked.finish(),
        parents=parents,
        folder=tmp_path,
        raw=cfg,
    )
