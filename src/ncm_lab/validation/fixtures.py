"""Small Qwen production classes and labeled software data; no research weights."""

from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.teacher.model import model_identity
from ncm_lab.inference.export import fixture_tokenizer
from ncm_lab.runtime.io import canonical_hash
from ncm_lab.data.artifacts import ArtifactWriter
from ncm_lab.data.records import Example


def model_spec():
    return {
        "model_id": "software-fixture/qwen3",
        "revision": "fixture-7",
        "tokenizer_revision": "fixture-7",
        "thinking": False,
        "fixture": True,
        "eos_ids": [2],
        "seed": 7,
        "config": {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 36,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 40960,
            "tie_word_embeddings": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
    }


def fixture_data(path, architecture=None, meter=None, train=8, dev=4):
    architecture = architecture or tiny_architecture()
    template = canonical_hash(fixture_tokenizer().chat_template)
    identity = {
        "model": model_identity(model_spec()),
        "template_sha256": template,
        "source_sha256": canonical_hash(["native-fixture-v1", train, dev]),
        "canvas": architecture.canvas,
        "thinking": False,
    }
    writer = ArtifactWriter(path, "native_data_v1", identity, meter)
    for split, count in [("train", train), ("dev", dev)]:
        for i in range(count):
            example = Example(
                split + str(i),
                split + str(i),
                split,
                "software_fixture",
                [1, 4 + i % 4],
                [8 + i % 4, 12 + i % 3, 2],
                True,
                False,
                {"template_sha256": template, "scope": "software fixture only", "eos_ids": [2]},
            )
            writer.append(example)
    return writer.finish({"fixture": True, "long_answer_learning": False})
