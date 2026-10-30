import copy
import json
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from ncm_lab.data.artifacts import Dataset
from ncm_lab.data.capacity import qualified_selection
from ncm_lab.data.tokenization import prompt_tokens, validated_messages
from ncm_lab.runtime.runner import execute
from ncm_lab.runtime.io import digest, read
from ncm_lab.teacher.model import load_tokenizer


def replace_rows(fixture, rows):
    asset = fixture["assets"][0]
    path = Path(fixture["data"]["data"]["assets_root"]) / asset["path"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    asset.update(size_bytes=path.stat().st_size, sha256=digest(path))
    second = fixture["assets"][1]
    path = Path(fixture["data"]["data"]["assets_root"]) / second["path"]
    pq.write_table(pa.Table.from_pylist(fixture["rows"][1::2]), path)
    second.update(size_bytes=path.stat().st_size, sha256=digest(path))
    fixture["lock"].write_text(json.dumps({"assets": fixture["assets"]}))


def test_multishard_context_provenance_capacity_and_order(tulu_fixture):
    cfg = tulu_fixture["data"]
    first = execute(cfg)
    assert first["status"] == "completed", first
    data = Dataset(first["artifact"])
    report = read(first["capacity"]["manifest"])
    assert report["scan"]["complete"] and report["scan"]["excluded"]["canonical_duplicate"] == 1
    assert report["selected"]["train"]["answer_positions"] >= 20
    selection = qualified_selection(data, 20, 3)
    assert selection == report["codec_selection"]
    assert 0 <= selection["whole_example_overshoot"] < 16
    groups = {s: set() for s in ("train", "dev")}
    prompts = {s: set() for s in ("train", "dev")}
    tokenizer = load_tokenizer(cfg["model"])
    for i, row in data.iter_metadata():
        e = row["example"]
        groups[e["split"]].add(e["group_id"])
        prompts[e["split"]].add(tuple(e["prompt_ids"]))
        raw = next(r for r in tulu_fixture["rows"] if r["id"] == e["provenance"]["original_row_id"])
        assert e["prompt_ids"] == prompt_tokens(tokenizer, raw["messages"][:-1])
        assert e["prompt_ids"] != prompt_tokens(tokenizer, raw["messages"][-2]["content"])
        assert e["answer_ids"] == tokenizer.encode(
            raw["messages"][-1]["content"], add_special_tokens=False
        ) + [2]
        assert e["provenance"]["answer_origin"] == "unknown"
        assert e["provenance"]["license"] == "CC-BY-NC-4.0"
        assert e["provenance"]["shard_sha256"] in {a["sha256"] for a in tulu_fixture["assets"]}
    assert not groups["train"] & groups["dev"] and not prompts["train"] & prompts["dev"]
    reversed_cfg = copy.deepcopy(cfg)
    reversed_cfg["data"]["source_asset_ids"].reverse()
    second = execute(reversed_cfg)
    assert second["status"] == "completed", second
    assert Dataset(second["artifact"]).sha256 == data.sha256
    assert read(second["capacity"]["manifest"])["codec_selection"] == selection


def test_rejections_overlap_language_and_bounds(tulu_fixture):
    rows = copy.deepcopy(tulu_fixture["rows"][::2])
    base = copy.deepcopy(rows[1])
    cases = []
    for name, change in [
        ("bad_role", lambda r: r["messages"][-1].update(role="tool")),
        ("empty", lambda r: r["messages"][-1].update(content="")),
        ("oversized", lambda r: r["messages"][-1].update(content="x" * 131073)),
        (
            "eos",
            lambda r: r["messages"][-1].update(
                content="Some complete English answer <|im_end|> hidden text."
            ),
        ),
        (
            "foreign",
            lambda r: r["messages"][-1].update(
                content="Ceci est une réponse entièrement française qui décrit clairement un exemple de logiciel."
            ),
        ),
        ("uncertain", lambda r: r["messages"][-1].update(content="Yes")),
        (
            "answer_length",
            lambda r: r["messages"][-1].update(
                content="This English explanation is repeated for a complete answer. " * 3
            ),
        ),
        (
            "prompt_length",
            lambda r: r["messages"][1].update(content="This is a complete English prompt. " * 100),
        ),
        ("excluded", lambda r: r.update(source="ai2-adapt-dev/oasst1_converted")),
    ]:
        item = copy.deepcopy(base)
        item["id"] = name
        change(item)
        cases.append(item)
    replace_rows(tulu_fixture, rows + cases)
    result = execute(tulu_fixture["data"])
    assert result["status"] == "completed", result
    counts = read(result["capacity"]["manifest"])["scan"]["excluded"]
    for reason in (
        "message_roles",
        "empty_message",
        "message_chars",
        "special_delimiter",
        "language_non_english",
        "language_uncertain",
        "answer_length",
        "prompt_length",
        "source_policy:oasst_component",
    ):
        assert counts.get(reason, 0) > 0, (reason, counts)


def test_insufficient_capacity_blocks_standalone_teacher_before_model(tulu_fixture, monkeypatch):
    tulu_fixture["data"]["data"]["codec_minimum_answer_positions"] = 10000
    result = execute(tulu_fixture["data"])
    assert result["status"] == "blocked", result
    import ncm_lab.teacher.stages as stages

    monkeypatch.setattr(
        stages, "load_original", lambda *a, **kw: pytest.fail("teacher allocated before capacity gate")
    )
    cfg = {
        **tulu_fixture["common"],
        "stage": "teacher",
        "inputs": {"data": result["artifact"]},
        "data": {"max_unique_answer_positions": 10000, "max_dev_examples": 3},
        "output_ref": None,
    }
    teacher = execute(cfg)
    assert teacher["status"] == "failed" and "before teacher allocation" in teacher["error"]


def test_messages_and_string_api_have_same_rendering(tulu_fixture):
    tokenizer = load_tokenizer(tulu_fixture["common"]["model"])
    assert prompt_tokens(tokenizer, "token5") == prompt_tokens(
        tokenizer, [{"role": "user", "content": "token5"}]
    )
    for messages in (
        [],
        [{"role": "assistant", "content": "bad"}],
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "target"}],
        [{"role": "user", "content": ["image"]}],
    ):
        with pytest.raises(ValueError):
            validated_messages(messages)


def test_exact_prompt_and_answer_boundaries(tulu_fixture):
    from ncm_lab.config.architecture import tiny_architecture

    tokenizer = load_tokenizer(tulu_fixture["common"]["model"])
    words = ["This", "is", "a", "clear", "English", "explanation"]

    def text(count):
        return " ".join(words[i % len(words)] for i in range(count))

    rows = copy.deepcopy(tulu_fixture["rows"][::2])
    for name, prompt_length, answer_length in [
        ("boundary", 512, 1024),
        ("prompt_too_long", 513, 1024),
        ("answer_too_long", 512, 1025),
    ]:
        row = {
            "id": name,
            "group_id": name,
            "source": rows[0]["source"],
            "messages": [
                {"role": "user", "content": text(prompt_length - 1)},
                {"role": "assistant", "content": text(answer_length - 1)},
            ],
        }
        assert len(prompt_tokens(tokenizer, row["messages"][:-1])) == prompt_length
        assert (
            len(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)) + 1
            == answer_length
        )
        rows.append(row)
    replace_rows(tulu_fixture, rows)
    cfg = tulu_fixture["data"]
    cfg["architecture"] = tiny_architecture(canvas=1024).to_dict()
    cfg["data"]["max_answer_tokens"] = 1024
    result = execute(cfg)
    assert result["status"] == "completed", result
    data = Dataset(result["artifact"])
    boundary = [
        e
        for _, r in data.iter_metadata()
        if (e := r["example"])["provenance"]["original_row_id"] == "boundary"
    ]
    assert (
        len(boundary) == 1
        and len(boundary[0]["prompt_ids"]) == 512
        and len(boundary[0]["answer_ids"]) == 1024
    )
    counts = read(result["capacity"]["manifest"])["scan"]["excluded"]
    assert counts["prompt_length"] == counts["answer_length"] == 1


def test_explicit_oasst_heldout_prompt_excludes_alternative_targets(tulu_fixture):
    from ncm_lab.data.artifacts import ArtifactWriter
    from ncm_lab.data.records import Example
    from ncm_lab.teacher.model import model_identity
    from ncm_lab.runtime.io import canonical_hash

    tokenizer = load_tokenizer(tulu_fixture["common"]["model"])
    template = canonical_hash(tokenizer.chat_template)
    writer = ArtifactWriter(
        tulu_fixture["folder"] / "heldout-overlap",
        "native_data_v1",
        {"model": model_identity(tulu_fixture["common"]["model"]), "template_sha256": template, "canvas": 16},
    )
    raw = tulu_fixture["rows"][0]
    writer.append(
        Example(
            "heldout",
            "heldout",
            "dev",
            "software_fixture",
            prompt_tokens(tokenizer, raw["messages"][:-1]),
            tokenizer.encode(raw["messages"][-1]["content"], add_special_tokens=False) + [2],
            True,
            False,
            {"template_sha256": template, "eos_ids": [2]},
        )
    )
    manifest = writer.finish()
    tulu_fixture["data"]["data"]["overlap"] = {
        "artifact": str(manifest),
        "policy": "explicit_heldout_exact_prompt_v1",
    }
    result = execute(tulu_fixture["data"])
    assert result["status"] == "completed", result
    counts = read(result["capacity"]["manifest"])["scan"]["excluded"]
    assert counts["oasst_exact_prompt_or_reference"] == 3
    assert not {"0", "duplicate", "alternative"} & {
        r["example"]["provenance"]["original_row_id"] for _, r in Dataset(result["artifact"]).iter_metadata()
    }


def test_changed_preprocessing_and_shard_content_invalidate_source(tulu_fixture, monkeypatch):
    first = execute(tulu_fixture["data"])
    assert first["status"] == "completed", first
    initial = Dataset(first["artifact"])
    changed = copy.deepcopy(tulu_fixture["data"])
    changed["data"]["language"]["minimum_probability"] = 0.91
    second = execute(changed)
    assert second["status"] == "completed", second
    newer = Dataset(second["artifact"])
    assert newer.sha256 != initial.sha256
    assert newer.manifest["identity"]["source_sha256"] != initial.manifest["identity"]["source_sha256"]
    asset = tulu_fixture["assets"][0]
    path = Path(changed["data"]["assets_root"]) / asset["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    import ncm_lab.data.sources.tulu as adapter

    monkeypatch.setattr(
        adapter, "load_tokenizer", lambda *a: pytest.fail("tokenizer loaded before corrupt source rejection")
    )
    rejected = execute(changed)
    assert rejected["status"] == "failed" and "missing/corrupt" in rejected["error"]
