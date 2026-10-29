import math
import torch
import pytest
from ncm_lab.teacher.model import load_generator
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.inference.export import generate_tokens
from ncm_lab.inference.policy import Inference


def test_chunked_head_ids_eos_partial_chunks_and_one_transfer(monkeypatch):
    model = load_generator(model_spec(), tiny_architecture().to_dict(), "cpu", torch.float32).eval()
    import ncm_lab.inference.export as export

    original = export.transfer_token_ids
    counts = {"head": 0, "positions": 0, "transfer": 0, "blocks": 0}

    def project(states):
        counts["head"] += 1
        n = states.shape[1]
        start = counts["positions"]
        counts["positions"] += n
        logits = states.new_zeros(1, n, 32)
        logits[..., 4] = 20.0
        if start <= 5 < start + n:
            logits[0, 5 - start, 2] = 40.0
        return logits

    def transfer(ids):
        counts["transfer"] += 1
        assert ids.numel() == 16
        return original(ids)

    monkeypatch.setattr(model.backbone.lm_head, "forward", project)
    monkeypatch.setattr(export, "transfer_token_ids", transfer)

    def hook(*a):
        counts["blocks"] += 1

    handles = [layer.register_forward_hook(hook) for layer in model.backbone.model.layers]
    try:
        for size in (1, 3, 7, 16, 128):
            counts.update(head=0, positions=0, transfer=0, blocks=0)
            result = generate_tokens(
                model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), [2], chunk_positions=size
            )
            assert result["tokens"] == [4] * 5 + [2] and result["eos_found"] and not result["truncated"]
            assert counts == {
                "head": math.ceil(16 / min(size, 16)),
                "positions": 16,
                "transfer": 1,
                "blocks": 36,
            }
            assert result["inference_projection"]["resolved_head_chunk_positions"] == min(size, 16)
        counts.update(head=0, positions=0, transfer=0, blocks=0)
        result = generate_tokens(model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), [31])
        assert result["truncated"] and len(result["tokens"]) == 16 and counts["head"] == 1
    finally:
        for handle in handles:
            handle.remove()


def test_full_canvas_chunk_and_memory_guard_before_backbone(monkeypatch):
    model = load_generator(model_spec(), tiny_architecture(1024).to_dict(), "cpu", torch.float32).eval()
    calls = []
    original = model.backbone.lm_head.forward

    def head(states):
        calls.append(states.shape[1])
        return original(states)

    monkeypatch.setattr(model.backbone.lm_head, "forward", head)
    result = generate_tokens(model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), [2])
    assert calls == [128] * 8 and result["computed_answer_positions"] == 1024
    calls.clear()
    generate_tokens(
        model, torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), [2], chunk_positions=1024
    )
    assert calls == [1024]

    def forbidden(*a, **kw):
        raise AssertionError("allocation guard ran too late")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="memory budget"):
        generate_tokens(
            model,
            torch.tensor([[1, 4]]),
            torch.ones(1, 2, dtype=torch.bool),
            [2],
            inference={"max_head_temp_bytes": 1},
        )
    policy = Inference(head_chunk_positions=1024, max_head_temp_bytes=1000000000).resolve(1024, 151936, 2)
    assert policy["estimated_temporary_bytes"] == 2 * 1024 * 151936 * 2 + 1024 * 8
