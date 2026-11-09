"""One bounded tiny BF16 CUDA hierarchy fixture; no full-8B feasibility claim."""

from dataclasses import asdict
import pytest
import torch
from ncm_lab.block8.config import Training, Decoding
from ncm_lab.block8.model import prefill, generate
from ncm_lab.block8.training import example_loss
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.embedded.model import EmbeddedGenerator
from ncm_lab.teacher.model import load_original
from ncm_lab.training.optimizer import TiledAdamW
from ncm_lab.validation.fixtures import model_spec


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; tiny hierarchy fixture only")
def test_tiny_hierarchical_block8_bf16_cuda_gradient_optimizer_and_cache():
    spec = model_spec()
    spec["config"].update(hidden_size=8, intermediate_size=16, num_attention_heads=1, head_dim=8)
    model = (
        EmbeddedGenerator(
            load_original(spec, device="cuda", dtype=torch.bfloat16, frozen=False),
            tiny_architecture(8, activation_checkpointing=True),
        )
        .to("cuda", dtype=torch.bfloat16)
        .train()
    )
    optimizer = TiledAdamW(model.named_parameters(), tile_elements=4096)
    fine = torch.randn(1, 8, 8, device="cuda", dtype=torch.float32)
    loss, _, counts = example_loss(
        model, [1, 4, 5], [8, 9, 10, 2], fine, {str(i): [1.0] * 8 for i in range(1, 32, 2)}, Training()
    )
    loss.backward()
    report = optimizer.check_gradients()
    report.clip_(1)
    optimizer.step(report)
    optimizer.zero_grad()
    assert optimizer.rounding_draws > 0 and len(counts) == 16 and model.queries.isfinite().all()
    model.eval()
    with torch.inference_mode():
        context = torch.tensor([[1, 4, 5]], device="cuda")
        cache = prefill(model, context)
        full = model(context, torch.ones_like(context, dtype=torch.bool), return_aux=True)
        cached = model.forward_cached_context(
            context, torch.ones_like(context, dtype=torch.bool), cache, return_aux=True
        )
        torch.testing.assert_close(cached.answer_states, full.answer_states, atol=0.08, rtol=0.08)
        assert len(cached.cumulative_codes) == 16 and cache.get_seq_length() == 3
        result = generate(
            model, [1, 4, 5], asdict(Decoding(max_new_tokens=16, max_blocks=2)), [2], max_context_tokens=32
        )
        assert result["hierarchy_updates"] in {32, 64}
