"""A single bounded tiny CUDA fixture; never a full-8B feasibility claim."""

import random
import pytest
import torch
from ncm_lab.block8.config import Training
from ncm_lab.block8.model import BlockDecoder
from ncm_lab.block8.training import example_loss
from ncm_lab.teacher.model import load_original
from ncm_lab.training.optimizer import TiledAdamW
from ncm_lab.validation.fixtures import model_spec


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; tiny software fixture only")
def test_tiny_block8_bf16_cuda_gradient_optimizer_and_cache():
    spec = model_spec()
    spec["config"]["num_hidden_layers"] = 2
    model = BlockDecoder(load_original(spec, device="cuda", dtype=torch.bfloat16, frozen=False)).train()
    optimizer = TiledAdamW(model.named_parameters(), tile_elements=4096)
    loss, _, _ = example_loss(model, [1, 4, 5], [8, 9, 10, 2], 8, random.Random(0), Training())
    loss.backward()
    report = optimizer.check_gradients()
    report.clip_(1)
    optimizer.step(report)
    optimizer.zero_grad()
    assert optimizer.rounding_draws > 0 and model.mask_embedding.isfinite().all()
    model.eval()
    with torch.inference_mode():
        context = torch.tensor([[1, 4, 5]], device="cuda")
        active = torch.full((1, 8), -1, device="cuda", dtype=torch.long)
        cache = model.prefill(context)
        full = model.denoise(context, active)
        cached = model.denoise(context, active, prefix_cache=cache)
        torch.testing.assert_close(cached, full, atol=0.05, rtol=0.05)
        assert cache.get_seq_length() == 3
