import pytest
import torch
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.teacher.model import load_original
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.embedded.model import EmbeddedGenerator


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available; no real-8B validation implied"
)
def test_tiny_native_cuda_kernel_forward_backward():
    model = EmbeddedGenerator(load_original(model_spec(), frozen=False), tiny_architecture()).to("cuda")
    output = model(
        torch.tensor([[1, 4]], device="cuda"),
        torch.ones(1, 2, device="cuda", dtype=torch.bool),
        return_aux=True,
    )
    output.answer_states.square().mean().backward()
    assert torch.isfinite(model.queries.grad).all()
