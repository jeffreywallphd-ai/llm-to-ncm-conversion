"""Explicit opt-in, cache-only released-checkpoint CUDA adapter check."""
import os
import pytest
import torch


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("NCM_CUDA_PARITY") != "1",reason="set NCM_CUDA_PARITY=1 on a budgeted CUDA host with locked cached weights")
def test_released_flat_adapter_cuda_parity():
    from ncm_lab.models.hierarchy import ProgressiveDrafter
    from ncm_lab.models.upstream import load_pair, FeatureExtractor
    from ncm_lab.runtime.io import root, read
    if not torch.cuda.is_available():
        pytest.fail("CUDA build/device required; CPU-only install cannot validate GPU parity")
    config = read(root()/"configs/models/qwen3_8b_dflash.yaml")
    target,backbone,_ = load_pair(config,"cuda",torch.bfloat16)
    wrapper = ProgressiveDrafter(backbone,target,[15]*5,skip=False).eval()
    ids = torch.tensor([[1,4,7,10,12,14,15,16]],device="cuda")
    with torch.no_grad():
        features = FeatureExtractor(target,backbone.target_layer_ids)(ids)
        prefix = torch.arange(8,device="cuda")[None]
        anchor = torch.tensor([8],device="cuda")
        anchor_token = torch.tensor([4],device="cuda")
        actual = wrapper(features,prefix,torch.zeros_like(prefix),anchor,anchor_token,torch.zeros_like(anchor))
        noise = torch.full((1,16),backbone.mask_token_id,device="cuda",dtype=torch.long)
        noise[:,0] = anchor_token
        positions = torch.arange(24,device="cuda")[None]
        direct = backbone(position_ids=positions,target_hidden=features,noise_embedding=wrapper.embeddings(noise))
        expected = backbone.compute_logits(direct[:,1:],wrapper.head)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
