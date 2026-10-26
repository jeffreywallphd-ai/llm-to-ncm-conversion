import copy
import pytest
import torch
from ncm_lab.models.hierarchy import ProgressiveDrafter, prefix_visibility
from ncm_lab.models.intervals import Partition, validate_schedule


def inputs(backbone, anchors=(4, 10)):
    batch, length = len(anchors), 24
    features = torch.randn(batch, length, backbone.fc.in_features, requires_grad=True)
    positions = torch.arange(length)[None].expand(batch, -1)
    documents = torch.zeros(batch, length, dtype=torch.long)
    return features, positions, documents, torch.tensor(anchors), torch.ones(batch, dtype=torch.long), torch.zeros(batch, dtype=torch.long)


def test_nested_partitions_and_anchor():
    parts = validate_schedule([4,4,8,15,15])
    assert parts[0].groups[-1] == (13,14,15)
    assert parts[0].representatives == [2,6,10,14]
    assert parts[0].parents(parts[2]) == [0,0,1,1,2,2,3,3]
    with pytest.raises(ValueError):
        Partition(((0,1,2),), 2)
    with pytest.raises(ValueError):
        validate_schedule([8,4,15])


@pytest.mark.parametrize("schedule,expected", [([4,4,8,15,15],[5,5,9,16,16]),
    ([8,8,8,8,15],[9,9,9,9,16]), ([15]*5,[16]*5), ([4,8,15,15,15],[5,9,16,16,16])])
def test_layers_physically_receive_smaller_tensors(pair, schedule, expected):
    target, backbone = pair
    model = ProgressiveDrafter(backbone, target, schedule, skip=False)
    observed = []
    hooks = [layer.register_forward_pre_hook(lambda module, args, kwargs: observed.append(kwargs["hidden_states"].shape[1]), with_kwargs=True) for layer in backbone.layers]
    logits = model(*inputs(backbone))
    for hook in hooks:
        hook.remove()
    assert observed == expected
    assert logits.shape == (2,15,32)


def test_forbidden_target_features_never_change_logits(pair):
    target, backbone = pair
    model = ProgressiveDrafter(backbone, target).eval()
    args = list(inputs(backbone))
    before = model(*args)
    allowed = prefix_visibility(args[1], args[2], args[3], args[5])
    args[0] = torch.where(allowed[...,None], args[0], torch.full_like(args[0], float("nan")))
    torch.testing.assert_close(model(*args), before, atol=0, rtol=0)


def test_documents_and_blocks_isolate_but_later_prefix_is_visible(pair):
    target, backbone = pair
    model = ProgressiveDrafter(backbone, target).eval()
    args = list(inputs(backbone))
    args[2][:,0] = 9  # Different document, even though before anchor.
    baseline = model(*args)
    changed = args[0].detach().clone()
    changed[:,0] += 100
    modified = [changed, *args[1:]]
    torch.testing.assert_close(model(*modified), baseline, atol=0, rtol=0)
    # Position 7 is forbidden to anchor 4, but legitimately visible to anchor 10.
    changed[:,7] += 100 * torch.randn_like(changed[:,7])
    actual = model(changed, *args[1:])
    torch.testing.assert_close(actual[0], baseline[0], atol=0, rtol=0)
    assert not torch.allclose(actual[1], baseline[1], atol=1e-7, rtol=1e-7)
    changed[1] += torch.randn_like(changed[1])
    actual2 = model(changed, *args[1:])
    torch.testing.assert_close(actual2[0], actual[0], atol=0, rtol=0)


def test_flat_adapter_agrees_with_locked_upstream(pair):
    target, backbone = pair
    original = copy.deepcopy(backbone).eval()
    model = ProgressiveDrafter(backbone, target, [15]*5, skip=False).eval()
    features, pos, docs, anchors, tokens, anchor_docs = inputs(backbone, (24,24))
    noise_ids = torch.full((2,16), backbone.mask_token_id, dtype=torch.long)
    noise_ids[:,0] = tokens
    positions = torch.cat([pos, anchors[:,None] + torch.arange(16)[None]], dim=1)
    direct = original(position_ids=positions, target_hidden=features.detach(), noise_embedding=model.embeddings(noise_ids))
    expected = original.compute_logits(direct[:,1:], model.head)
    torch.testing.assert_close(model(features,pos,docs,anchors,tokens,anchor_docs), expected, atol=1e-6, rtol=1e-5)


def test_main_path_gradients_and_skip_initialization(pair):
    target, backbone = pair
    model = ProgressiveDrafter(backbone, target)
    args = inputs(backbone)
    frozen_head = model.head.weight.detach().clone()
    output = model(*args, retain_trace=True)
    output.square().sum().backward()
    assert model.trace["coarse"].grad.abs().sum() > 0
    assert all(model.trace[name].grad.abs().sum() > 0 for name in model.trace if name.startswith("expansion"))
    assert model.skip_gate.grad.abs() > 0
    assert model.skip_up.weight.grad.abs().sum() == 0
    assert all(p.grad is None for p in target.parameters())
    assert args[0].grad is None
    assert all(p.offset.grad.abs().sum() > 0 for p in model.expansions.values())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.skip_gate.fill_(.2)
    model(*args).square().sum().backward()
    assert model.skip_up.weight.grad.abs().sum() > 0
    assert model.skip_down.weight.grad.abs().sum() > 0
    torch.testing.assert_close(model.head.weight, frozen_head, atol=0, rtol=0)
