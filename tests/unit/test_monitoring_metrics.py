import math
import torch
import pytest
from ncm_lab.config.architecture import Architecture, tiny_architecture
from ncm_lab.training.losses import auxiliary_loss
from ncm_lab.monitoring.records import Measure, Accumulator, mappings, mean_measure
from ncm_lab.monitoring.metrics import quantities
from ncm_lab.monitoring.diagnostics import Activity, gradients
from ncm_lab.teacher.model import load_generator
from ncm_lab.validation.fixtures import model_spec


def test_raw_normalized_weighted_errors_and_masked_padding():
    predictions = [torch.tensor([[[2.0, -2.0], [99999.0, -99999.0]]]) for _ in range(16)]
    targets = torch.zeros(1, 2, 2)
    valid = torch.tensor([[True, False]])
    details = []
    total, groups, counts = auxiliary_loss(
        predictions, targets, valid, [2] * 32, {str(i): [2.0, 1.0] for i in range(1, 32, 2)}, details
    )
    assert total == 2.5 and counts == [1] * 16
    values = quantities(
        {"loss": float(total * 0.1) + 3.0, "text_loss": 3.0, "latent_loss": float(total)},
        {"groups": details},
        generator=True,
        auxiliary_weight=0.1,
        answer_count=1,
        stage="generator",
        split="train",
    )
    prefix = "generator/train/aux/group_01/"
    assert values[prefix + "raw_mse"].value == 4
    assert values[prefix + "normalized_mse"].value == 2.5
    assert values[prefix + "weighted_contribution"].value == 0.015625
    assert sum(
        v.value for k, v in values.items() if k.endswith("weighted_contribution")
    ) + 3.0 == pytest.approx(3.25)


def test_count_weighted_data_is_distinct_from_actual_microbatch_objective():
    acc = Accumulator()
    acc.add({"data": mean_measure(1, 1), "objective": Measure(1)})
    acc.add({"data": mean_measure(27, 3), "objective": Measure(9)})
    result = acc.finish()
    assert result["data"].value == 7 and result["data"].count == 4
    assert result["objective"].value == 5
    assert Measure(None, 0).json()["status"] == "unavailable"
    assert Measure(float("nan")).json()["status"] == "nonfinite"
    assert Measure(float("inf")).json()["value"] is None


def test_default_and_repeated_resolution_mappings_have_stable_ids():
    default = mappings(Architecture().to_dict())
    assert len(default["groups"]) == 16 and len(default["updates"]) == 32
    for group in default["groups"]:
        assert group["block"] == 8 + group["group"] and group["update"] == 2 * group["group"]
    tiny = mappings(tiny_architecture().to_dict())
    assert len(tiny["updates"]) == 32 and len({r["resolution"] for r in tiny["updates"]}) < 32


def test_activity_is_gated_and_checkpoint_recompute_is_not_observed_twice():
    model = load_generator(model_spec(), tiny_architecture().to_dict(), "cpu", torch.float32)
    model.train()
    observer = Activity()
    model.diagnostic_observer = observer
    result = model(torch.tensor([[1, 4]]), torch.ones(1, 2, dtype=torch.bool), return_aux=True)
    assert len(observer.rows) == 32
    assert observer.rows[0]["latent_refinement_base_ratio"] is None
    assert observer.rows[0]["latent_base_rms"] == 0.0
    assert observer.rows[0]["gate_effective"] == pytest.approx(0.1)
    result.answer_states.square().mean().backward()
    assert len(observer.rows) == 32
    assert all(isinstance(value, (int, float, type(None))) for row in observer.rows for value in row.values())
    diagnostic = gradients(model)
    assert sum(v.count for k, v in diagnostic.items() if "shared_group_" in k) == sum(
        1 for n, _ in model.named_parameters() if n.startswith("refinements.")
    )
    model.backbone.requires_grad_(False)
    model.backbone.model.norm.weight.grad = None
    diagnostic = gradients(model)
    assert diagnostic["gradients/backbone/model/norm/weight/combined_l2"].status == "frozen"
    assert all(v.value is None or math.isfinite(v.value) for v in diagnostic.values())
