from dataclasses import replace
import pytest
import torch
from ncm_lab.training.checkpoints import save_checkpoint, CheckpointReader, RECOVERY, INFERENCE
from ncm_lab.training.optimizer import TiledAdamW
from ncm_lab.runtime.io import read
from ncm_lab.runtime.budget import Budget, Meter, BudgetStop


def test_tiled_fp32_moment_adam_matches_pytorch_without_master_parameters():
    torch.manual_seed(3)
    a = torch.nn.Linear(5, 3, bias=False)
    b = torch.nn.Linear(5, 3, bias=False)
    b.load_state_dict(a.state_dict())
    tiled = TiledAdamW(a.named_parameters(), lr=0.001, weight_decay=0.01, tile_elements=4)
    reference = torch.optim.AdamW(b.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.randn(2, 5)
    for _ in range(3):
        tiled.zero_grad()
        reference.zero_grad()
        a(x).square().sum().backward()
        b(x).square().sum().backward()
        tiled.step()
        reference.step()
    assert torch.allclose(a.weight, b.weight, atol=1e-7)
    assert tiled.state["weight"]["exp_avg"].dtype == torch.float32
    assert set(tiled.state["weight"]) == {"step", "exp_avg", "exp_avg_sq"}


def test_shards_are_bounded_and_stream_into_complete_model(tmp_path, monkeypatch):
    model = torch.nn.Linear(200, 100, bias=False)
    payload = {
        "identity": "native-test",
        "model": model.state_dict(keep_vars=True),
        "metadata": {"component": "fixture"},
        "training": {"steps": 2},
    }
    pointer = save_checkpoint(tmp_path, payload, shard_bytes=16384)
    manifest = read(pointer["manifest"])
    assert len(manifest["files"]) >= 5
    for name in manifest["files"]:
        shard = torch.load(
            __import__("pathlib").Path(pointer["manifest"]).parent / name, weights_only=True, mmap=True
        )
        assert sum(t.numel() * t.element_size() for t in shard.values()) <= 16384
    expected = model.weight.detach().clone()
    model.weight.data.zero_()
    reader = CheckpointReader(tmp_path / "current.json", identity="native-test")
    reader.load_model(model)
    assert torch.equal(model.weight, expected)
    with pytest.raises(ValueError, match="stream directly"):
        reader.materialize(reader.field("model"), max_tensor_bytes=1024)
    with pytest.raises(ValueError, match="identity"):
        CheckpointReader(tmp_path / "current.json", identity="wrong")


@pytest.mark.parametrize(
    "stage",
    [
        "shard_finished",
        "before_manifest",
        "manifest_written",
        "before_commit",
        "after_commit",
        "before_pointer",
        "after_pointer",
    ],
)
@pytest.mark.parametrize("previous", [False, True])
def test_atomic_checkpoint_publication_is_previous_new_or_absent(tmp_path, monkeypatch, stage, previous):
    import ncm_lab.training.checkpoints as checkpoint

    if previous:
        save_checkpoint(tmp_path, {"identity": "native", "model": {"x": torch.tensor([1.0])}})

    def interrupt(at):
        if at == stage:
            raise RuntimeError("injected process boundary")

    monkeypatch.setattr(checkpoint, "boundary", interrupt)
    with pytest.raises(RuntimeError):
        save_checkpoint(tmp_path, {"identity": "native", "model": {"x": torch.tensor([2.0])}})
    path = tmp_path / "current.json"
    if not previous and stage != "after_pointer":
        assert not path.exists()
    else:
        reader = CheckpointReader(path)
        value = reader.materialize(reader.field("model"))["x"].item()
        assert value == (2.0 if stage == "after_pointer" else 1.0)


def test_corrupt_shard_and_inference_resume_are_rejected(tmp_path):
    pointer = save_checkpoint(
        tmp_path, {"identity": "native", "model": {"x": torch.ones(10000)}}, shard_bytes=4096
    )
    manifest = read(pointer["manifest"])
    file = next(iter(manifest["files"]))
    path = __import__("pathlib").Path(pointer["manifest"]).parent / file
    with path.open("r+b") as stream:
        stream.seek(100)
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        CheckpointReader(tmp_path / "current.json")
    other = tmp_path / "export"
    save_checkpoint(other, {"identity": "native", "model": {"x": torch.ones(2)}}, kind=INFERENCE)
    with pytest.raises(ValueError, match="schema"):
        CheckpointReader(other / "current.json", expected_kind=RECOVERY)


def test_aggregate_work_and_crash_reservations_survive_new_run_ids(tmp_path):
    budget = Budget("fixed", max_steps=2, max_phase_steps=3, max_wall_seconds=10, max_phase_wall_seconds=30)
    first = tmp_path / "first"
    first.mkdir()
    meter = Meter(budget, first, tmp_path / "ledger")
    meter.charge(steps=2, source=11)
    # An abrupt process exit does not refund its wall reservation or its charges.
    second = tmp_path / "second"
    second.mkdir()
    next_meter = Meter(budget, second, tmp_path / "ledger")
    next_meter.charge(steps=1)
    with pytest.raises(BudgetStop):
        next_meter.charge(steps=1)
    state = read(tmp_path / "ledger/fixed.json")
    assert state["charges"]["source_tokens"] == 11 and state["wall"] == 20
    next_meter.close()
    meter.close()
    with pytest.raises(ValueError, match="cannot change"):
        Meter(replace(budget, max_phase_steps=4), second, tmp_path / "ledger")
