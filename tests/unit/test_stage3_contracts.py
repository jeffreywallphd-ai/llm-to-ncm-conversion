"""Independent recipe, natural completion, scheduler and offline interface contracts."""

import math
import pytest
import torch
from ncm_lab.stage3.schedule import Schedule
from ncm_lab.stage3.config import resolve
from ncm_lab.stage3.data import acceptance, generation_policy
from ncm_lab.stage3.launcher import main, parser
from ncm_lab.teacher.model import load_tokenizer, load_original
from ncm_lab.validation.fixtures import model_spec


def test_defaults_and_independent_stage2():
    settings = resolve()
    assert settings["canvas"]["positions"] == 256 and settings["target_policy"] == "teacher-consistent"
    assert settings["recipe_label"] == "standard_research_candidate"
    assert settings["stages"]["source"]["data"]["format"] == "tulu_parquet"
    for name, expected in (("codec", (10000, 4, 8, 500)), ("generator", (20000, 1, 16, 1000))):
        cfg = settings["stages"][name]
        t = cfg["training"]
        assert (
            t["steps"],
            t["batch_size"],
            t["accumulation"],
            cfg["stage3"]["schedule"]["warmup"],
        ) == expected
        assert t["warmup_processed_tokens"] == 0
        assert t["steps"] * t["batch_size"] * t["accumulation"] == 320000
        assert not cfg["budget"]["enforce_wall_time_limits"]
    from ncm_lab.stage1.config import resolve as legacy

    old = legacy("configs/stage2/experiment.yaml")
    assert old["canvas"]["positions"] == 64 and old["stages"]["generator"]["training"]["steps"] == 4000
    assert not old["stages"]["generator"].get("stage3")


@pytest.mark.parametrize(
    "name,size", [("xsmall", 8), ("small", 32), ("medium", 64), ("large", 256), ("xlarge", 1024)]
)
def test_alias_and_variant(name, size):
    assert resolve(canvas_size=name)["canvas"] == resolve(canvas_size=str(size))["canvas"]
    assert resolve(target_policy="reference-only")["recipe_label"] == "explicit_variant"


@pytest.mark.parametrize("total,warmup", [(10000, 500), (20000, 1000), (5, 0)])
def test_analytic_endpoints_and_group_ratio(total, warmup):
    from ncm_lab.training.optimizer import TiledAdamW

    p, q = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    opt = TiledAdamW([("backbone.p", p), ("new.q", q)], lr=1e-5, new_lr=1e-4)
    schedule = Schedule(total, warmup)
    assert schedule.factor(1) == pytest.approx(1 / warmup if warmup else 1)
    assert schedule.factor(total) == pytest.approx(0.1)
    if warmup:
        assert schedule.factor(warmup) == 1
        expected = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / (total - warmup)))
        assert schedule.factor(warmup + 1) == pytest.approx(expected)
    schedule.apply(opt, [1e-5, 1e-4], 1)
    assert opt.groups[1]["lr"] / opt.groups[0]["lr"] == pytest.approx(10)
    state = schedule.state([1e-5, 1e-4], 0)
    p.grad = torch.tensor([float("nan")])
    q.grad = torch.ones(1)
    with pytest.raises(ValueError):
        opt.step()
    assert schedule.state([1e-5, 1e-4], 0) == state


@pytest.mark.parametrize("total,warmup", [(0, 0), (10, -1), (10, 10), (10, 11), (1.5, 0)])
def test_invalid_schedule(total, warmup):
    with pytest.raises(ValueError):
        Schedule(total, warmup)


def test_natural_eos_and_generation_neutralization():
    tokenizer = load_tokenizer(model_spec())
    model = load_original(model_spec(), frozen=True)
    model.generation_config.forced_eos_token_id = 2
    model.generation_config.min_new_tokens = 999
    policy = generation_policy(model, tokenizer, 256, [2])
    assert policy.forced_eos_token_id is None and policy.min_new_tokens == 0 and policy.max_time is None
    assert acceptance([8] * 255 + [2], tokenizer, [2], 256, 1.0) is None
    assert acceptance([8] * 256, tokenizer, [2], 256, 0.9) == "no_natural_eos"
    assert acceptance([2], tokenizer, [2], 256, 0.9) == "empty_content"
    assert acceptance([8, 2, 9, 2], tokenizer, [2], 256, 0.9) == "interior_eos"
    assert acceptance([8] * 32 + [2], tokenizer, [2], 256, 0.9) == "degenerate_repetition"


def test_offline_no_action_and_mutual_exclusion(monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError("offline planning allocated/scanned")

    monkeypatch.setattr("ncm_lab.teacher.model.load_original", forbidden)
    monkeypatch.setattr("ncm_lab.teacher.model.load_tokenizer", forbidden)
    assert main([]) == main(["--plan-only"]) == 0
    output = capsys.readouterr().out
    assert '"research_work_started": false' in output
    assert "model_asset_acquisition" in output and "source_asset_acquisition" in output
    assert "configs/validation/real_pilot.yaml" in output
    assert "configs/preparation/tulu3_sources.yaml" in output
    with pytest.raises(SystemExit):
        parser().parse_args(["--execute", "--qualify-only"])


def test_retention_preserves_two_recoveries_and_their_selected_dependencies(tmp_path):
    from ncm_lab.training.checkpoints import save_checkpoint, RECOVERY, INFERENCE, CheckpointReader
    from ncm_lab.training.engine import prune
    from ncm_lab.runtime.io import read

    folder = tmp_path / "checkpoints"

    def save(kind, step, selected=None):
        return save_checkpoint(
            folder,
            {
                "identity": "software-fixture-retention",
                "model": {"p": torch.ones(2)},
                "training": {"optimizer_steps": step, "selected": selected, "selected_this_step": False},
            },
            kind=kind,
            pointer="selected.json" if kind == INFERENCE else "current.json",
            shard_bytes=1024,
        )

    first = save(RECOVERY, 1)
    best = save(INFERENCE, 2)
    second = save(RECOVERY, 3, best)
    stale = save(INFERENCE, 4)
    current = save(INFERENCE, 5)
    pending = folder / ".pending-failed-write"
    pending.mkdir()
    prune(folder, current, separate_selection=True)
    for ref in (first, best, second, current):
        assert __import__("pathlib").Path(ref["manifest"]).exists()
    assert not __import__("pathlib").Path(stale["manifest"]).exists()
    assert pending.is_dir()
    assert CheckpointReader(folder / "current.json").path == __import__("pathlib").Path(second["manifest"])
    with pytest.raises(ValueError, match="cannot resume"):
        CheckpointReader(best["manifest"])
    assert read(best["manifest"])["schema"] == INFERENCE


def test_stage3_policy_cannot_weaken_historical_recipe(stage1_fixture):
    from ncm_lab.runtime.io import read, write_json
    from ncm_lab.stage1.config import resolve as historical_resolve

    path = stage1_fixture["experiment"]["stages"]["codec"]
    cfg = read(path)
    cfg["stage3"] = {"schema": "stage3_single_pass_v1"}
    write_json(path, cfg)
    with pytest.raises(ValueError, match="independent Stage 3 recipe"):
        historical_resolve(stage1_fixture["config"])
