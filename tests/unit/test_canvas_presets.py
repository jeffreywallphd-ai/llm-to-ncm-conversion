"""Bounded software geometry/contracts; no research weights or data acquisition."""

import argparse
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import shlex
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from ncm_lab.config.architecture import Architecture, RESOLUTIONS
from ncm_lab.config.canvas import PRESETS, resolve_canvas, validate_production
from ncm_lab.config.stages import config
from ncm_lab.embedded.model import EmbeddedGenerator
from ncm_lab.teacher.model import load_original
from ncm_lab.validation.fixtures import model_spec
from ncm_lab.training.losses import chunked_ce, auxiliary_loss
from ncm_lab.runtime import pilot
from ncm_lab.runtime.canvas_pipeline import pipeline_configs, namespace, exclusive_writer
from ncm_lab.runtime.io import read, write_json, canonical_hash


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_presets_aliases_configs_and_schedules(preset):
    assert all(resolve_canvas(v) == preset for v in (preset.name, preset.positions, str(preset.positions)))
    a = preset.architecture()
    validate_production(a.to_dict())
    assert a.latent_width == 384 and a.initial_queries == 4
    assert (a.early_blocks, a.hierarchy_blocks, a.late_blocks) == (8, 16, 12)
    assert len(a.resolutions) == 32 and a.resolutions[0] == 4 and a.resolutions[-1] == a.canvas
    assert all(type(v) is int and 4 <= v <= a.canvas for v in a.resolutions)
    assert sorted(a.resolutions) == list(a.resolutions)
    assert Architecture(**read(f"configs/architecture/embedded_{preset.name}.yaml")) == a
    if a.canvas == 1024:
        assert a.resolutions == RESOLUTIONS
        assert Architecture(**read("configs/architecture/embedded_1024.yaml")) == a


@pytest.mark.parametrize("value", [True, False, -1, 0, 16, 8.0, 3.5, "8.0", "Medium", "", None, [], {}])
def test_invalid_values_fail_before_children(value, monkeypatch):
    monkeypatch.setattr(pilot, "train_child", lambda *a: pytest.fail("child started"))
    with pytest.raises(ValueError):
        resolve_canvas(value)


def test_override_preserves_unrelated_fields_and_default_policy(tmp_path):
    base = replace(Architecture(), gate_initial=0.3, activation_checkpointing=False)
    a = resolve_canvas("small").architecture(base)
    assert a.gate_initial == 0.3 and a.activation_checkpointing is False and a.latent_width == 384
    cfg = config("configs/validation/generator_pilot.yaml")
    assert cfg["architecture"]["canvas"] == 1024
    del cfg["architecture"]
    path = tmp_path / "unspecified.json"
    write_json(path, cfg)
    assert config(path)["architecture"]["canvas"] == 64
    for steps in (2, 100):
        assert config(f"configs/validation/generator_{steps}_step_pilot.yaml")["architecture"]["canvas"] == 64


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_all_canvas_forward_backward_and_export_roundtrip(preset, tmp_path):
    from ncm_lab.data.records import Example, collate
    from ncm_lab.inference.export import export_generator, load_export, generate_tokens, fixture_tokenizer

    # Explicit tiny fixture hidden/latent widths, with the production 36-block/32-update structure.
    spec = model_spec()
    spec["config"].update(hidden_size=8, intermediate_size=16, num_attention_heads=1, head_dim=8)
    a = replace(preset.architecture(), latent_width=8, activation_checkpointing=False)
    model = EmbeddedGenerator(load_original(spec, frozen=False), a)
    model.set_phase("full")
    template = canonical_hash(fixture_tokenizer().chat_template)
    examples = [
        Example(
            str(i),
            str(i),
            "train",
            "software_fixture",
            [1, 4] + [5] * i,
            [8] * (i + 1) + [2],
            True,
            False,
            {"template_sha256": template, "eos_ids": [2]},
        )
        for i in range(2)
    ]
    ids, mask, labels, valid = collate(examples, a.canvas)
    updates, blocks = [], []
    hooks = [
        r.register_forward_hook(lambda m, args, out: updates.append(out[0].shape[1]))
        for r in model.refinements
    ]
    hooks += [
        layer.register_forward_pre_hook(lambda m, args: blocks.append(args[0].shape[1] - 3))
        for layer in model.backbone.model.layers
    ]
    output = model(ids, mask, return_aux=True)
    assert output.answer_states.shape == (2, a.canvas, 8)
    assert updates == list(a.resolutions) and len(blocks) == 36
    assert blocks[24:] == [a.canvas] * 12 and max(blocks) == a.canvas
    scales = {str(i): [1.0] * 8 for i in range(1, 32, 2)}
    latent, groups, counts = auxiliary_loss(
        output.cumulative_codes, torch.zeros(2, a.canvas, 8), valid, a.resolutions, scales
    )
    ce = chunked_ce(model.backbone.lm_head, output.answer_states, labels, valid, 3)
    assert len(groups) == len(counts) == 16 and torch.isfinite(ce + latent)
    (ce + 0.1 * latent).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.backbone.model.layers[35].self_attn.o_proj.weight.grad.abs().sum() > 0
    assert set(inspect.signature(model.forward).parameters) == {
        "prompt_input_ids",
        "prompt_attention_mask",
        "return_aux",
    }
    for hook in hooks:
        hook.remove()
    metadata = {
        "architecture": a.to_dict(),
        "backbone_config": model.backbone.config.to_dict(),
        "dtype": "float32",
        "model": {**spec, "fixture": True},
        "source": {"template_sha256": template},
    }
    export = export_generator(model, metadata, tmp_path / "export", None, 65536, spec, selected_step=2)
    restored, _, manifest = load_export(export)
    assert restored.architecture == a and manifest["selected_optimizer_step"] == 2
    if a.canvas == 1024:
        # Pre-canvas exports had no selected-step label; their explicit geometry remains authoritative.
        legacy = dict(manifest)
        legacy.pop("selected_optimizer_step")
        legacy_path = export.parent / "legacy-format.json"
        write_json(legacy_path, legacy)
        old, _, _ = load_export(legacy_path)
        assert old.architecture.canvas == 1024
    with torch.no_grad():
        assert torch.equal(model.eval()(ids, mask).answer_states, restored(ids, mask).answer_states)
    # Force early EOS, then force no EOS; computation still covers the full canvas.
    for eos, expected_tokens, truncated in ((True, 1, False), (False, a.canvas, True)):

        class Head(torch.nn.Module):
            weight = restored.backbone.lm_head.weight

            def forward(self, states):
                logits = states.new_zeros(*states.shape[:-1], 32)
                logits[..., 2 if eos else 7] = 1
                return logits

        restored.backbone.lm_head = Head()
        result = generate_tokens(restored, ids[:1], mask[:1], [2], chunk_positions=3)
        assert result["allocated_answer_positions"] == result["computed_answer_positions"] == a.canvas
        assert len(result["tokens"]) == expected_tokens and result["truncated"] == truncated
        assert result["eos_found"] == eos


def test_production_parameter_shapes_identical_on_meta():
    shapes = []
    for preset in PRESETS:
        with torch.device("meta"):
            backbone = Qwen3ForCausalLM(
                Qwen3Config(
                    hidden_size=4096,
                    intermediate_size=12288,
                    num_hidden_layers=36,
                    num_attention_heads=32,
                    num_key_value_heads=8,
                    head_dim=128,
                    vocab_size=151936,
                    tie_word_embeddings=False,
                )
            )
            model = EmbeddedGenerator(backbone, preset.architecture())
        assert all(p.device.type == "meta" for p in model.parameters())
        shapes.append({name: tuple(p.shape) for name, p in model.named_parameters()})
    assert all(shape == shapes[0] for shape in shapes)


def test_namespace_source_fingerprints_and_writer_lock(tmp_path):
    cfg = config("configs/validation/generator_100_step_pilot.yaml")
    cfg["run_root"] = str(tmp_path)
    paths = []
    for preset in PRESETS:
        cfg["architecture"] = preset.architecture().to_dict()
        values = pipeline_configs(cfg)
        paths.append(values["generator"]["output_ref"])
        assert values["generator"]["budget"]["enforce_wall_time_limits"] is False
        assert all(v["architecture"] == cfg["architecture"] for v in values.values())
        other = copy.deepcopy(cfg)
        other["training"]["seed"] += 1
        assert pipeline_configs(other)["targets"]["output_ref"] != values["targets"]["output_ref"]
        two = copy.deepcopy(cfg)
        two["training"].update(steps=2, validation_interval=1, checkpoint_interval=1)
        assert pipeline_configs(two)["targets"]["output_ref"] == values["targets"]["output_ref"]
    assert len(set(paths)) == 5 and all("handoffs/canvas" in Path(p).as_posix() for p in paths)
    lock = namespace(tmp_path, cfg["architecture"], "fixture") / "writer.lock"
    with exclusive_writer(lock):
        with pytest.raises(ValueError, match="writer locked"):
            with exclusive_writer(lock):
                pytest.fail("conflicting writer admitted")


@pytest.mark.parametrize("steps", [2, 100])
@pytest.mark.parametrize("size", [None, *[p.name for p in PRESETS], "64"])
def test_launchers_share_parser_plan_and_no_mutation(steps, size, tmp_path, monkeypatch):
    from ncm_lab.cli import parser

    p = argparse.ArgumentParser()
    pilot.add_arguments(p)
    tokens = ["--fresh", "--prepare", "--no-chat", "--plan-only"]
    if size:
        tokens += ["--canvas-size", size]
    args = vars(p.parse_args(tokens))
    native = vars(parser().parse_args(["pilot", "train", "--steps", str(steps), *tokens]))
    assert all(native[k] == v for k, v in args.items())
    monkeypatch.setattr(pilot, "train_child", lambda *a: pytest.fail("training during plan"))
    monkeypatch.setattr(pilot, "find_spec", lambda *a: pytest.fail("optional chat dependency during plan"))
    before = list(tmp_path.rglob("*"))
    result = pilot.run_pilot(steps, **args)
    assert result["canvas"]["positions"] == resolve_canvas(size or "medium").positions
    assert result["generator_steps"] == steps and result["data_capacity"].startswith("unknown")
    assert not any(result["clock_policy"]["stages"].values())
    assert list(tmp_path.rglob("*")) == before


def test_all_documented_pilot_commands_parse():
    from ncm_lab.cli import parser

    text = Path("README.md").read_text(encoding="utf-8")
    count = 0
    for line in text.splitlines():
        if line.startswith("python scripts/run_") and "pilot.py" in line:
            tokens = shlex.split(line)
            p = argparse.ArgumentParser()
            pilot.add_arguments(p)
            p.parse_args(tokens[2:])
            count += 1
        elif line.startswith("ncm pilot train"):
            parser().parse_args(shlex.split(line)[1:])
    assert count >= 9


@pytest.mark.parametrize(
    "command,action,path",
    [
        ("data", "prepare", "configs/data/real_pilot.yaml"),
        ("teacher", "extract", "configs/validation/teacher_features_pilot.yaml"),
        ("codec", "train", "configs/validation/codec_pilot.yaml"),
        ("targets", "build", "configs/validation/latent_targets_pilot.yaml"),
        ("generator", "train", "configs/validation/generator_100_step_pilot.yaml"),
        ("preflight", None, "configs/validation/generator_100_step_pilot.yaml"),
    ],
)
def test_direct_stage_overrides(command, action, path):
    from ncm_lab.cli import parser

    args = parser().parse_args(
        [command, *([action] if action else []), "--config", path, "--canvas-size", "8"]
    )
    value = config(args.config, args.canvas_size)
    assert value["architecture"]["canvas"] == 8 and value["architecture"]["latent_width"] == 384
    assert "/canvas/xsmall/" in Path(value["output_ref"]).as_posix()


def test_conflicting_options_and_explicit_config_geometry(tmp_path):
    p = argparse.ArgumentParser()
    pilot.add_arguments(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--fresh", "--resume", "checkpoint.json"])
    with pytest.raises(ValueError, match="--data-config requires --prepare"):
        pilot.run_pilot(100, fresh=True, data_config=tmp_path / "data.json", plan_only=True)
    value = config("configs/validation/generator_100_step_pilot.yaml")
    value["architecture"] = resolve_canvas("xlarge").architecture().to_dict()
    path = tmp_path / "explicit.json"
    write_json(path, value)
    plan = pilot.run_pilot(100, fresh=True, config_path=path, plan_only=True)
    assert plan["canvas"]["positions"] == 1024
    plan = pilot.run_pilot(
        100, fresh=True, config_path=path, canvas_size="small", prepare=True, no_chat=True, plan_only=True
    )
    assert plan["canvas"]["positions"] == 32 and plan["architecture"]["latent_width"] == 384


def test_named_pilot_requires_step_objective_and_disables_explicit_clocks(tmp_path):
    value = config("configs/validation/generator_100_step_pilot.yaml", _bind_canvas=False)
    value["budget"]["enforce_wall_time_limits"] = True
    path = tmp_path / "config.json"
    write_json(path, value)
    result = pilot.run_pilot(100, fresh=True, config_path=path, plan_only=True)
    assert result["clock_policy"]["stages"]["generator"] is False
    value["training"]["objective"] = "tokens"
    write_json(path, value)
    with pytest.raises(ValueError, match="step objective"):
        pilot.run_pilot(100, fresh=True, config_path=path, plan_only=True)


def test_direct_generator_contract_uses_actual_optimizer_settings(tmp_path):
    value = config("configs/validation/generator_100_step_pilot.yaml", _bind_canvas=False)
    value["training"]["learning_rate"] *= 7
    path = tmp_path / "custom-generator.json"
    write_json(path, value)
    direct = config(path, "small")
    explicit = config(path, "small", _bind_canvas=False)
    assert direct["output_ref"] == pipeline_configs(explicit)["generator"]["output_ref"]
    assert direct["training"]["learning_rate"] == value["training"]["learning_rate"]


def test_plan_process_imports_no_model_libraries(tmp_path):
    import subprocess
    import sys

    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'werkzeug'}:
        raise AssertionError('plan imported model/chat library: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ncm_lab.cli import main
assert main(['pilot','train','--steps','100','--fresh','--prepare','--no-chat','--plan-only']) == 0
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout[result.stdout.index("{") :])["canvas"]["positions"] == 64
