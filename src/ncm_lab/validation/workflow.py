"""Bounded connected engineering ladder. Larger 10M/100M runs are never invoked."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from ncm_lab.validation.fixtures import model_spec, fixture_data
from ncm_lab.config.architecture import tiny_architecture
from ncm_lab.runtime.io import atomic_json, reference, read, digest, root


def validate(run, value):
    from ncm_lab.runtime.runner import execute

    architecture = tiny_architecture()
    data = fixture_data(run.path / "artifacts/data", architecture, run.meter)
    handoffs = run.path / "handoffs"
    handoffs.mkdir()
    atomic_json(handoffs / "data.json", reference(data))
    common = {
        **value,
        "model": model_spec(),
        "architecture": architecture.to_dict(),
        "inputs": {},
        "training": {
            **value["training"],
            "steps": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "new_learning_rate": 0.002,
            "max_validation_examples": 2,
            "checkpoint_interval": 1,
            "validation_interval": 1,
            "warmup_processed_tokens": 0,
        },
        "run_root": str(root() / "runs"),
    }
    stages = []

    def stage(name, inputs, **overrides):
        current = {
            **copy.deepcopy(common),
            "stage": name,
            "inputs": {key: str(path) for key, path in inputs.items()},
            "output_ref": str(handoffs / (name + ".json")),
        }
        current.update(overrides)
        result = execute(current)
        stages.append(result)
        if result["status"] != "completed":
            raise ValueError("native fixture stage failed: " + json.dumps(result))
        return current, result

    stage("teacher", {"data": handoffs / "data.json"})
    codec_config, codec_result = stage("codec", {"features": handoffs / "teacher.json"})
    # Explicit one-step extension under the unchanged aggregate budget and contract.
    extended = {**codec_config, "training": {**codec_config["training"], "steps": 3}}
    resumed = execute(extended, codec_result["checkpoint"])
    stages.append(resumed)
    if resumed["status"] != "completed":
        raise ValueError("codec fixture recovery failed: " + str(resumed))
    stage(
        "targets",
        {
            "data": handoffs / "data.json",
            "codec": handoffs / "codec.json",
            "features": handoffs / "teacher.json",
        },
    )
    gen_config, gen_result = stage("generator", {"targets": handoffs / "targets.json"})
    extended = {**gen_config, "training": {**gen_config["training"], "steps": 3}}
    resumed_gen = execute(extended, gen_result["checkpoint"])
    stages.append(resumed_gen)
    if resumed_gen["status"] != "completed":
        raise ValueError("generator fixture recovery failed: " + str(resumed_gen))
    # Micro-overfit is a separately recorded diagnostic with fixed predeclared caps.
    micro_steps = value["data"].get("micro_overfit_steps", 40)
    if not 1 <= micro_steps <= 200:
        raise ValueError("micro-overfit step cap must be within 1..200")
    micro = []
    for name, inputs in [
        ("codec", {"features": handoffs / "teacher.json"}),
        ("generator", {"targets": handoffs / "targets.json"}),
    ]:
        cfg = {
            **copy.deepcopy(common),
            "stage": name,
            "inputs": {k: str(p) for k, p in inputs.items()},
            "training": {
                **common["training"],
                "steps": micro_steps,
                "validation_interval": 10,
                "checkpoint_interval": 10,
                "batch_size": 8,
            },
            "output_ref": None,
        }
        result = execute(cfg)
        stages.append(result)
        metrics = [
            json.loads(line)
            for line in (Path(result["path"]) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        micro.append(
            {
                "stage": name,
                "run_id": result["run_id"],
                "status": result["status"],
                "first": metrics[0],
                "last": metrics[-1],
                "observed_loss_decrease": metrics[-1]["loss"] < metrics[0]["loss"],
                "meaning": "fixed-batch software debugging; no generalization or real-8B evidence",
            }
        )
        if result["status"] != "completed":
            raise ValueError("bounded micro-overfit failed; inspect immutable run " + result["path"])
    export = Path(resumed_gen["artifact"])
    prompt = run.path / "artifacts/prompt.txt"
    prompt.write_text("token4", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "ncm_lab",
        "generate",
        "--checkpoint",
        str(export),
        "--prompt-file",
        str(prompt),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=min(60, value["budget"]["max_wall_seconds"]),
        check=True,
        cwd=root(),
    )
    generation = json.loads(completed.stdout)
    from ncm_lab.runtime.io import write_json

    write_json(run.path / "artifacts/fresh_process_generation.json", generation, exclusive=True)
    return {
        "stages": [
            {
                "run_id": r["run_id"],
                "stage": read(Path(r["path"]) / "resolved_config.json")["stage"],
                "path": r["path"],
                "status": r["status"],
            }
            for r in stages
        ],
        "micro_overfit": micro,
        "fresh_process_generation": generation,
        "data_identity": digest(data),
        "automatic_large_training": False,
        "real_data_codec": "not executed",
        "real_8b_full_finetuning": "not executed",
        "quality_or_speed_claim": False,
    }
