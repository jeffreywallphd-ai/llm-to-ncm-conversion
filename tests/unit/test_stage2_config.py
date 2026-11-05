import json
import subprocess
import sys

import pytest

from ncm_lab.runtime.io import root
from ncm_lab.stage1.config import resolve


@pytest.mark.parametrize("source,format_", [("oasst", "oasst_parquet"), ("tulu", "tulu_parquet")])
def test_stage2_production_plan_is_offline_and_frozen(source, format_):
    launcher = root() / "scripts/run_stage2.py"
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; sys.modules.update(torch=None,transformers=None,pyarrow=None,werkzeug=None); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
            str(launcher),
            "--plan-only",
            "--source",
            source,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(process.stdout)
    plan = result["plan"]
    assert plan["schema"] == "stage2_plan_v1" and not plan["research_work_started"]
    settings = plan["scientific_settings"]
    assert settings["canvas"]["positions"] == 64
    assert settings["qualification"]["train_examples"] == 16000
    assert settings["qualification"]["dev_examples"] == settings["qualification"]["test_examples"] == 200
    stages = settings["stages"]
    assert stages["prepare"]["data"]["format"] == format_
    for stage, expected in {"codec": (1000, 4, 4), "generator": (4000, 1, 8)}.items():
        training = stages[stage]["training"]
        assert tuple(training[field] for field in ("steps", "batch_size", "accumulation")) == expected
    assert all(not stage["budget"]["enforce_wall_time_limits"] for stage in stages.values())
    if source == "tulu":
        assert stages["prepare"]["data"]["max_source_rows"] == 400_000
        assert stages["prepare"]["data"]["max_answer_tokens"] == 64


def test_stage2_rejects_changed_production_canvas_and_steps():
    config = root() / "configs/stage2/experiment.yaml"
    with pytest.raises(ValueError, match="64-position"):
        resolve(config, canvas_size="xsmall")
    with pytest.raises(ValueError, match="4,000 generator"):
        resolve(config, generator_steps=2000)
