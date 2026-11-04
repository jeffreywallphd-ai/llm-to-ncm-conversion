"""Literal README Bash workflow: syntax, real tiny backend, shell failure isolation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from ncm_lab.runtime.io import root, read
from ncm_lab.stage1.cli import parser


def block():
    text = (root() / "README.md").read_text(encoding="utf-8")
    section = text.split("<!-- BEGIN REQUIRED README STAGE1 BLOCK -->", 1)[1].split(
        "<!-- END REQUIRED README STAGE1 BLOCK -->", 1
    )[0]
    return section.split("```bash", 1)[1].split("```", 1)[0].strip()


def bash():
    candidate = (
        Path("C:/Program Files/Git/bin/bash.exe")
        if os.name == "nt"
        else Path(shutil.which("bash") or "/missing")
    )
    if not candidate.is_file():
        pytest.skip("Bash unavailable; literal shell check cannot execute on this host")
    return str(candidate)


def shell_environment(tmp_path, stage1_fixture=None, failure=None):
    repo = tmp_path / "repository with spaces"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".ncm-lab").touch()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True, timeout=10)
    for name in ("run_stage1", "train_stage1_codec", "train_stage1_generator", "evaluate_stage1"):
        shutil.copyfile(root() / "scripts" / (name + ".py"), repo / "scripts" / (name + ".py"))
    shim = tmp_path / "test-only-python-shim"
    shim.mkdir()
    launcher = shim / "python"
    launcher.write_text(
        """#!/usr/bin/env bash
args=("$@")
script=""
new_plan=0
requested=""
for ((i=0;i<${#args[@]};i++)); do
  case "${args[$i]}" in
    scripts/*.py) script="${args[$i]}" ;;
    --experiment-dir) new_plan=1 ;;
    --stage) requested="${args[$((i+1))]}" ;;
  esac
done
if [[ "$new_plan" == 1 && -n "${STAGE1_FIXTURE_CONFIG:-}" ]]; then
  args+=(--config "$STAGE1_FIXTURE_CONFIG")
fi
{ printf '%s\\0' "${args[@]}"; printf '\\0'; } >> "$STAGE1_SHELL_LOG"
if [[ -n "$script" && -n "${STAGE1_SHELL_FAILURE:-}" ]]; then
  case "$script" in
    *train_stage1_codec.py) requested="codec" ;;
    *train_stage1_generator.py) requested="generator" ;;
    *evaluate_stage1.py) requested="evaluate" ;;
  esac
  if [[ "$requested" == "$STAGE1_SHELL_FAILURE" ]]; then
    printf 'explicit shell-only failure/review fixture: %s\\n' "$requested" >&2
    exit 2
  fi
  # Shell/order fixture only. The real tiny-backend test below does not use this branch.
  exit 0
fi
exec "$STAGE1_REAL_PYTHON" "${args[@]}"
""",
        encoding="utf-8",
        newline="\n",
    )
    launcher.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(shim) + os.pathsep + os.environ["PATH"],
        "STAGE1_REAL_PYTHON": sys.executable.replace("\\", "/"),
        "STAGE1_SHELL_LOG": str(tmp_path / "commands.bin").replace("\\", "/"),
        "NCM_REPO_ROOT": str(repo),
    }
    if stage1_fixture:
        env["STAGE1_FIXTURE_CONFIG"] = str(stage1_fixture["config"]).replace("\\", "/")
    if failure:
        env["STAGE1_SHELL_FAILURE"] = failure
    (repo / "nested working directory").mkdir()
    script = (
        'printf "%s\\n%s\\n" "$PWD" "$SHELLOPTS" > ../before.txt\n'
        + block()
        + '\nresult=$?\nprintf "%s\\n%s\\n" "$PWD" "$SHELLOPTS" > ../after.txt\nprintf "PARENT_SHELL_ALIVE\\n"\nexit "$result"\n'
    )
    return repo, env, script


def commands(tmp_path):
    return [
        [arg.decode("utf-8") for arg in row.split(b"\0")]
        for row in (tmp_path / "commands.bin").read_bytes().split(b"\0\0")
        if row
    ]


def test_exact_readme_block_bash_syntax_and_parsers():
    result = subprocess.run(
        [bash(), "--noprofile", "--norc", "-n"], input=block(), text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    import shlex

    parsed = []
    for line in block().splitlines():
        if line.strip().startswith("python -u scripts/"):
            tokens = shlex.split(line.strip())
            tokens = ["medium" if token == "$CANVAS_SIZE" else token for token in tokens]
            name = tokens[2]
            parsed.append(
                parser(
                    "codec"
                    if "train_stage1_codec" in name
                    else "generator"
                    if "train_stage1_generator" in name
                    else "evaluate"
                    if "evaluate_stage1" in name
                    else None
                ).parse_args(tokens[3:])
            )
    assert len(parsed) == 7 and parsed[0].plan_only
    assert all(p.experiment == Path("$MANIFEST") for p in parsed[1:])
    assert [p.stage for p in parsed if p.stage] == ["prepare", "teacher", "targets"]


def test_literal_bash_block_runs_actual_tiny_backend_and_keeps_parent_shell(tmp_path, stage1_fixture):
    repo, env, script = shell_environment(tmp_path, stage1_fixture=stage1_fixture)
    result = subprocess.run(
        [bash(), "--noprofile", "--norc", "-c", script],
        cwd=repo / "nested working directory",
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Stage 1 complete." in result.stdout and "PARENT_SHELL_ALIVE" in result.stdout
    assert (repo / "before.txt").read_bytes() == (repo / "after.txt").read_bytes()
    manifests = list((repo / "runs/stage1/medium").glob("*/manifest.json"))
    assert len(manifests) == 1
    state = read(manifests[0])
    assert state["status"] == "completed"
    report = read(manifests[0].parent / "reports/stage1_summary.json")
    assert report["origin"] == "software_fixture" and report["targets"] == {"codec": 12, "generator": 3}
    assert report["settings"]["canvas"]["positions"] == 64
    calls = commands(tmp_path)
    actual = [c for c in calls if any(a.startswith("scripts/") for a in c)]
    assert len(actual) == 7
    refs = [c[c.index("--experiment") + 1] for c in actual if "--experiment" in c]
    assert len(set(refs)) == 1
    assert (
        "--config" in actual[0]
    )  # Explicit test-only fixture override, not a production environment fallback.


@pytest.mark.parametrize("failure", ["prepare", "teacher", "codec", "targets", "generator", "evaluate"])
def test_literal_block_failure_stops_remaining_commands_without_closing_parent(tmp_path, failure):
    repo, env, script = shell_environment(tmp_path, failure=failure)
    result = subprocess.run(
        [bash(), "--noprofile", "--norc", "-c", script],
        cwd=repo / "nested working directory",
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Stage 1 complete." not in result.stdout and "PARENT_SHELL_ALIVE" in result.stdout
    assert (repo / "before.txt").read_bytes() == (repo / "after.txt").read_bytes()
    calls = [c for c in commands(tmp_path) if any(a.startswith("scripts/") for a in c)]
    order = ["prepare", "teacher", "codec", "targets", "generator", "evaluate"]
    assert len(calls) == 2 + order.index(failure)
