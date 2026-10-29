param([ValidateSet('install','lint','test','smoke','prepare-plan')][string]$Task = 'test')
$ErrorActionPreference = 'Stop'
switch ($Task) {
  'install' { uv sync --locked --extra cpu --extra prepare --group dev }
  'lint' { uv run --no-sync ruff check . }
  'test' { uv run --no-sync python -m pytest -q }
  'smoke' { uv run --no-sync ncm validate --config configs/validation/cpu_native.yaml }
  'prepare-plan' { uv run --no-sync ncm prepare --config configs/validation/real_pilot.yaml --plan-only }
}
exit $LASTEXITCODE
