param([ValidateSet('install','lint','test','data-check','smoke','plan','report','prepare-plan')][string]$Task = 'test')
$ErrorActionPreference = 'Stop'
switch ($Task) {
  'install' { uv sync --locked --extra cpu --extra scoring --extra prepare --group dev }
  'lint' { uv run --no-sync ruff check . }
  'test' { uv run --no-sync python -m pytest -q }
  'data-check' { uv run --no-sync ncm data check }
  'smoke' { uv run --no-sync ncm smoke }
  'plan' { uv run --no-sync ncm experiment plan --id ACL-001 --profile spark }
  'prepare-plan' { uv run --no-sync ncm prepare --profile spark --dry-run }
  'report' { uv run --no-sync ncm report build --study acl; if ($LASTEXITCODE -eq 0) { uv run --no-sync ncm report build --study is_journal } }
}
exit $LASTEXITCODE
