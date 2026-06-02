# Run preflight checks from repo root (Python always; Rust via cargo_test_windows.ps1).
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent

Push-Location $Repo
Write-Host "[verify] export_golden"
python scripts/export_golden.py

Write-Host "[verify] python-legacy pytest"
Push-Location python-legacy
python -m pytest -q
Pop-Location

Write-Host "[verify] cargo (Windows GNU bootstrap)"
& (Join-Path $Repo "scripts\cargo_test_windows.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[verify] all checks passed"
Pop-Location
