# Run TSL translate API + Next.js frontend (two windows).
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Legacy = Join-Path $Repo "python-legacy"
$Web = Join-Path $Repo "web\translate"

Write-Host "Starting TSL Translate API on http://127.0.0.1:8000 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Legacy'; python '$Repo\scripts\tsl_translate_api.py'"
)

if (-not (Test-Path (Join-Path $Web "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Set-Location $Web
    npm install
    Set-Location $Repo
}

Write-Host "Starting Next.js on http://localhost:3000 ..."
Set-Location $Web
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
npm run dev
