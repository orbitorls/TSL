# run_full_pipeline.ps1 — Full TSL-51 dataset expansion + training pipeline
#
# STRATEGY A: Retrain on full HF dataset (44k+ samples) — removes the 12k cap
# STRATEGY B: Fine-tune with YouTube clips — reduces domain gap on real-world clips
#
# Run from repo root:
#   powershell -File scripts\run_full_pipeline.ps1
#   powershell -File scripts\run_full_pipeline.ps1 -Strategy A   (HF only)
#   powershell -File scripts\run_full_pipeline.ps1 -Strategy B   (YouTube fine-tune only)
#   powershell -File scripts\run_full_pipeline.ps1 -Strategy AB  (both, default)

param(
    [string]$Strategy = "AB",
    [int]$Epochs = 40,
    [int]$BatchSize = 32,
    [string]$PythonExe = ".venv-train\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$BaseArtifacts  = "artifacts\tsl51"
$YtNpz          = "data\external_tsl51\yt_clips.npz"
$YtManifest     = "data\external_tsl51\manifest.csv"
$YtLabels       = "artifacts\tsl51\tsl51_labels.json"

# ── helpers ───────────────────────────────────────────────────────────────────
function Run($desc, $cmd) {
    Write-Host "`n=== $desc ===" -ForegroundColor Cyan
    Write-Host $cmd -ForegroundColor Gray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}

# ── Strategy A: full HF dataset ───────────────────────────────────────────────
if ($Strategy -match "A") {
    Write-Host "`n[Strategy A] Retrain on full HF dataset (44,561 samples, no cap)" -ForegroundColor Yellow
    Write-Host "NOTE: This downloads ~44k landmark CSVs from HuggingFace on first run." -ForegroundColor Gray
    Write-Host "      Expected: 4-8 hours on GPU (WSL2 recommended)." -ForegroundColor Gray

    $cmd = @"
$PythonExe python-legacy\scripts\train_local_all.py ``
  --tracks tsl51 ``
  --tsl51-epochs $Epochs ``
  --batch-size $BatchSize ``
  --artifact-dir artifacts\tsl51-fullset
"@
    # Note: no --max-tsl51-samples flag = use all 44k+ samples
    Run "Train on full HF dataset (Strategy A)" $cmd

    Write-Host "`n[Strategy A] Done! Artifacts in artifacts\tsl51-fullset\" -ForegroundColor Green
    Write-Host "To use this model, copy artifacts to artifacts\tsl51\" -ForegroundColor Gray
    Write-Host "  Copy-Item artifacts\tsl51-fullset\* artifacts\tsl51\" -ForegroundColor Gray
}

# ── Strategy B: YouTube clips fine-tune ────────────────────────────────────────
if ($Strategy -match "B") {
    Write-Host "`n[Strategy B] Fine-tune with YouTube clips" -ForegroundColor Yellow

    # Step B1: Collect YouTube clips (if not already done)
    if (-not (Test-Path "data\external_tsl51\videos") -or
        (Get-ChildItem "data\external_tsl51\videos\*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count -lt 10) {
        Write-Host "  Collecting YouTube clips for all 47 signs..." -ForegroundColor Gray
        Run "Collect YouTube clips" "$PythonExe scripts\collect_yt_signs.py --labels $YtLabels --output-dir data\external_tsl51\videos --manifest-out $YtManifest --results-per-sign 3 --max-duration 30"
    } else {
        Write-Host "  YouTube clips already downloaded, skipping collection." -ForegroundColor Gray
    }

    # Step B2: Build external NPZ cache
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    $env:TF_ENABLE_ONEDNN_OPTS = "0"
    Run "Build external NPZ cache" "$PythonExe scripts\build_external_dataset.py --manifest $YtManifest --track tsl51 --labels $YtLabels --out $YtNpz --target-fps 15 --min-detected-frames 8"

    # Verify NPZ — allow_pickle=True is safe here: this file was just written by
    # build_external_dataset.py (our own code) using np.savez_compressed. The only
    # pickled object is the class_names string array, not executable code.
    $shapeCheck = & $PythonExe -c "import numpy as np; d=np.load('$YtNpz',allow_pickle=True); print(f'NPZ OK: {d[chr(88)].shape[0]} samples')"
    Write-Host "  $shapeCheck" -ForegroundColor Green

    # Step B3: Find the best base cache
    $baseCaches = @(
        ".tools\train_runs_25690531-175818\tsl51_e40_b32_s12000\work\features\tsl51_features.npz",
        "artifacts\tsl51-fullset\tsl51_features.npz"  # if Strategy A ran first
    )
    $baseCache = $baseCaches | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $baseCache) {
        Write-Host "[WARN] No base cache found. Fine-tuning without base HF data." -ForegroundColor Yellow
        $baseCache = $YtNpz  # fallback: use YouTube-only data
    }
    Write-Host "  Using base cache: $baseCache" -ForegroundColor Gray

    # Step B4: Fine-tune
    $outDir = ".tools\yt_augmented_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Run "Fine-tune model (Strategy B)" "$PythonExe scripts\train_external_augmented.py --track tsl51 --base-cache $baseCache --external-cache $YtNpz --out-dir $outDir --base-artifact-dir $BaseArtifacts --epochs 20 --batch-size 32"

    Write-Host "`n[Strategy B] Done! New artifacts in $outDir\artifacts\tsl51\" -ForegroundColor Green

    # Step B5: Evaluate on test clips
    if (Test-Path "reports\external_tsl51_eval\short_word_samples.csv") {
        $evalOut = "reports\external_tsl51_eval\yt_augmented"
        Run "Evaluate on test clips" "$PythonExe scripts\evaluate_tsl51_video.py --samples reports\external_tsl51_eval\short_word_samples.csv --artifact-dir $outDir\artifacts\tsl51 --strategy uniform --target-fps 30 --out-dir $evalOut"
        Write-Host "`n[Eval] Results in $evalOut\summary.json" -ForegroundColor Green
    }

    # Copy to canonical artifacts path
    Write-Host "`nTo deploy the new model:" -ForegroundColor Yellow
    Write-Host "  Copy-Item $outDir\artifacts\tsl51\* artifacts\tsl51\" -ForegroundColor Gray
}

Write-Host "`n[Done] Pipeline complete." -ForegroundColor Green
