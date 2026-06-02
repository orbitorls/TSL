# Print the WSL command for full local GPU training (run from repo root or anywhere).
$Repo = if ($env:TSL_REPO_ROOT) { $env:TSL_REPO_ROOT } else { "D:\TSL" }
$WslRepo = ($Repo -replace '\\', '/') -replace '^([A-Za-z]):', '/mnt/$1' -replace ':', ''
$WslRepo = $WslRepo.ToLower()

$cmd = @"
wsl bash -lc 'export TSL_REPO_ROOT=$WslRepo; bash $WslRepo/scripts/train_local_wsl.sh'
"@

Write-Host "Run this in PowerShell to start WSL training:"
Write-Host $cmd
Write-Host ""
Write-Host "Smoke test (512 TSL-51 samples, 2 epochs):"
Write-Host "wsl bash -lc 'cd $WslRepo/python-legacy && source ~/venvs/tsl/bin/activate && python scripts/train_local_all.py --tracks both --max-tsl51-samples 512 --tsl51-epochs 2'"
