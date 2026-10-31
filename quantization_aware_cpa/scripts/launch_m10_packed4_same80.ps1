param(
    [string]$Python = "C:\miniconda\envs\DLG\python.exe",
    [ValidateSet("prepare", "packet", "run", "report")]
    [string]$Phase = "run"
)
$ErrorActionPreference = "Stop"
$TaskProject = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TaskConfig = Join-Path $TaskProject "configs\m10_cpa_fia_packed4_same80_B16.json"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
Push-Location -LiteralPath $TaskProject
try {
    & $Python -u -m qacpa.run_m10_packed4_same80 --config $TaskConfig --phase $Phase
    if ($LASTEXITCODE -ne 0) { throw "same80 packed4 runner failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
