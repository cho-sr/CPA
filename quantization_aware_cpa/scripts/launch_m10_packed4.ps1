param([string]$Python = "C:\miniconda\envs\DLG\python.exe")
$ErrorActionPreference = "Stop"
$TaskProject = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TaskOutput = Join-Path $TaskProject "results\m10_cpa_fia_packed4_best10"
if (-not (Test-Path -LiteralPath (Join-Path $TaskOutput "input_manifest.json"))) {
    throw "Prepare qacpa.run_m10_packed4 before launching the worker."
}
$TaskExisting = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match '-m qacpa.run_m10_packed4' }
if ($TaskExisting) { throw "A selected10 packed4 worker or runner is already active." }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$TaskWorker = Start-Process -FilePath $Python -ArgumentList @("-u", "-m", "qacpa.run_m10_packed4", "--phase", "worker") -WorkingDirectory $TaskProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $TaskOutput "worker.stdout.log") -RedirectStandardError (Join-Path $TaskOutput "worker.stderr.log") -PassThru
Write-Output "Started selected10 packed4 CPA+FIA worker PID=$($TaskWorker.Id)."
