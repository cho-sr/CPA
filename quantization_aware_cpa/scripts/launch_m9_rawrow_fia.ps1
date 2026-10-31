param([string]$Python = "C:\miniconda\envs\DLG\python.exe")
$ErrorActionPreference = "Stop"
$TaskProject = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TaskOutput = Join-Path $TaskProject "results\m9_fedavg_rawrow_dither4_B16"
if (-not (Test-Path -LiteralPath (Join-Path $TaskOutput "input_manifest.json"))) {
    throw "Prepare qacpa.run_m9_rawrow_fia before launching the worker."
}
$TaskExisting = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-m qacpa.run_m9_rawrow_fia' -and $_.ProcessId -ne $PID }
if ($TaskExisting) { throw "An M9 raw-row worker or runner is already active." }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$TaskWorker = Start-Process -FilePath $Python -ArgumentList @("-u", "-m", "qacpa.run_m9_rawrow_fia", "--phase", "worker") -WorkingDirectory $TaskProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $TaskOutput "worker.stdout.log") -RedirectStandardError (Join-Path $TaskOutput "worker.stderr.log") -PassThru
Write-Output "Started hidden raw-row + FIA worker PID=$($TaskWorker.Id)."
