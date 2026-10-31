param([string]$Python = "C:\miniconda\envs\DLG\python.exe")
$ErrorActionPreference = "Stop"
$TaskProject = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TaskOutput = Join-Path $TaskProject "results\m8_fedavg_naive_cpa_dither4_B16"
if (-not (Test-Path -LiteralPath (Join-Path $TaskOutput "input_manifest.json"))) {
    throw "Run python -m qacpa.run_m8_fedavg_naive --phase prepare first."
}
$TaskExisting = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-m qacpa.run_m8_fedavg_naive' -and $_.ProcessId -ne $PID }
if ($TaskExisting) { throw "A FedAvg naive worker or runner is already active." }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$TaskWorker = Start-Process -FilePath $Python -ArgumentList @("-u", "-m", "qacpa.run_m8_fedavg_naive", "--phase", "worker") -WorkingDirectory $TaskProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $TaskOutput "worker.stdout.log") -RedirectStandardError (Join-Path $TaskOutput "worker.stderr.log") -PassThru
Write-Output "Started hidden direct-upstream FedAvg naive worker PID=$($TaskWorker.Id)."
