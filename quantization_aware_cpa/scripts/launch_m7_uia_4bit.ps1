param([string]$Python = "C:\miniconda\envs\DLG\python.exe")
$ErrorActionPreference = "Stop"
$TaskProject = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TaskOutput = Join-Path $TaskProject "results\m7_uia_4bit_B16"
$TaskRelease = Join-Path $TaskOutput "gpu_release.json"
if (-not (Test-Path -LiteralPath $TaskRelease)) { throw "The scheduler must release the GPU after shuffled LPIPS completes." }
$TaskReleaseValue = Get-Content -LiteralPath $TaskRelease -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $TaskReleaseValue.released) { throw "The GPU is not released." }
$TaskExisting = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-m qacpa.run_m7_uia_4bit' -and $_.ProcessId -ne $PID }
if ($TaskExisting) { throw "An M7 UIA4bit worker or runner is already active." }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$TaskWorker = Start-Process -FilePath $Python -ArgumentList @("-u", "-m", "qacpa.run_m7_uia_4bit", "--phase", "worker") -WorkingDirectory $TaskProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $TaskOutput "worker.stdout.log") -RedirectStandardError (Join-Path $TaskOutput "worker.stderr.log") -PassThru
Write-Output "Started hidden M7 UIA4bit worker PID=$($TaskWorker.Id). Python subprocess integer exit handling verifies all80 completion artifacts."
