param(
    [string]$Python = "C:\miniconda\envs\DLG\python.exe"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Output = Join-Path $Project "results\m7_no_cpa_control"
$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '-m qacpa\.run_m7_controls'
}
if ($Existing) {
    throw "M7 is already running (PID $($Existing.ProcessId -join ', ')); refusing a duplicate queue."
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$Child = Start-Process -FilePath $Python -ArgumentList @("-u", "-m", "qacpa.run_m7_controls", "--phase", "all") -WorkingDirectory $Project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output "run.stdout.log") -RedirectStandardError (Join-Path $Output "run.stderr.log") -PassThru
[ordered]@{
    pid = $Child.Id
    started_at_utc = [DateTime]::UtcNow.ToString("o")
    command = "python -u -m qacpa.run_m7_controls --phase all"
    order = @("no-CPA fp32 (40)", "no-CPA 4-bit (40)", "prior-only (40)", "report")
    stdout = (Join-Path $Output "run.stdout.log")
    stderr = (Join-Path $Output "run.stderr.log")
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "launch.json") -Encoding utf8
Write-Output "M7 started: PID=$($Child.Id). Logs: $Output"
