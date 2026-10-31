param(
    [int]$TargetPid = 20888,
    [string]$LogPath = ".\run_cora_single_client_keep50_30_10_mb24.log",
    [string]$WatcherLog = ".\stop_after_keep0p3_watcher.log"
)

$ErrorActionPreference = "Continue"
$resolvedLog = (Resolve-Path -LiteralPath $LogPath).Path
$workspace = (Resolve-Path -LiteralPath ".").Path

function Write-WatcherLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $WatcherLog -Value "[$stamp] $Message"
}

if (-not $resolvedLog.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Log path is outside workspace: $resolvedLog"
}

Write-WatcherLog "watcher started target_pid=$TargetPid log=$resolvedLog"

while ($true) {
    $process = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Write-WatcherLog "target process already exited"
        break
    }

    $tail = Get-Content -LiteralPath $resolvedLog -Tail 120 -ErrorAction SilentlyContinue
    $joined = $tail -join "`n"

    if ($joined -match "\[single-rank\] keep=0\.3 ranking saved") {
        Write-WatcherLog "keep 0.3 ranking saved; stopping target"
        Stop-Process -Id $TargetPid -Force
        break
    }

    if ($joined -match "keep=0\.1") {
        Write-WatcherLog "keep 0.1 was about to start or started; stopping target"
        Stop-Process -Id $TargetPid -Force
        break
    }

    Start-Sleep -Seconds 2
}
