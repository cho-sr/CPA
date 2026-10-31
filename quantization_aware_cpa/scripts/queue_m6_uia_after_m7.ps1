param(
    [string]$Python = "C:\miniconda\envs\DLG\python.exe",
    [switch]$Worker,
    [int]$PredecessorPid = 0,
    [string]$PredecessorCreatedUtc = ""
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Output = Join-Path $Project "results\m6_uia_fp32_B16_fia"
$QueuePath = Join-Path $Output "queue_status.json"
$ConfigPath = Join-Path $Project "configs\m6_uia_fp32_B16_fia.json"
$ConfigLockPath = Join-Path $Project "configs\m6_uia_fp32_B16_fia.lock.json"
$M7Output = Join-Path $Project "results\m7_no_cpa_control"
New-Item -ItemType Directory -Path $Output -Force | Out-Null

function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Save-Queue([System.Collections.IDictionary]$Value) {
    $Value.updated_at_utc = [DateTime]::UtcNow.ToString("o")
    $TempPath = "$QueuePath.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $TempPath -Encoding UTF8
    Move-Item -LiteralPath $TempPath -Destination $QueuePath -Force
}

function Assert-LockedConfig {
    $TaskLock = Read-Json $ConfigLockPath
    $Digest = (Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Digest -ne $TaskLock.sha256 -or $TaskLock.protocol_id -ne "m6_uia_fp32_B16_fia_v1") {
        throw "UIA fp32 preregistration was changed after locking."
    }
    $TaskConfig = Read-Json $ConfigPath
    if ($TaskConfig.run_tag -ne "m6_uia_fp32_B16_fia" -or $TaskConfig.quantization.bits -ne "fp32" -or $TaskConfig.quantization.enabled -or $TaskConfig.decision.automatic_4bit) {
        throw "Queue accepts only the registered uncompressed UIA fp32 condition."
    }
    return $TaskLock
}

function Assert-M7Complete {
    $TaskConfig = Read-Json $ConfigPath
    $Execution = Read-Json (Join-Path $M7Output "execution_status.json")
    $Report = Read-Json (Join-Path $M7Output "summary.json")
    if ($Execution.state -ne "complete" -or $Report.report_status -ne "complete") {
        throw "M7 did not complete successfully; the queued UIA experiment will not start."
    }
    if ($Report.preregistration_sha256 -ne $TaskConfig.dependency.preregistration_sha256) {
        throw "M7 completion belongs to a different registered protocol."
    }
}

$Registration = Assert-LockedConfig
if (-not $Worker) {
    if (Test-Path -LiteralPath $QueuePath) {
        $Previous = Read-Json $QueuePath
        if ($Previous.state -eq "complete") {
            Write-Output "UIA fp32 queue already completed. Results: $Output"
            return
        }
        if ($Previous.runner_pid) {
            $ExistingRunner = Get-CimInstance Win32_Process -Filter "ProcessId=$($Previous.runner_pid)" -ErrorAction SilentlyContinue
            if ($ExistingRunner -and $ExistingRunner.CommandLine -match '-m qacpa\.run_m6_uia_fp32_fia') {
                Write-Output "UIA fp32 runner is still active (PID=$($Previous.runner_pid)); refusing to duplicate it or truncate its logs."
                return
            }
        }
        if ($Previous.worker_pid) {
            $Existing = Get-CimInstance Win32_Process -Filter "ProcessId=$($Previous.worker_pid)" -ErrorAction SilentlyContinue
            if ($Existing -and $Existing.CommandLine -match 'queue_m6_uia_after_m7\.ps1') {
                Write-Output "UIA fp32 is already queued/running (worker PID=$($Previous.worker_pid))."
                return
            }
        }
    }
    if ($PredecessorPid -eq 0) {
        $PredecessorPid = [int](Read-Json (Join-Path $M7Output "launch.json")).pid
    }
    $Predecessor = Get-CimInstance Win32_Process -Filter "ProcessId=$PredecessorPid" -ErrorAction SilentlyContinue
    if ($Predecessor) {
        if ($Predecessor.CommandLine -notmatch '-m qacpa\.run_m7_controls') {
            throw "Predecessor PID no longer identifies the registered M7 process."
        }
        $PredecessorCreatedUtc = $Predecessor.CreationDate.ToUniversalTime().ToString("o")
    } else {
        Assert-M7Complete
    }
    $QueueArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $PSCommandPath + '"'),
        "-Worker", "-Python", ('"' + $Python + '"'), "-PredecessorPid", $PredecessorPid
    )
    if ($PredecessorCreatedUtc) {
        $QueueArguments += @("-PredecessorCreatedUtc", $PredecessorCreatedUtc)
    }
    $Queued = Start-Process -FilePath "powershell.exe" -ArgumentList $QueueArguments -WorkingDirectory $Project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output "queue.stdout.log") -RedirectStandardError (Join-Path $Output "queue.stderr.log") -PassThru
    Write-Output "UIA fp32 successor queued: worker PID=$($Queued.Id), predecessor PID=$PredecessorPid. No GPU work starts before M7 finishes."
    return
}

$QueueGuard = [System.IO.File]::Open((Join-Path $Output ".queue.lock"), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$QueueState = [ordered]@{
    state = "waiting_for_m7"
    worker_pid = $PID
    predecessor_pid = $PredecessorPid
    predecessor_created_at_utc = $PredecessorCreatedUtc
    queued_at_utc = [DateTime]::UtcNow.ToString("o")
    run_tag = "m6_uia_fp32_B16_fia"
    config_sha256 = $Registration.sha256
    command = "python -u -m qacpa.run_m6_uia_fp32_fia --phase run"
    automatic_4bit = $false
}
try {
    Save-Queue $QueueState
    $Predecessor = Get-CimInstance Win32_Process -Filter "ProcessId=$PredecessorPid" -ErrorAction SilentlyContinue
    if ($Predecessor) {
        if ($Predecessor.CommandLine -notmatch '-m qacpa\.run_m7_controls' -or $Predecessor.CreationDate.ToUniversalTime().ToString("o") -ne $PredecessorCreatedUtc) {
            throw "Predecessor process identity changed; refusing to wait on or replace it."
        }
        Write-Output "Waiting for the existing M7 process to exit (OS process wait; no metric polling)."
        $ProcessToWait = $null
        try {
            $ProcessToWait = [System.Diagnostics.Process]::GetProcessById($PredecessorPid)
        } catch [System.ArgumentException] {
            # It exited between the identity check and acquiring its handle.
            # The completion markers below must still validate successfully.
        }
        if ($null -ne $ProcessToWait) {
            try { $ProcessToWait.WaitForExit() } finally { $ProcessToWait.Dispose() }
        }
    }
    Assert-M7Complete
    $Registration = Assert-LockedConfig
    if ($Registration.sha256 -ne $QueueState.config_sha256) {
        throw "UIA registration changed while waiting for M7."
    }
    $QueueState.state = "starting_uia_fp32"
    $QueueState.predecessor_completed_at_utc = [DateTime]::UtcNow.ToString("o")
    Save-Queue $QueueState
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONUTF8 = "1"
    $Runner = Start-Process -FilePath $Python -ArgumentList @("-u", "-m", "qacpa.run_m6_uia_fp32_fia", "--phase", "run") -WorkingDirectory $Project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output "run.stdout.log") -RedirectStandardError (Join-Path $Output "run.stderr.log") -PassThru
    $QueueState.state = "running_uia_fp32"
    $QueueState.runner_pid = $Runner.Id
    $QueueState.runner_created_at_utc = $Runner.StartTime.ToUniversalTime().ToString("o")
    Save-Queue $QueueState
    $Runner.WaitForExit()
    $Runner.Refresh()
    if ($Runner.ExitCode -ne 0) {
        throw "UIA fp32 runner failed (exit code $($Runner.ExitCode)); inspect run.stderr.log. No later experiment will run."
    }
    $Result = Read-Json (Join-Path $Output "summary.json")
    if ($Result.status -ne "complete") {
        throw "UIA fp32 runner exited without a complete summary."
    }
    if ($Result.config_sha256 -ne $QueueState.config_sha256) {
        throw "UIA fp32 completion belongs to a different registered config."
    }
    $QueueState.state = "complete"
    $QueueState.completed_at_utc = [DateTime]::UtcNow.ToString("o")
    Save-Queue $QueueState
    Write-Output "UIA fp32 condition complete. No 4-bit or GMA successor is scheduled."
} catch {
    $QueueState.state = "failed"
    $QueueState.error = $_.Exception.Message
    Save-Queue $QueueState
    Write-Error $_
    exit 1
} finally {
    $QueueGuard.Dispose()
}
