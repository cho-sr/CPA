$ErrorActionPreference = "Continue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imagenetRoot = Join-Path $repoRoot "datasets\imagenet"
$logPath = Join-Path $imagenetRoot "imagenet_bits_watcher.log"

New-Item -ItemType Directory -Force -Path $imagenetRoot | Out-Null

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] $Message"
}

Write-Log "Watcher started."

while ($true) {
    $jobs = @(Get-BitsTransfer -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -like "ImageNet *" })

    if ($jobs.Count -eq 0) {
        Write-Log "No ImageNet BITS jobs remain. Watcher exiting."
        break
    }

    foreach ($job in $jobs) {
        $totalGb = if ($job.BytesTotal -gt 0 -and $job.BytesTotal -lt [uint64]::MaxValue) {
            [math]::Round($job.BytesTotal / 1GB, 2)
        } else {
            -1
        }
        $doneGb = [math]::Round($job.BytesTransferred / 1GB, 2)
        Write-Log "$($job.DisplayName): $($job.JobState) $doneGb/$totalGb GB"

        if ($job.JobState -eq "Transferred") {
            try {
                Complete-BitsTransfer -BitsJob $job
                Write-Log "$($job.DisplayName): completed."
            } catch {
                Write-Log "$($job.DisplayName): complete failed: $($_.Exception.Message)"
            }
        } elseif ($job.JobState -eq "TransientError") {
            try {
                Resume-BitsTransfer -BitsJob $job
                Write-Log "$($job.DisplayName): resumed after transient error."
            } catch {
                Write-Log "$($job.DisplayName): resume failed: $($_.Exception.Message)"
            }
        } elseif ($job.JobState -eq "Error") {
            Write-Log "$($job.DisplayName): error: $($job.ErrorDescription)"
        }
    }

    Start-Sleep -Seconds 60
}
