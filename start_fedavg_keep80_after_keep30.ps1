$Workspace = $PSScriptRoot
$OldLog = Join-Path $Workspace "run_cora_single_client_keep50_30_10_mb24.log"
$NewLog = Join-Path $Workspace "run_cora_fedavg_single_keep0p80.log"
$WatcherLog = Join-Path $Workspace "start_fedavg_keep80_after_keep30.log"
$StartedFlag = Join-Path $Workspace "run_cora_fedavg_single_keep0p80.started"
$Marker = "[single-rank] keep=0.3 ranking saved"
$Python = "C:\miniconda\envs\DLG\python.exe"

function Write-WatcherLog($Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $WatcherLog -Value "[$stamp] $Message"
}

Write-WatcherLog "watcher started old_log=$OldLog new_log=$NewLog"

while ($true) {
    if (Test-Path -LiteralPath $OldLog) {
        $tail = Get-Content -LiteralPath $OldLog -Tail 300 -ErrorAction SilentlyContinue
        if (($tail -join "`n").Contains($Marker)) {
            if (Test-Path -LiteralPath $StartedFlag) {
                Write-WatcherLog "marker found but start flag already exists; exiting"
                exit 0
            }
            Set-Content -LiteralPath $StartedFlag -Value (Get-Date -Format "o")
            Write-WatcherLog "marker found; launching FedAvg-context keep=0.8"
            $arguments = @(
                "cora_fedavg_single_client_cpa\run_cora_fedavg_single_client_cpa.py",
                "--ds", "imagenet",
                "--model", "vgg16",
                "--num_clients", "3",
                "--rounds", "20",
                "--observe_round", "20",
                "--samples_per_client", "24",
                "--target_client_id", "0",
                "--local_epochs", "1",
                "--keep_ratios", "0.8",
                "--ensemble_size", "8",
                "--noise_seed", "1234",
                "--desparse_sampler", "retained_rescaled",
                "--run_attack",
                "--attack_n_iter", "25000",
                "--attack_n_log", "5000",
                "--n_iter_fi", "25000",
                "--n_log_fi", "5000",
                "--n_sample_fi", "24",
                "--fia_micro_batch_size", "24",
                "--reuse_existing"
            )
            $quoted = @($Python) + $arguments | ForEach-Object {
                '"' + ($_ -replace '"', '\"') + '"'
            }
            $cmd = ($quoted -join " ") + ' > "' + $NewLog + '" 2>&1'
            $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $cmd" -WorkingDirectory $Workspace -WindowStyle Hidden -PassThru
            Write-WatcherLog "launched pid=$($proc.Id)"
            exit 0
        }
    }
    Start-Sleep -Seconds 20
}
