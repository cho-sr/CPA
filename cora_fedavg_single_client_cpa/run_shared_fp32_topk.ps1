[CmdletBinding()]
param(
    [string]$Python = "C:\\miniconda\\envs\\DLG\\python.exe",
    [switch]$RunAttack,
    [switch]$ReuseExisting,
    [int]$EnsembleSize = 8,
    [int]$AttackIterations = 25000,
    [int]$FiaIterations = 25000,
    [string]$ControlName = "imagenet_vgg16_round20_target0_spc24_sharedfp32_from_keep50",
    [string]$AttackTag = "sfp32r20c0k50"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceFile = Join-Path $scriptRoot "outputs_v1\fedavg_single_client_updates\imagenet\vgg16\fedavg_single_client_clients3_round20of20_spc24_target0_local1_e920201510be97a8\global_topk\keep_0p5.pickle"
$controlDir = Join-Path $scriptRoot "outputs_v1\shared_fp32_controls\$ControlName\global_topk"
$prepareScript = Join-Path $scriptRoot "prepare_shared_fp32_topk.py"
$runnerScript = Join-Path $scriptRoot "run_cora_fedavg_single_client_cpa.py"

if (-not (Test-Path -LiteralPath $controlDir)) {
    & $Python $prepareScript --source_file $sourceFile --output_dir $controlDir --keep_ratios 1 0.8 0.5 0.3 0.1 --reconstruct_from_checkpoint
    if ($LASTEXITCODE -ne 0) {
        throw "Shared FP32 control preparation failed with exit code $LASTEXITCODE."
    }
}
else {
    Write-Host "[shared-fp32] reuse prepared artifacts: $controlDir"
}

if (-not $RunAttack) {
    Write-Host "[shared-fp32] preparation complete. Re-run with -RunAttack to launch CPA/FIA."
    exit 0
}

$runArgs = @(
    $runnerScript,
    "--prepared_source_dir", $controlDir,
    "--experiment_tag", $AttackTag,
    "--ds", "imagenet",
    "--model", "vgg16",
    "--num_clients", "3",
    "--rounds", "20",
    "--observe_round", "20",
    "--samples_per_client", "24",
    "--target_client_id", "0",
    "--local_epochs", "1",
    "--local_lr", "0.001",
    "--local_batch_size", "24",
    "--micro_batch_size", "4",
    "--keep_ratios", "1", "0.8", "0.5", "0.3", "0.1",
    "--ensemble_size", $EnsembleSize,
    "--noise_seed", "1234",
    "--desparse_sampler", "retained_rescaled",
    "--run_attack",
    "--attack_n_iter", $AttackIterations,
    "--attack_n_log", "5000",
    "--n_iter_fi", $FiaIterations,
    "--n_log_fi", "5000",
    "--n_sample_fi", "24",
    "--fia_micro_batch_size", "24"
)

if ($ReuseExisting) {
    $runArgs += "--reuse_existing"
}

& $Python @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "Shared FP32 CPA/FIA run failed with exit code $LASTEXITCODE."
}
