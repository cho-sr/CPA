$ErrorActionPreference = "Continue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imagenetRoot = Join-Path $repoRoot "datasets\imagenet"
$runLog = Join-Path $repoRoot "quantized_update_cpa\run_01_imagenet_cpa64_fia64_25k.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $runLog -Value $line
}

function Count-Dirs {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Path -Directory -Filter "n*" -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Count-Files {
    param(
        [string]$Path,
        [string]$Filter
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Path -File -Filter $Filter -ErrorAction SilentlyContinue | Measure-Object).Count
}

$trainRoot = Join-Path $imagenetRoot "train"
$valRoot = Join-Path $imagenetRoot "val"

Write-Log "Waiting for ImageNet train/val ImageFolder layout."
while ($true) {
    $trainClasses = Count-Dirs -Path $trainRoot
    $valClasses = Count-Dirs -Path $valRoot
    $trainTars = Count-Files -Path $trainRoot -Filter "n*.tar"
    Write-Log "ImageNet status: train_classes=$trainClasses val_classes=$valClasses train_tars_remaining=$trainTars"
    if ($trainClasses -ge 1000 -and $valClasses -ge 1000 -and $trainTars -eq 0) {
        break
    }
    Start-Sleep -Seconds 60
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$env:PYTHONPATH = ""
$env:WANDB_MODE = "offline"
$argsList = @(
    "quantized_update_cpa/01_fp32_update_cpa.py",
    "--ds", "imagenet",
    "--model", "vgg16",
    "--n_samples", "64",
    "--attack_trials", "10",
    "--local_epochs", "10",
    "--local_lr", "1e-3",
    "--seed", "42",
    "--run_attack",
    "--attack_n_iter", "25000",
    "--n_iter_fi", "25000",
    "--n_sample_fi", "64"
)

if ($env:CPA_FRESH_START -eq "1") {
    Write-Log "Fresh start requested for author run output."
    $argsList += "--fresh_start"
}

Write-Log "Starting FP32 author CPA64+FIA64 25k/25k without micro-batching."
Write-Log "$python $($argsList -join ' ')"
& $python @argsList *>> $runLog
$exitCode = $LASTEXITCODE
Write-Log "Run finished with exit code $exitCode."
exit $exitCode
