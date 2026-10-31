param(
    [string]$Python = "C:\miniconda\envs\DLG\python.exe",
    [int]$NBatch = 5,
    [int]$NIter = 25000,
    [int]$NLog = 5000,
    [switch]$RunGma4Bit,
    [switch]$RunB32
)

$ErrorActionPreference = "Stop"

# Kept as a compatibility argument so old launch commands fail explicitly.
# The fp32 GMA baseline failed; the user closed the 4-bit GMA experiment in M7.
if ($RunGma4Bit) {
    throw "GMA 4-bit is closed by the M7 protocol: fp32 GMA already failed, so compression and method effects cannot be separated. Run m7_no_cpa_control instead."
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Official = Join-Path $Root "official_cpa"
$env:CPA_DATASETS_ROOT = Join-Path $Root "datasets"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )
    Write-Host ""
    Write-Host "===== $Name ====="
    & $Body
}

function Summarize-Attack {
    param(
        [string]$RootPath,
        [string]$OutputCsv
    )
    & $Python "src\report_m6_attack_results.py" --roots $RootPath --output $OutputCsv
}

function Assert-GmaFp32Range {
    param([string]$SummaryCsv)
    $row = Import-Csv -LiteralPath $SummaryCsv | Where-Object { $_.stage -eq "gi" } | Select-Object -First 1
    if ($null -eq $row) {
        throw "No GMA summary row found in $SummaryCsv"
    }
    $lpips = [double]$row.lpips_mean
    Write-Host ("GMA fp32 B=16 LPIPS={0:N4}" -f $lpips)
    if ($lpips -lt 0.598 -or $lpips -gt 0.639) {
        throw "GMA fp32 B=16 is outside CPA Table 5 reference range 0.598-0.639; skipping 4-bit interpretation."
    }
}

Set-Location $Official

$GmaFp32Root = "exp\imagenet\vgg16\attack\gm\nodef\m6_gma_fp32_B16"
$GmaFp32Summary = "exp\imagenet\vgg16\attack\m6_gma_fp32_B16_summary.csv"
$UiaFp32Root = "exp\imagenet\vgg16\attack\cp_direct\nodef\m6_uia_fp32_B16_fia_uia"
$UiaFp32Summary = "exp\imagenet\vgg16\attack\m6_uia_fp32_B16_fia_summary.csv"

Invoke-Step "Collect fp32 FedSGD gradients, B=16" {
    if (-not (Test-Path "exp\imagenet\vgg16\grads\nodef\16.pickle")) {
        & $Python "src\collect_grads.py" --ds=imagenet --model=vgg16 --defense=nodef --n_batch=$NBatch --batch_sizes=16 --num_workers=0
    } else {
        Write-Host "Using existing exp\imagenet\vgg16\grads\nodef\16.pickle"
    }
}

Invoke-Step "GMA fp32, B=16" {
    & $Python "src\attack.py" --n_log=$NLog --model=vgg16 --ds=imagenet --attack=gm --batch_size=16 --n_batch=$NBatch --lr=0.1 --sch=cosine --tv=0.001 --fl_alg=fedsgd --defense=nodef --n_iter=$NIter --dry_run --fresh_start --disable_pbar --exp_name=m6_gma_fp32_B16
    Summarize-Attack -RootPath $GmaFp32Root -OutputCsv $GmaFp32Summary
    Assert-GmaFp32Range -SummaryCsv $GmaFp32Summary
}

Invoke-Step "Collect fp32 FedAvg/UIA updates, n=16, T=10, conv frozen" {
    & $Python "src\collect_updates.py" --ds=imagenet --model=vgg16 --n_samples_list=16 --n_rounds=$NBatch --batch_size=32 --n_epochs=10 --lr=0.001 --freeze_conv --num_workers=0
}

Invoke-Step "UIA fp32 CPA+FIA, n=16, T=10" {
    & $Python "src\attack.py" --attack=cp --batch_size=16 --n_batch=$NBatch --defense=nodef --ds=imagenet --model=vgg16 --nv=0.13 --l1=5 --T=7.7 --decor=5.3 --fi=1 --tv=0.1 --fi_method=direct --n_sample_fi=16 --fl_alg=fedavg --n_iter=$NIter --n_log=$NLog --n_iter_fi=$NIter --n_log_fi=$NLog --dry_run --fresh_start --disable_pbar --exp_name=m6_uia_fp32_B16_fia
    Summarize-Attack -RootPath $UiaFp32Root -OutputCsv $UiaFp32Summary
}

Write-Host "GMA 4-bit is permanently disabled under the M7 protocol."

if ($RunB32) {
    Invoke-Step "B=32 reminder" {
        Write-Host "B=32 is optional for m6 and should be launched only after B=16 summaries are accepted."
    }
}
