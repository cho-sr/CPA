$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$imagenetRoot = Resolve-Path (Join-Path $repoRoot "datasets\imagenet")
$logPath = Join-Path $imagenetRoot "prepare_imagenet.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

function Assert-Within {
    param(
        [string]$Path,
        [string]$Root
    )
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root)
    if (-not $resolvedPath.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside root: $resolvedPath"
    }
}

function Remove-WithRetry {
    param([string]$Path)
    for ($attempt = 1; $attempt -le 10; $attempt += 1) {
        try {
            Remove-Item -LiteralPath $Path -Force
            return
        } catch {
            if ($attempt -eq 10) {
                throw
            }
            Start-Sleep -Seconds 3
        }
    }
}

$trainTar = Join-Path $imagenetRoot "ILSVRC2012_img_train.tar"
$valTar = Join-Path $imagenetRoot "ILSVRC2012_img_val.tar"
$devkitTar = Join-Path $imagenetRoot "ILSVRC2012_devkit_t12.tar.gz"
$devkitRoot = Join-Path $imagenetRoot "ILSVRC2012_devkit_t12"
$trainRoot = Join-Path $imagenetRoot "train"
$valRoot = Join-Path $imagenetRoot "val"
$valRawRoot = Join-Path $imagenetRoot "val_raw"

foreach ($required in @($trainTar, $valTar, $devkitTar)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required archive: $required"
    }
}

Write-Log "Preparing ImageNet under $imagenetRoot"

if (-not (Test-Path -LiteralPath (Join-Path $devkitRoot "data\meta.mat"))) {
    Write-Log "Extracting devkit."
    tar -xzf $devkitTar -C $imagenetRoot
} else {
    Write-Log "Devkit already extracted."
}

New-Item -ItemType Directory -Force -Path $trainRoot | Out-Null
$trainClassDirs = @(Get-ChildItem -LiteralPath $trainRoot -Directory -Filter "n*" -ErrorAction SilentlyContinue)
if ($trainClassDirs.Count -lt 1000) {
    $classTars = @(Get-ChildItem -LiteralPath $trainRoot -File -Filter "n*.tar" -ErrorAction SilentlyContinue)
    if (($trainClassDirs.Count + $classTars.Count) -ge 1000) {
        Write-Log "Resuming with $($trainClassDirs.Count) class folders and $($classTars.Count) class tar files."
    } elseif ($classTars.Count -lt 1000) {
        Write-Log "Extracting train archive into class tar files. This can take a while."
        tar -xf $trainTar -C $trainRoot
    } else {
        Write-Log "Class tar files already present."
    }

    $classTars = @(Get-ChildItem -LiteralPath $trainRoot -File -Filter "n*.tar" | Sort-Object Name)
    $index = 0
    foreach ($classTar in $classTars) {
        $index += 1
        $wnid = [System.IO.Path]::GetFileNameWithoutExtension($classTar.Name)
        $classDir = Join-Path $trainRoot $wnid
        Assert-Within -Path $classDir -Root $trainRoot
        New-Item -ItemType Directory -Force -Path $classDir | Out-Null
        $existingImages = @(Get-ChildItem -LiteralPath $classDir -File -Filter "*.JPEG" -ErrorAction SilentlyContinue)
        if ($existingImages.Count -lt 1000) {
            Write-Log "Extracting train class $index/$($classTars.Count): $wnid"
            tar -xf $classTar.FullName -C $classDir
        } else {
            Write-Log "Train class already extracted $index/$($classTars.Count): $wnid"
        }
        Assert-Within -Path $classTar.FullName -Root $trainRoot
        Remove-WithRetry -Path $classTar.FullName
    }
} else {
    Write-Log "Train directory already appears organized with $($trainClassDirs.Count) class folders."
}

New-Item -ItemType Directory -Force -Path $valRoot | Out-Null
$valClassDirs = @(Get-ChildItem -LiteralPath $valRoot -Directory -Filter "n*" -ErrorAction SilentlyContinue)
if ($valClassDirs.Count -lt 1000) {
    if (-not (Test-Path -LiteralPath $valRawRoot)) {
        New-Item -ItemType Directory -Force -Path $valRawRoot | Out-Null
        Write-Log "Extracting validation archive to val_raw."
        tar -xf $valTar -C $valRawRoot
    } else {
        Write-Log "Validation raw directory already present."
    }
    Write-Log "Organizing validation images into class folders."
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        $python = "python"
    }
    & $python (Join-Path $PSScriptRoot "organize_imagenet_val.py") --imagenet-root $imagenetRoot
} else {
    Write-Log "Validation directory already appears organized with $($valClassDirs.Count) class folders."
}

$trainCount = (Get-ChildItem -LiteralPath $trainRoot -Directory -Filter "n*" | Measure-Object).Count
$valCount = (Get-ChildItem -LiteralPath $valRoot -Directory -Filter "n*" | Measure-Object).Count
Write-Log "Done. train_classes=$trainCount val_classes=$valCount"
