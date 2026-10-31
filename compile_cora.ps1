param([switch]$OpenPdf)

$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$paperCompiler = Join-Path $paperRoot '.tools\tectonic-0.17.0\tectonic.exe'
$paperSource = Join-Path $paperRoot 'test_cora.tex'
$paperPdf = Join-Path $paperRoot 'test_cora.pdf'

if (-not (Test-Path -LiteralPath $paperCompiler -PathType Leaf)) {
    throw "Portable Tectonic is missing: $paperCompiler"
}
if (-not (Test-Path -LiteralPath $paperSource -PathType Leaf)) {
    throw "LaTeX source is missing: $paperSource"
}

$previousTectonicCache = $env:TECTONIC_CACHE_DIR
Push-Location -LiteralPath $paperRoot
try {
    $env:TECTONIC_CACHE_DIR = Join-Path $paperRoot '.tools\tectonic-cache'
    & $paperCompiler -X compile 'test_cora.tex' --untrusted --keep-logs --synctex
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX compilation failed with exit code $LASTEXITCODE."
    }
    Get-Item -LiteralPath $paperPdf | Select-Object FullName, Length, LastWriteTime
    if ($OpenPdf) {
        Invoke-Item -LiteralPath $paperPdf
    }
}
finally {
    $env:TECTONIC_CACHE_DIR = $previousTectonicCache
    Pop-Location
}
