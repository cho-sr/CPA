# Isolated dependency-gate checks. This never starts the queue or a GPU job.
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Fixture = Join-Path ([System.IO.Path]::GetTempPath()) ("cpa_uia_queue_test_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Fixture -Force | Out-Null
$Tokens = $null
$Errors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $Project "scripts\queue_m6_uia_after_m7.ps1"), [ref]$Tokens, [ref]$Errors)
if ($Errors.Count -ne 0) { throw "Queue script failed to parse" }
foreach ($Definition in $Ast.FindAll({param($Node) $Node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    . ([ScriptBlock]::Create($Definition.Extent.Text))
}
$ConfigPath = Join-Path $Fixture "config.json"
$ConfigLockPath = Join-Path $Fixture "config.lock.json"
$M7Output = $Fixture
$TaskConfig = @{run_tag="m6_uia_fp32_B16_fia";quantization=@{bits="fp32";enabled=$false};decision=@{automatic_4bit=$false};dependency=@{preregistration_sha256="m7-locked"}}
$TaskConfig | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
@{protocol_id="m6_uia_fp32_B16_fia_v1";sha256=(Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $ConfigLockPath -Encoding UTF8
Assert-LockedConfig | Out-Null
@{state="complete"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Fixture "execution_status.json") -Encoding UTF8
@{report_status="complete";preregistration_sha256="m7-locked"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Fixture "summary.json") -Encoding UTF8
Assert-M7Complete

foreach ($State in @("running", "failed", "fia_complete")) {
    @{state=$State} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Fixture "execution_status.json") -Encoding UTF8
    $Rejected = $false
    try { Assert-M7Complete } catch { $Rejected = $true }
    if (-not $Rejected) { throw "Unexpectedly accepted unfinished predecessor: $State" }
}
@{state="complete"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Fixture "execution_status.json") -Encoding UTF8
@{report_status="complete";preregistration_sha256="different"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Fixture "summary.json") -Encoding UTF8
$Rejected = $false
try { Assert-M7Complete } catch { $Rejected = $true }
if (-not $Rejected) { throw "Unexpectedly accepted different M7 protocol" }

# A rehashed 4-bit config must still be rejected by the fp32-only guard.
$TaskConfig.quantization.bits = "4-bit"
$TaskConfig | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
@{protocol_id="m6_uia_fp32_B16_fia_v1";sha256=(Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $ConfigLockPath -Encoding UTF8
$Rejected = $false
try { Assert-LockedConfig | Out-Null } catch { $Rejected = $true }
if (-not $Rejected) { throw "Unexpectedly accepted 4-bit successor" }

Write-Output "6 queue gate checks passed; no experiment process started."
