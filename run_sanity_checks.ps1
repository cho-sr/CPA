param(
  [string]$Python = "python",
  [string]$Dataset = "cifar10",
  [string]$Model = "fc2",
  [int]$Samples = 4,
  [int]$Trials = 1,
  [int]$LocalEpochs = 2,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

& $Python "quantized_update_cpa/sanity_checks.py" `
  --ds $Dataset `
  --model $Model `
  --n_samples $Samples `
  --attack_trials $Trials `
  --local_epochs $LocalEpochs `
  --seed $Seed

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
