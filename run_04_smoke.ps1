param(
  [string]$Python = "python",
  [string]$Dataset = "cifar10",
  [string]$Model = "fc2",
  [int]$Samples = 4,
  [int]$Trials = 1,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

& $Python "quantized_update_cpa/04_quant_aware_ica_objective_cpa.py" `
  --ds $Dataset `
  --model $Model `
  --n_samples $Samples `
  --attack_trials $Trials `
  --local_epochs 1 `
  --seed $Seed `
  --qica_ablation all `
  --qica_n_iter 2 `
  --qica_n_log 1 `
  --qica_columns_per_step 64 `
  --fresh_start

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
