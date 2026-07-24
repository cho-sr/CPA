param(
  [string]$Python = "python",
  [string]$Dataset = "cifar10",
  [string]$Model = "fc2",
  [int]$Samples = 2,
  [int]$Trials = 1,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

& $Python "quantized_update_cpa/05_quant_aware_fedavg_refinement_cpa.py" `
  --ds $Dataset `
  --model $Model `
  --n_samples $Samples `
  --attack_trials $Trials `
  --local_epochs 1 `
  --seed $Seed `
  --label_mode known `
  --run_init_attack `
  --attack_n_iter 2 `
  --attack_n_log 1 `
  --n_iter_fi 0 `
  --refine_n_iter 1 `
  --refine_n_log 1 `
  --display_samples $Samples `
  --fresh_start

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
