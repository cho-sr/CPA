param(
  [string]$Python = "python",
  [string]$Dataset = "cifar10",
  [string]$Model = "fc2",
  [int]$Samples = 4,
  [int]$Trials = 1,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$base = @(
  "--ds", $Dataset,
  "--model", $Model,
  "--n_samples", $Samples,
  "--attack_trials", $Trials,
  "--local_epochs", 1,
  "--seed", $Seed,
  "--run_attack",
  "--attack_n_iter", 2,
  "--attack_n_log", 1,
  "--n_iter_fi", 0,
  "--fresh_start"
)

& $Python "quantized_update_cpa/01_fp32_update_cpa.py" @base
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "quantized_update_cpa/02_naive_4bit_update_cpa.py" @base
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py" @base `
  --ensemble_size 2 `
  --perturb_scope attack_layer
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
