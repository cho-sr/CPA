param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Full experiment commands are listed below; this script does not launch them automatically."
Write-Host "Expected high-memory cases: VGG16, n=128, 10 local steps, especially 05 with create_graph=True."
Write-Host ""

$common = "--ds imagenet --model vgg16 --n_samples 128 --attack_trials 10 --local_epochs 10 --local_lr 1e-3 --seed 42"

Write-Host "01 FP32 author CPA:"
Write-Host "$Python quantized_update_cpa/01_fp32_update_cpa.py $common --run_attack --attack_n_iter 25000 --n_iter_fi 25000 --n_sample_fi 128"
Write-Host ""

Write-Host "02 Naive int4 author CPA:"
Write-Host "$Python quantized_update_cpa/02_naive_4bit_update_cpa.py $common --run_attack --attack_n_iter 25000 --n_iter_fi 25000 --n_sample_fi 128"
Write-Host ""

Write-Host "03 Randomized dequant ensemble CPA plus blind aggregate FIA:"
Write-Host "$Python quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py $common --run_attack --ensemble_size 8 --perturb_scope attack_layer --aggregate mean --reference medoid --run_aggregate_fia --attack_n_iter 25000 --n_iter_fi 25000 --n_sample_fi 128"
Write-Host ""

Write-Host "04 Quantization-aware ICA ablations:"
Write-Host "$Python quantized_update_cpa/04_quant_aware_ica_objective_cpa.py $common --qica_ablation all --qica_n_iter 25000 --qica_n_log 1000 --qica_columns_per_step 8192"
Write-Host ""

Write-Host "05 Known-label FedAvg refinement:"
Write-Host "$Python quantized_update_cpa/05_quant_aware_fedavg_refinement_cpa.py $common --label_mode known --init_run_path <path-to-05_init_author_cpa-run> --refine_n_iter 1000 --refine_n_log 50 --display_samples 16"
