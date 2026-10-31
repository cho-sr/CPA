@echo off
setlocal

cd /d "%~dp0.."
chcp 65001 > nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1

set INPUT_PICKLE=int 4 fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42\round_020_int4_dequant_aggregate_cpa.pickle
set OUT_DIR=int 4 fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42\fastica_dequantized_aggregate
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set RUN_LOG=%OUT_DIR%\run_stdout.log

echo [%DATE% %TIME%] Starting FastICA on int4 dequantized FedAvg aggregate > "%RUN_LOG%"

"C:\miniconda\envs\DLG\python.exe" "int 4 fedavg\fastica_dequantized_aggregate.py" ^
  --input_pickle "%INPUT_PICKLE%" ^
  --output_dir "%OUT_DIR%" ^
  --batch_index 0 ^
  --attack_layer_index 26 ^
  --n_components 24 ^
  --max_iter 1000 ^
  --tol 1e-4 ^
  --random_state 42 ^
  --dtype float32 ^
  --whiten unit-variance ^
  --fun logcosh ^
  --fresh_start >> "%RUN_LOG%" 2>&1

set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE% >> "%RUN_LOG%"
exit /b %EXIT_CODE%
