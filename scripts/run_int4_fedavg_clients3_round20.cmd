@echo off
setlocal

cd /d "%~dp0.."
chcp 65001 > nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1

set OUT_DIR=int 4 fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set RUN_LOG=%OUT_DIR%\run_stdout.log

echo [%DATE% %TIME%] Starting int4 FedAvg aggregate collection > "%RUN_LOG%"

"C:\miniconda\envs\DLG\python.exe" "int 4 fedavg\collect_int4_fedavg_aggregate_updates.py" ^
  --ds imagenet ^
  --model vgg16 ^
  --num_clients 3 ^
  --rounds 20 ^
  --samples_per_client 8 ^
  --local_epochs 10 ^
  --local_batch_size 8 ^
  --lr 0.001 ^
  --seed 42 ^
  --device cuda ^
  --private_split test ^
  --save_final_only ^
  --output_dir "%OUT_DIR%" ^
  --export_baseline_update exp\imagenet\vgg16\updates\nodef\24.pickle >> "%RUN_LOG%" 2>&1

set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE% >> "%RUN_LOG%"
exit /b %EXIT_CODE%
