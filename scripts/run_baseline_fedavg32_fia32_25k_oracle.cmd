@echo off
setlocal

cd /d "%~dp0.."
chcp 65001 > nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set WANDB_MODE=offline

set EXP_DIR=exp\imagenet\vgg16\attack\cp_direct\nodef\baseline_fedavg32_fia32_25k_oracle_uia
if not exist "%EXP_DIR%" mkdir "%EXP_DIR%"
set RUN_LOG=%EXP_DIR%\tmux_stdout.log

echo [%DATE% %TIME%] Starting baseline FedAvg batch32 FIA32 25k oracle > "%RUN_LOG%"

"C:\miniconda\envs\DLG\python.exe" baseline\src\attack.py ^
  --ds imagenet ^
  --model vgg16 ^
  --fl_alg fedavg ^
  --defense nodef ^
  --attack cp ^
  --fi_method direct ^
  --batch_size 32 ^
  --n_batch 1 ^
  --n_iter 25000 ^
  --n_iter_fi 25000 ^
  --n_sample_fi 32 ^
  --decor 5.3 ^
  --T 7.7 ^
  --nv 0.13 ^
  --l1 5 ^
  --tv 0.1 ^
  --fi 1 ^
  --oracle_emb_reorder ^
  --fresh_start ^
  --dry_run ^
  --disable_pbar ^
  --exp_name baseline_fedavg32_fia32_25k_oracle >> "%RUN_LOG%" 2>&1

set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE% >> "%RUN_LOG%"
exit /b %EXIT_CODE%
