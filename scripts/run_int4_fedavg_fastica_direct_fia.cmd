@echo off
setlocal

cd /d "%~dp0.."
chcp 65001 > nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1

set OUT_DIR=int 4 fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42\fastica_dequantized_aggregate\direct_fia_oracle
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set RUN_LOG=int 4 fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42\fastica_dequantized_aggregate\direct_fia_oracle_run_stdout.log

echo [%DATE% %TIME%] Starting Direct FIA from FastICA embeddings > "%RUN_LOG%"

"C:\miniconda\envs\DLG\python.exe" "int 4 fedavg\run_fastica_direct_fia.py" ^
  --embedding_source oracle ^
  --n_sample 24 ^
  --n_iter_fi 25000 ^
  --n_log_fi 500 ^
  --lr_fi 0.1 ^
  --tv 0.1 ^
  --fi 1 ^
  --opt_fi adam ^
  --sch_fi cosine ^
  --device cuda ^
  --fresh_start >> "%RUN_LOG%" 2>&1

set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE% >> "%RUN_LOG%"
exit /b %EXIT_CODE%
