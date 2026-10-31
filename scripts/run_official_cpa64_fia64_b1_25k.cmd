@echo off
setlocal

cd /d "%~dp0\.."
set "RUN_LOG=quantized_update_cpa\run_official_cpa64_fia64_b1_25k_rerun.log"
set "PYTHON=.venv\Scripts\python.exe"
set "WANDB_MODE=offline"
set "PYTHONPATH="

echo [%DATE% %TIME%] Starting official CPA64/FIA64 25k run, n_batch=1.>> "%RUN_LOG%"
echo [%DATE% %TIME%] Collecting official FedSGD gradients batch_size=64 n_batch=1.>> "%RUN_LOG%"
"%PYTHON%" scripts\collect_official_grads_one_batch_size.py --ds imagenet --model vgg16 --batch_size 64 --n_batch 1 >> "%RUN_LOG%" 2>&1
if errorlevel 1 (
    echo [%DATE% %TIME%] Gradient collection failed with exit code %ERRORLEVEL%.>> "%RUN_LOG%"
    exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Running official attack.py CPA+FIA, n_batch=1.>> "%RUN_LOG%"
"%PYTHON%" scripts\run_official_cpa.py --script attack -- --attack cp --batch_size 64 --nv 0.13 --l1 5 --T 7.7 --decor 5.3 --fi 1 --tv 0.1 --fi_method direct --n_sample_fi 64 --ds imagenet --model vgg16 --n_iter 25000 --n_iter_fi 25000 --n_batch 1 --fl_alg fedsgd --exp_name official_cpa64_fia64_b1_25k --fresh_start --dry_run --disable_pbar >> "%RUN_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Official CPA64/FIA64 b1 25k run finished with exit code %EXIT_CODE%.>> "%RUN_LOG%"
exit /b %EXIT_CODE%
