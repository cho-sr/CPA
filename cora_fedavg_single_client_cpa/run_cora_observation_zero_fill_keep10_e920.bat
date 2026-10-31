@echo off
setlocal

set "PYTHON=C:\miniconda\envs\DLG\python.exe"
set "SCRIPT_DIR=%~dp0"
set "ATTACK_TAG=zf_keep10_e920"
set "CORA_OBS_DIR=%SCRIPT_DIR%outputs_v1\fedavg_single_client_updates\imagenet\vgg16\fedavg_single_client_clients3_round20of20_spc24_target0_local1_e920201510be97a8\global_topk"
set "UPDATE_FILE=%CORA_OBS_DIR%\keep_0p1.pickle"
set "OBS_FILE=%CORA_OBS_DIR%\keep_0p1_attack_layer_observation.pickle"
set "CKPT_FILE=%CORA_OBS_DIR%\keep_0p1.fedavg_round_before.pt"

if not exist "%PYTHON%" (
    echo [zf-k10] Python interpreter not found: %PYTHON%
    exit /b 1
)

if /I not "%~1"=="run" (
    echo [zf-k10] Ready. Run "%~nx0 run" to launch same-series keep 10%% zero-fill baseline.
    echo [zf-k10] Observation root: %CORA_OBS_DIR%
    exit /b 0
)

set "HAVE_ALL=1"
if not exist "%UPDATE_FILE%" set "HAVE_ALL=0"
if not exist "%OBS_FILE%" set "HAVE_ALL=0"
if not exist "%CKPT_FILE%" set "HAVE_ALL=0"

if "%HAVE_ALL%"=="1" (
    echo [zf-k10] Reusing keep=0.1 source, observation, and checkpoint.
) else (
    echo [zf-k10] Preparing missing keep=0.1 source/observation/checkpoint.
    "%PYTHON%" "%SCRIPT_DIR%run_cora_fedavg_single_client_cpa.py" ^
        --experiment_tag "%ATTACK_TAG%_prepare" ^
        --ds imagenet --model vgg16 ^
        --num_clients 3 --rounds 20 --observe_round 20 ^
        --samples_per_client 24 --target_client_id 0 ^
        --local_epochs 1 --local_lr 0.001 --local_batch_size 24 --micro_batch_size 4 ^
        --keep_ratios 0.1 ^
        --ensemble_size 1 --noise_seed 1234 --desparse_sampler zero_fill ^
        --skip_consistency_ranking ^
        --attack_n_iter 25000 --attack_n_log 5000 ^
        --n_iter_fi 25000 --n_log_fi 5000 --n_sample_fi 24 --fia_micro_batch_size 24

    if errorlevel 1 (
        if exist "%UPDATE_FILE%" if exist "%OBS_FILE%" if exist "%CKPT_FILE%" (
            echo [zf-k10] Prepare reported a failure after required files were written; continuing.
        ) else (
            echo [zf-k10] Failed while preparing keep=0.1.
            exit /b 1
        )
    )
)

set "REUSE_ARG=--reuse_existing"
if /I "%~2"=="fresh" set "REUSE_ARG=--fresh_start"

"%PYTHON%" "%SCRIPT_DIR%run_cora_fedavg_single_client_cpa.py" ^
    --experiment_tag "%ATTACK_TAG%" ^
    --ds imagenet --model vgg16 ^
    --num_clients 3 --rounds 20 --observe_round 20 ^
    --samples_per_client 24 --target_client_id 0 ^
    --local_epochs 1 --local_lr 0.001 --local_batch_size 24 --micro_batch_size 4 ^
    --keep_ratios 0.1 ^
    --ensemble_size 1 --noise_seed 1234 --desparse_sampler zero_fill ^
    --run_attack --skip_consistency_ranking ^
    --attack_n_iter 25000 --attack_n_log 5000 ^
    --n_iter_fi 25000 --n_log_fi 5000 --n_sample_fi 24 --fia_micro_batch_size 24 ^
    %REUSE_ARG%

if errorlevel 1 (
    echo [zf-k10] CPA/FIA run failed.
    exit /b 1
)

endlocal
