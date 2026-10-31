@echo off
setlocal

set "PYTHON=C:\miniconda\envs\DLG\python.exe"
set "SCRIPT_DIR=%~dp0"
set "ATTACK_TAG=cora_obs_zero_fill_keep80_50_30"
set "CORA_OBS_DIR=%SCRIPT_DIR%outputs_v1\fedavg_single_client_updates\imagenet\vgg16\fedavg_single_client_clients3_round20of20_spc24_target0_local1_e920201510be97a8\global_topk"

if not exist "%PYTHON%" (
    echo [cora-obs-zero] Python interpreter not found: %PYTHON%
    exit /b 1
)

if /I not "%~1"=="run" (
    echo [cora-obs-zero] Ready. Run "%~nx0 run" to launch same-CoRA-observation zero-fill baselines for keep 80/50/30.
    echo [cora-obs-zero] Observation root: %CORA_OBS_DIR%
    exit /b 0
)

call :ensure_ratio 0p8 0.8
if errorlevel 1 exit /b 1
call :ensure_ratio 0p5 0.5
if errorlevel 1 exit /b 1
call :ensure_ratio 0p3 0.3
if errorlevel 1 exit /b 1

set "REUSE_ARG=--reuse_existing"
if /I "%~2"=="fresh" set "REUSE_ARG=--fresh_start"

"%PYTHON%" "%SCRIPT_DIR%run_cora_fedavg_single_client_cpa.py" ^
    --experiment_tag "%ATTACK_TAG%" ^
    --ds imagenet --model vgg16 ^
    --num_clients 3 --rounds 20 --observe_round 20 ^
    --samples_per_client 24 --target_client_id 0 ^
    --local_epochs 1 --local_lr 0.001 --local_batch_size 24 --micro_batch_size 4 ^
    --keep_ratios 0.8 0.5 0.3 ^
    --ensemble_size 1 --noise_seed 1234 --desparse_sampler zero_fill ^
    --run_attack --skip_consistency_ranking ^
    --attack_n_iter 25000 --attack_n_log 5000 ^
    --n_iter_fi 25000 --n_log_fi 5000 --n_sample_fi 24 --fia_micro_batch_size 24 ^
    %REUSE_ARG%

if errorlevel 1 (
    echo [cora-obs-zero] CPA/FIA run failed.
    exit /b 1
)

endlocal
exit /b 0

:ensure_ratio
set "RATIO_TAG=%~1"
set "RATIO_VALUE=%~2"
set "UPDATE_FILE=%CORA_OBS_DIR%\keep_%RATIO_TAG%.pickle"
set "OBS_FILE=%CORA_OBS_DIR%\keep_%RATIO_TAG%_attack_layer_observation.pickle"
set "CKPT_FILE=%CORA_OBS_DIR%\keep_%RATIO_TAG%.fedavg_round_before.pt"

if exist "%UPDATE_FILE%" if exist "%OBS_FILE%" if exist "%CKPT_FILE%" (
    echo [cora-obs-zero] Reusing keep=%RATIO_VALUE% source, observation, and checkpoint.
    exit /b 0
)

echo [cora-obs-zero] Preparing missing keep=%RATIO_VALUE% source/observation/checkpoint.
"%PYTHON%" "%SCRIPT_DIR%run_cora_fedavg_single_client_cpa.py" ^
    --experiment_tag "%ATTACK_TAG%_prepare" ^
    --ds imagenet --model vgg16 ^
    --num_clients 3 --rounds 20 --observe_round 20 ^
    --samples_per_client 24 --target_client_id 0 ^
    --local_epochs 1 --local_lr 0.001 --local_batch_size 24 --micro_batch_size 4 ^
    --keep_ratios %RATIO_VALUE% ^
    --ensemble_size 1 --noise_seed 1234 --desparse_sampler zero_fill ^
    --skip_consistency_ranking ^
    --attack_n_iter 25000 --attack_n_log 5000 ^
    --n_iter_fi 25000 --n_log_fi 5000 --n_sample_fi 24 --fia_micro_batch_size 24

if errorlevel 1 (
    if exist "%UPDATE_FILE%" if exist "%OBS_FILE%" if exist "%CKPT_FILE%" (
        echo [cora-obs-zero] Prepare reported a failure after required keep=%RATIO_VALUE% files were written; continuing.
        exit /b 0
    )
    echo [cora-obs-zero] Failed while preparing keep=%RATIO_VALUE%.
    exit /b 1
)

if not exist "%UPDATE_FILE%" (
    echo [cora-obs-zero] Missing after prepare: %UPDATE_FILE%
    exit /b 1
)
if not exist "%OBS_FILE%" (
    echo [cora-obs-zero] Missing after prepare: %OBS_FILE%
    exit /b 1
)
if not exist "%CKPT_FILE%" (
    echo [cora-obs-zero] Missing after prepare: %CKPT_FILE%
    exit /b 1
)
exit /b 0
