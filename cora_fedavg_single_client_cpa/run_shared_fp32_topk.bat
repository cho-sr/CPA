@echo off
setlocal

set "PYTHON=C:\miniconda\envs\DLG\python.exe"
set "SCRIPT_DIR=%~dp0"
set "CONTROL_NAME=imagenet_vgg16_round20_target0_spc24_sharedfp32_from_keep50"
set "ATTACK_TAG=sfp32r20c0k50"
set "SOURCE_FILE=%SCRIPT_DIR%outputs_v1\fedavg_single_client_updates\imagenet\vgg16\fedavg_single_client_clients3_round20of20_spc24_target0_local1_e920201510be97a8\global_topk\keep_0p5.pickle"
set "CONTROL_DIR=%SCRIPT_DIR%outputs_v1\shared_fp32_controls\%CONTROL_NAME%\global_topk"

if not exist "%PYTHON%" (
    echo [shared-fp32] Python interpreter not found: %PYTHON%
    exit /b 1
)

if not exist "%CONTROL_DIR%\keep_1.pickle" (
    "%PYTHON%" "%SCRIPT_DIR%prepare_shared_fp32_topk.py" --source_file "%SOURCE_FILE%" --output_dir "%CONTROL_DIR%" --keep_ratios 1 0.8 0.5 0.3 0.1 --reconstruct_from_checkpoint
    if errorlevel 1 (
        echo [shared-fp32] Preparation failed.
        exit /b 1
    )
) else (
    echo [shared-fp32] Reusing prepared artifacts: %CONTROL_DIR%
)

if /I not "%~1"=="run" (
    echo [shared-fp32] Preparation complete. Run "%~nx0 run" to launch CPA/FIA.
    exit /b 0
)

set "REUSE_ARG="
if /I "%~2"=="reuse" set "REUSE_ARG=--reuse_existing"

"%PYTHON%" "%SCRIPT_DIR%run_cora_fedavg_single_client_cpa.py" ^
    --prepared_source_dir "%CONTROL_DIR%" ^
    --experiment_tag "%ATTACK_TAG%" ^
    --ds imagenet --model vgg16 ^
    --num_clients 3 --rounds 20 --observe_round 20 ^
    --samples_per_client 24 --target_client_id 0 ^
    --local_epochs 1 --local_lr 0.001 --local_batch_size 24 --micro_batch_size 4 ^
    --keep_ratios 1 0.8 0.5 0.3 0.1 ^
    --ensemble_size 8 --noise_seed 1234 --desparse_sampler retained_rescaled ^
    --run_attack --attack_n_iter 25000 --attack_n_log 5000 ^
    --n_iter_fi 25000 --n_log_fi 5000 --n_sample_fi 24 --fia_micro_batch_size 24 ^
    %REUSE_ARG%

if errorlevel 1 (
    echo [shared-fp32] CPA/FIA run failed.
    exit /b 1
)

endlocal
