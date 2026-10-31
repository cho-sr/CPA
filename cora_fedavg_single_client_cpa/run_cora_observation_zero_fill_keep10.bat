@echo off
setlocal

set "PYTHON=C:\miniconda\envs\DLG\python.exe"
set "SCRIPT_DIR=%~dp0"
set "ATTACK_TAG=cora_obs_zero_fill_keep10"
set "CORA_OBS_DIR=%SCRIPT_DIR%outputs_v1\fedavg_single_client_updates\imagenet\vgg16\fedavg_single_client_clients3_round20of20_spc24_target0_local1_5441531bedd12a66\global_topk"

if not exist "%PYTHON%" (
    echo [cora-obs-zero] Python interpreter not found: %PYTHON%
    exit /b 1
)

if not exist "%CORA_OBS_DIR%\keep_0p1.pickle" (
    echo [cora-obs-zero] Missing CoRA observation source: %CORA_OBS_DIR%\keep_0p1.pickle
    exit /b 1
)

if not exist "%CORA_OBS_DIR%\keep_0p1_attack_layer_observation.pickle" (
    echo [cora-obs-zero] Missing CoRA attack-layer observation: %CORA_OBS_DIR%\keep_0p1_attack_layer_observation.pickle
    exit /b 1
)

if not exist "%CORA_OBS_DIR%\keep_0p1.fedavg_round_before.pt" (
    echo [cora-obs-zero] Missing CoRA FedAvg checkpoint: %CORA_OBS_DIR%\keep_0p1.fedavg_round_before.pt
    exit /b 1
)

if /I not "%~1"=="run" (
    echo [cora-obs-zero] Ready. Run "%~nx0 run" to launch same-CoRA-observation zero-fill baseline.
    echo [cora-obs-zero] Observation: %CORA_OBS_DIR%\keep_0p1.pickle
    exit /b 0
)

set "REUSE_ARG=--reuse_existing"
if /I "%~2"=="fresh" set "REUSE_ARG=--fresh_start"

"%PYTHON%" "%SCRIPT_DIR%run_cora_fedavg_single_client_cpa.py" ^
    --experiment_tag "%ATTACK_TAG%" ^
    --ds imagenet --model vgg16 ^
    --num_clients 3 --rounds 20 --observe_round 20 ^
    --samples_per_client 24 --target_client_id 0 ^
    --local_epochs 1 --local_lr 0.001 --local_batch_size 24 --micro_batch_size 24 ^
    --keep_ratios 0.1 ^
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
