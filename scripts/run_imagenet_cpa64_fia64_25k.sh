#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
imagenet_root="$repo_root/datasets/imagenet"
run_log="$repo_root/quantized_update_cpa/run_01_imagenet_cpa64_fia64_25k_tmux.log"

write_log() {
    local message="$1"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] $message" | tee -a "$run_log"
}

count_dirs() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo 0
        return
    fi
    find "$path" -maxdepth 1 -type d -name 'n*' | wc -l
}

count_files() {
    local path="$1"
    local pattern="$2"
    if [[ ! -d "$path" ]]; then
        echo 0
        return
    fi
    find "$path" -maxdepth 1 -type f -name "$pattern" | wc -l
}

train_root="$imagenet_root/train"
val_root="$imagenet_root/val"

write_log "Waiting for ImageNet train/val ImageFolder layout."
while true; do
    train_classes="$(count_dirs "$train_root")"
    val_classes="$(count_dirs "$val_root")"
    train_tars="$(count_files "$train_root" 'n*.tar')"
    write_log "ImageNet status: train_classes=$train_classes val_classes=$val_classes train_tars_remaining=$train_tars"
    if [[ "$train_classes" -ge 1000 && "$val_classes" -ge 1000 && "$train_tars" -eq 0 ]]; then
        break
    fi
    sleep 60
done

python="$repo_root/.venv/Scripts/python.exe"
if [[ ! -x "$python" ]]; then
    python="python.exe"
fi

export PYTHONPATH=
export WANDB_MODE=offline

args=(
    "quantized_update_cpa/01_fp32_update_cpa.py"
    "--ds" "imagenet"
    "--model" "vgg16"
    "--n_samples" "64"
    "--attack_trials" "10"
    "--local_epochs" "10"
    "--local_lr" "1e-3"
    "--seed" "42"
    "--run_attack"
    "--attack_n_iter" "25000"
    "--n_iter_fi" "25000"
    "--n_sample_fi" "64"
)

if [[ "${CPA_FRESH_START:-}" == "1" ]]; then
    write_log "Fresh start requested for author run output."
    args+=("--fresh_start")
fi

write_log "Starting FP32 author CPA64+FIA64 25k/25k without micro-batching."
write_log "$python ${args[*]}"
cd "$repo_root" || exit 1
"$python" "${args[@]}" >> "$run_log" 2>&1
exit_code=$?
write_log "Run finished with exit code $exit_code."
exit "$exit_code"
