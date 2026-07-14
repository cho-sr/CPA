#!/usr/bin/env bash
set -euo pipefail

cd /root/CPA

CKPT="/root/CPA/fedavg_fp32/outputs_vgg16_parallel/fp32_regen/fp32/global_round_020.pt"
LOG="/root/CPA/quantized_update_cpa/03_randomized_dequant_ensemble_tmux.log"

echo "[$(date -u)] waiting for ${CKPT}" > "${LOG}"

last_size=-1
stable_count=0
while true; do
  if [[ -f "${CKPT}" ]]; then
    size="$(stat -c %s "${CKPT}")"
    echo "[$(date -u)] checkpoint size=${size} stable_count=${stable_count}" >> "${LOG}"
    if [[ "${size}" == "${last_size}" && "${size}" -gt 0 ]]; then
      stable_count=$((stable_count + 1))
    else
      stable_count=0
      last_size="${size}"
    fi

    if [[ "${stable_count}" -ge 2 ]]; then
      break
    fi
  else
    echo "[$(date -u)] checkpoint not found yet" >> "${LOG}"
  fi
  sleep 30
done

echo "[$(date -u)] starting stage 3 randomized dequant ensemble" >> "${LOG}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1

source /opt/conda/etc/profile.d/conda.sh
conda activate cpa

python quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py \
  --ds tiny_imagenet --model vgg16 \
  --global_checkpoint "${CKPT}" \
  --n_samples 128 --n_rounds 10 \
  --local_epochs 10 --local_batch_size 128 \
  --rounding nearest \
  --ensemble_size 8 \
  --attack_n_iter 5000 --n_iter_fi 5000 \
  --reuse_existing --fresh_start \
  --run_attack >> "${LOG}" 2>&1
