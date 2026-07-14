#!/usr/bin/env bash
set -euo pipefail

cd /root/CPA

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1

CKPT="/root/CPA/fedavg_fp32/outputs_vgg16_parallel/fp32_regen/fp32/global_round_020.pt"
BASE_EXP="/root/CPA/exp/tiny_imagenet/vgg16/attack/cp_direct/nodef"
STAGE3_PREFIX="03_randomized_dequant_ensemble_cpa_nearest"
STAGE3_LOG="/root/CPA/quantized_update_cpa/03_randomized_dequant_ensemble_tmux.log"
POST_LOG="/root/CPA/quantized_update_cpa/post_stage3_then_stage4.log"

echo "[$(date -u)] waiting for stage 3 K=8 outputs" > "${POST_LOG}"

for k in 0 1 2 3 4 5 6 7; do
  member="$(printf 'k%02d' "${k}")"
  rec_file="${BASE_EXP}/${STAGE3_PREFIX}_${member}_uia/128_rec.pkl"
  fi_summary_file="${BASE_EXP}/${STAGE3_PREFIX}_${member}_uia/128_fi_summary.pkl"
  while [[ ! -s "${rec_file}" || ! -s "${fi_summary_file}" ]]; do
    echo "[$(date -u)] waiting for ${member}: ${rec_file}" >> "${POST_LOG}"
    sleep 60
  done
  echo "[$(date -u)] found ${member}" >> "${POST_LOG}"
done

echo "[$(date -u)] exporting reconstruction PNG grids" >> "${POST_LOG}"

source /opt/conda/etc/profile.d/conda.sh
conda activate cpa

for k in 0 1 2 3 4 5 6 7; do
  member="$(printf 'k%02d' "${k}")"
  rec_file="${BASE_EXP}/${STAGE3_PREFIX}_${member}_uia/128_rec.pkl"
  out_dir="/root/CPA/quantized_update_cpa/stage3_png/${member}"
  python cocktail_party_attack/src/export_attack_pngs.py \
    --rec_file "${rec_file}" \
    --out_dir "${out_dir}" \
    --ds tiny_imagenet \
    --batch all \
    --nrow 8 \
    --max_images 32 >> "${POST_LOG}" 2>&1
done

echo "[$(date -u)] plotting stage 3 metrics" >> "${POST_LOG}"
python - <<'PY' >> "${POST_LOG}" 2>&1
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

base = Path("/root/CPA/exp/tiny_imagenet/vgg16/attack/cp_direct/nodef")
prefix = "03_randomized_dequant_ensemble_cpa_nearest"
out_dir = Path("/root/CPA/quantized_update_cpa/stage3_png/metric_plots")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for k in range(8):
    member = f"k{k:02d}"
    exp_dir = base / f"{prefix}_{member}_uia"
    summary = pd.read_pickle(exp_dir / "128_summary.pkl")
    fi_summary = pd.read_pickle(exp_dir / "128_fi_summary.pkl")
    row = {"member": member}
    if "cs" in summary:
        row["cs"] = float(summary["cs"].mean())
    for metric in ("lpips", "psnr", "ssim"):
        if metric in fi_summary:
            row[metric] = float(fi_summary[metric].mean())
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(out_dir / "stage3_member_metrics.csv", index=False)

for metric in ("cs", "lpips", "psnr", "ssim"):
    if metric not in table:
        continue
    plt.figure(figsize=(7, 4))
    plt.plot(table["member"], table[metric], marker="o")
    plt.xlabel("ensemble member")
    plt.ylabel(metric)
    plt.title(f"Stage 3 {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"stage3_{metric}.png", dpi=160)
    plt.close()

print(table)
PY

echo "[$(date -u)] preparing stage 4 quantized-gradient consistency update pickle reuse" >> "${POST_LOG}"
STAGE3_UPDATE="/root/CPA/quantized_update_cpa/outputs/03_randomized_dequant_ensemble_cpa_nearest/tiny_imagenet/vgg16/updates/128.pickle"
STAGE4_UPDATE="/root/CPA/quantized_update_cpa/outputs/04_quantized_gradient_consistency_fia_nearest_qgm/tiny_imagenet/vgg16/updates/128.pickle"
mkdir -p "$(dirname "${STAGE4_UPDATE}")"
cp -f "${STAGE3_UPDATE}" "${STAGE4_UPDATE}"

echo "[$(date -u)] starting stage 4 quantized-gradient consistency FIA" >> "${POST_LOG}"
python quantized_update_cpa/04_quant_aware_fia_update_cpa.py \
  --ds tiny_imagenet --model vgg16 \
  --global_checkpoint "${CKPT}" \
  --n_samples 128 --n_rounds 10 \
  --local_epochs 10 --local_batch_size 128 \
  --rounding nearest \
  --attack_n_iter 5000 --n_iter_fi 5000 \
  --reuse_existing --fresh_start \
  --run_attack >> "${POST_LOG}" 2>&1

echo "[$(date -u)] stage 4 finished" >> "${POST_LOG}"
