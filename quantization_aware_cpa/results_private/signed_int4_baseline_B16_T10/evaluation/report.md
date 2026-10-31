# FP32 Reference / signed INT4 naive — B16/T10

[FACT] One new local update; frozen Conv, all FC trained. Both arms: raw Delta/raw sq → official CPA 25,000 → abs sources → Direct FIA 25,000.

| Arm | PSNR dB | SSIM | VGG LPIPS | Embedding cosine |
|---|---:|---:|---:|---:|
| reference | 12.391145 | 0.346884 | 0.459648 | 0.995274 |
| naive | 11.517332 | 0.284894 | 0.822925 | 0.311605 |

[FACT] Naive minus reference: {"mean_psnr_db": -0.8738133125594807, "mean_ssim": -0.06198987517853449, "mean_lpips": 0.3632761724293232, "mean_mse": 0.012518882126713121}

[FACT] All 16 final candidates evaluated. One pixel-MSE Hungarian matching per arm supplies PSNR/SSIM/LPIPS pairs. Embedding matching is separate and offline only.
[FACT] Identical FIA initialization hashes and numerical runtime settings. No true-label attack, oracle selection, reranking, or quantization-aware objective.
[UNKNOWN] No success threshold, significance, or generalization established by this single batch/seed.
[FACT] all16_comparison.png and all16_compact.png are evaluation-only figures.
