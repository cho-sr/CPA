# FedAvg round_001 client 0: FC1 INT4 scale 비교 (B=16, SGD 3 steps)

round_001 global VGG16에서 client 0의 batch00 이미지 16장만 사용해 FC 6개 텐서를 SGD로 3 step 업데이트했습니다. 
FP32, INT4 per-tensor, INT4 per-channel의 FC1 업데이트를 각각 CPA 25,000회 후 abs(S_hat)와 
round_001 convolution feature extractor를 쓰는 원본 Direct FIA 25,000회로 복원했습니다.

| Arm | FC1 update 상대 L2 오차 | update cosine | PSNR (dB) | SSIM | VGG LPIPS | embedding cosine |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 0 | 1 | 12.29150 | 0.343760 | 0.498426 | 0.999318 |
| int4_per_tensor | 0.690266 | 0.758264 | 11.60285 | 0.292153 | 0.816701 | 0.609303 |
| int4_per_channel | 0.339809 | 0.944539 | 11.68786 | 0.295502 | 0.686431 | 0.819255 |

- Per-tensor scale: `9.4620291e-06`; active/zero FC1 rows: 4088/8.
- Active row scales below `s_tensor/14`: 151; these rows contain 0.1080% of total update energy.
- Per-channel scale quantiles exclude zero rows; zero rows are reported separately and use stored scale 1.
- Pixel metrics use one pixel-MSE Hungarian matching per arm, reused for PSNR, SSIM, and LPIPS. 
  Embedding cosine uses Hungarian matching against round_001 convolution features. No success cutoff was applied.
- This is a descriptive comparison of 16 images and one seed; it does not establish generalization.
- INT4 is applied only to FC1 weight update; the other five FC updates remain FP32 inputs to their respective CPA runs.

Numeric details: `comparison.json`, row-scale plot: `scale_only.png`, reconstruction grid: `comparison.png`.
