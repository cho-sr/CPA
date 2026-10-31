# Top-k retention comparison: same80 CPA + FIA

Same original80 images, five n16/T10 round0 FC-only SGD updates, seeds, CPA/FIA budgets and post-FIA matching. Only retained fraction changes. Top-k50 is the completed historical paired run; 30/20/10 are the requested new runs.

| Retained | LPIPS (lower) | SSIM (higher) | PSNR dB (higher) | FC1 bytes/batch | Saved |
|---|---:|---:|---:|---:|---:|
| 50% | 0.486383 | 0.334436 | 22.2878 | 218,365,984 | 46.8750% |
| 30% | 0.509832 | 0.314084 | 22.1371 | 136,157,624 | 66.8750% |
| 20% | 0.553605 | 0.290602 | 21.8720 | 95,053,444 | 76.8750% |
| 10% | 0.632326 | 0.263122 | 21.5305 | 53,949,264 | 86.8750% |

Communication savings cover the actual attacked FC1 bitmap + FP32 packet relative to a same-shape dense packet. This does not measure full-model network traffic, latency or training utility. Source-cosine counts are embedding diagnostics, not image-success counts. Top-k50 was numerically lossless for these updates; see the new input audit for 30/20/10 distortion.

[Top-k 50%: all80 originals and reconstructions](../cpa_fia_topk50_same80_B16/all80_reconstruction_landscape.png)
[Top-k 30%: all80 originals and reconstructions](../cpa_fia_topk30_same80_B16/all80_reconstruction_landscape.png)
[Top-k 20%: all80 originals and reconstructions](../cpa_fia_topk20_same80_B16/all80_reconstruction_landscape.png)
[Top-k 10%: all80 originals and reconstructions](../cpa_fia_topk10_same80_B16/all80_reconstruction_landscape.png)
