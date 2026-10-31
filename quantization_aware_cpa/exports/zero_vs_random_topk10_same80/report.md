# Top-k10: Zero fill vs uniform random fill

Same80 image identities/order, original n16x5/T10 updates and identical FC1 packets verified by hashes. Report reads completed results only.

| Metric | Zero fill | Random fill | Random minus zero | Random better images |
|---|---:|---:|---:|---:|
| lpips | 0.632326 | 0.629989 | -0.002337 | 47/80 |
| ssim | 0.263122 | 0.261004 | -0.002118 | 39/80 |
| psnr | 21.530539 | 21.510400 | -0.020139 | 44/80 |

LPIPS is slightly lower; SSIM and PSNR are slightly lower. These descriptive means show no consistent improvement across metrics. All80 are included; counts are paired image comparisons, not independent experimental replicates or a significance test.

Communication is identical:53,949,264 bytes per attacked FC1 packet,86.874994% below a same-shape dense FP32 packet. Random values are generated on the attacker side.

Search compute differs by user request. Per batch zero fill uses one CPA25k/FIA25k candidate; random fill totals20 CPA25k,20 FIA5k plus3 additional FIA20k (500k CPA iterations and160k FIA iterations). Finalists and the final whole-batch candidate were selected by observed-update consistency, not image metrics. No additional control attacks were run.

Figures: same80 paired zero/random images, five original/zero/random batch sheets, and a mean-metric graph. Every exported image tile was checked against the saved tensor rendered with the same RGB conversion and isotropic Lanczos resizing. There is no contrast enhancement or color adjustment.
