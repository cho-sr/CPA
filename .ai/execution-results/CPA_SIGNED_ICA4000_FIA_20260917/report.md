# CPA_SIGNED_ICA4000_FIA_20260917

Status: **COMPLETED**. Scientific quality: **EVALUATED**.

The one registered experiment completed with SEPARATE exit 0, INVERT exit 0, and PRIVATE_EVALUATION exit 0. The public gate was passed before private x-only loading. No cell optimization, FC replay, label search, generator, extra arm, seed, tuning, or continuation was executed.

| Set | MSE | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| ica_gd1000_direct_fia25000 | 0.0776844093863 | 11.524946432 | 0.293223489803 | 0.809157475829 |
| ica_gd4000_direct_fia25000 | 0.0778652300878 | 11.5196560233 | 0.292093418971 | 0.807073246688 |

New minus archived 1000-step control:
- MSE `0.000180820701432`, PSNR `-0.00529040869266`, SSIM `-0.00113007083139`, LPIPS `-0.00208422914147`.
- Image-level improved/worsened/tied: MSE 9/7/0; PSNR 9/7/0; SSIM 10/6/0; LPIPS 10/6/0.

Validation:
- 4000 ICA rows and exact 1000-row structured prefix reproduction passed.
- FIA completed exactly 25000 steps; peak CUDA allocated bytes were 3526829568 under the 12884901888 cap.
- Archived control reproduced all four reviewed mean metrics with zero residual and unchanged Hungarian correspondences.
- All 16 images were evaluated with the registered pixel-MSE Hungarian correspondence reused for MSE/PSNR/SSIM/LPIPS and the aligned figure.
- Exact actual-image cell feasibility is false by task scope and was not executed.

Detailed comparison: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_ica4000_fia_B16_T10_20260917/comparison.json`; figure: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_ica4000_fia_B16_T10_20260917/all16_comparison.png`.
Final audit: `C:/Users/조상제/Desktop/CPA/.ai/execution-results/CPA_SIGNED_ICA4000_FIA_20260917/final_audit.json`.
