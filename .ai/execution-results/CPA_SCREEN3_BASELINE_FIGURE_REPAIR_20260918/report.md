# CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918

Status: **COMPLETED_AVAILABLE_PENDING_PARENT**.

The original plotting bug initialized the first displayed column with ground truth, then labeled it with the first candidate name and skipped that candidate. The isolated CPU renderer now writes a new five-column final sheet and ten-column 5000-step sheet.

The public gate and every candidate file/hash were authenticated before loading the registered private x bundle. The stored `image_metrics[*].per_image` Hungarian correspondence and `mean_metrics` were reused exactly; no matching, metric, LPIPS, reranking, optimizer, or model inference was run.

## Corrected figures

### 8-bit

- Final: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918/figures_with_baseline_20260918/contact_sheet_b8_25000_with_baseline.png`
- Screen: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918/figures_with_baseline_20260918/contact_sheet_b8_5000_with_baseline.png`
- Manifest: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918/figures_with_baseline_20260918/figure_manifest.json`
- Frozen public selection: `[1, 7, 5]`

### 7-bit

- Final: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/contact_sheet_b7_25000_with_baseline.png`
- Screen: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/contact_sheet_b7_5000_with_baseline.png`
- Manifest: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/figure_manifest.json`
- Frozen public selection: `[0, 5, 3]`

### 6-bit

- Final: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/contact_sheet_b6_25000_with_baseline.png`
- Screen: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/contact_sheet_b6_5000_with_baseline.png`
- Manifest: `C:/Users/조상제/Desktop/CPA/quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/figure_manifest.json`
- Frozen public selection: `[1, 3, 4]`

## Existing metric comparison
Values below are copied from the original comparison.json files; no metric or matching was recomputed.

| Bits | Set | MSE | PSNR dB | SSIM | LPIPS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | center_25000 | 0.0706620541313 | 12.0569057933 | 0.292355354623 | 0.559406101704 |
| 8 | rank1_member_01_25000 | 0.0701470816373 | 12.0567617973 | 0.307018888898 | 0.555841319263 |
| 8 | rank2_member_07_25000 | 0.0703305734302 | 12.0440286399 | 0.305054534773 | 0.554504306987 |
| 8 | rank3_member_05_25000 | 0.0704995550494 | 12.0291946019 | 0.305001579278 | 0.553453439847 |
| 7 | center_25000 | 0.0730015561342 | 11.7954583999 | 0.276422813596 | 0.611827094108 |
| 7 | rank1_member_00_25000 | 0.0726417561477 | 11.8313662434 | 0.282749930874 | 0.6331724599 |
| 7 | rank2_member_05_25000 | 0.0727417305073 | 11.8058270494 | 0.292888123884 | 0.609851209447 |
| 7 | rank3_member_03_25000 | 0.0735281625781 | 11.7488672675 | 0.290781549221 | 0.650342069566 |
| 6 | center_25000 | 0.0755217832955 | 11.6397359737 | 0.259638357996 | 0.69612095505 |
| 6 | rank1_member_01_25000 | 0.0754578508736 | 11.6674870966 | 0.26898764354 | 0.665904251859 |
| 6 | rank2_member_03_25000 | 0.0756782127871 | 11.5976584402 | 0.272768729289 | 0.670988338068 |
| 6 | rank3_member_04_25000 | 0.0763237465047 | 11.5724373039 | 0.271388472833 | 0.705177102238 |

The baseline is same-bit naive dequantization -> FastICA/GD ICA4000 -> same Direct FIA, center target. The old PNGs remain in their original evaluation directories and are marked preserved in `available_manifest.json`; they retain the truth-column labeling defect. Bits 5 and 4 had no completed authenticated evaluation directory at this reporting pass and remain pending for the existing sweep completion flow.

Scientific experiment state: unchanged. Actual-image cell feasibility: NOT_EVALUATED.
