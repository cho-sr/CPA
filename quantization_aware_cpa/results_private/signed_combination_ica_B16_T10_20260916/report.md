# CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916

Status: **NEEDS_REVIEW**.

The public provenance gate passed before private truth loading. The evaluator consumed x only; bundled labels and true updates were not consumed.

| Set | MSE | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| original_separator_direct | 0.0781093468344 | 11.517332145631135 | 0.284893989708 | 0.822924602777 |
| original_separator_cell | 0.0781608770773 | 11.521742364310006 | 0.282700870678 | 0.808156996965 |
| combination_separator_direct | 0.0776844093863 | 11.524946431998847 | 0.293223489803 | 0.809157457203 |
| combination_separator_cell | 0.0777360436803 | 11.521405662286421 | 0.284907516905 | 0.800327986479 |

Control reproduction: **NEEDS_REVIEW**.
Endpoint code/cell profiles for all six FC tensors are embedded in `comparison.json`.
The aligned figure reuses each set's one pixel-MSE Hungarian correspondence.
