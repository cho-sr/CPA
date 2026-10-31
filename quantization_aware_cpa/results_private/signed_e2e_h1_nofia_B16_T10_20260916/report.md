# CPA_SIGNED_E2E_H1_NOFIA_20260916 finalization

Status: **COMPLETED**.

The parent wrapper exited 1 at its reporting gate after the C_H1_high_noFIA 100-step arm had completed. This finalization reused the saved x0, B_H1_high control, and C_H1_high_noFIA candidate. Private evaluation consumed image tensor `x` only; labels and true updates were not consumed.

| Candidate | MSE | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| x0 | 0.0781093468 | 11.517332145631135 | 0.28489399 | 0.822924625 |
| B_H1_high | 0.0781608771 | 11.521742364310004 | 0.282700871 | 0.808157001 |
| C_H1_high_noFIA | 0.0781504504 | 11.51461116063817 | 0.277993884 | 0.821158864 |

Metrics are descriptive for the fixed all16-image batch and do not change the registered decision rules.
The figure uses each candidate's recorded Hungarian pixel-MSE correspondence.
