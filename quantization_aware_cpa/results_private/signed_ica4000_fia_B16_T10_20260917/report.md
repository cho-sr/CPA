# CPA_SIGNED_ICA4000_FIA_20260917

Status: **COMPLETED**.

The public provenance gate passed before the private truth bundle was opened. The evaluator consumed x only; labels and true updates were not consumed.

| Set | MSE | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| ica_gd1000_direct_fia25000 | 0.0776844093863 | 11.524946432 | 0.293223489803 | 0.809157475829 |
| ica_gd4000_direct_fia25000 | 0.0778652300878 | 11.5196560233 | 0.292093418971 | 0.807073246688 |

Control reproduction: **PASSED** (absolute tolerance 2e-08).
Comparison values are new 4000-step ICA minus archived 1000-step ICA after the same FIA25000.
Exact actual-image cell feasibility is false by task scope and was not executed.
Aligned figure uses each set's one pixel-MSE Hungarian correspondence.
