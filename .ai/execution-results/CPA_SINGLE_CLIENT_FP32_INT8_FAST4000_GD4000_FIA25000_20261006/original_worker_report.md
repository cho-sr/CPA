# CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006

Same single-client B16/T10 update, VGG16 Conv frozen/FC trained, seed42.

FastICA4000 + GD-ICA4000 + Direct FIA25000 per new arm. Existing INT4 result reused without rerunning.

| Bits | PSNR dB | SSIM | VGG LPIPS | Attack seconds |
|---|---:|---:|---:|---:|
| 8 | 12.054908 | 0.295547 | 0.562428 | 2100.293 |
| 4 | 11.547459 | 0.287770 | 0.805637 | 2049.116 |

All16 final images; pixel-MSE Hungarian matching once per arm shared by every metric. Truth x used only after both attacks; no generator, private feedback, tuning or intermediate selection.

Software completion alone does not establish successful reconstruction; no numerical success cutoff registered.

[Bits8 image comparison](C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\single_client_fp32_int8_fast4000_gd4000_fia25000_20261006\int8\comparison.png)

[Bits4 image comparison](C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\single_client_int4_fast4000_gd4000_fia25000_20261006\comparison.png)
