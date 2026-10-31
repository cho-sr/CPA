# CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006

Status: COMPLETED.

Single-client B16/T10 signed symmetric deterministic per-tensor INT4; frozen VGG16 Conv, trained FC; seed42.

FastICA exactly4000 (early stop disabled), GD-ICA exactly4000, Direct FIA exactly25000. No generator, cell optimization, new observation, candidate selection or private feedback.

PSNR: 11.547459 dB; SSIM: 0.287770; VGG LPIPS: 0.805637; MSE: 0.07783512.

Separation: 45.66s; FIA: 2002.92s.

All16 images evaluated. Pixel-MSE Hungarian matching is computed once and shared by every metric. Only x is consumed from the deserialized evaluation bundle; bundled keys are disclosed in metrics.json. No quantitative success threshold was supplied; software completion alone is not proof of successful reconstruction. No actual-update cell feasibility claim.

![Original and reconstructed images](C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\single_client_int4_fast4000_gd4000_fia25000_20261006\comparison.png)

Evaluation/reporting repair: original worker exited 1 after completing all optimization. Its evaluator compared the source-data SHA to the derived evaluation bundle. The evaluation bundle SHA matches both original sender validation and the 2026-09-17 evaluation record. Only this file-identity expectation was corrected in memory; the same frozen candidate, original public gate and metric code were used. No optimization was restarted; all sealed public artifacts and source files are unchanged.

Attack wall time: 2049.116s (34.152min). Peak CUDA allocated memory: 3.285GiB.

FastICA first residual below tolerance: iteration 84; final residual 1.54972076e-06 is above the 1e-6 tolerance. The run nevertheless completed exactly 4000 updates with early stopping disabled.

Visual review of all16 paired images: reconstructions show colored textures without clearly recognizable original subjects or shapes. This is a qualitative observation; no numerical success threshold or failure cause was established.
