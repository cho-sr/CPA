# CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006

Status: PARTIAL_COMPLETION; original worker exit1.

Same single-client VGG16 B16/T10 update, Conv frozen/FC trained, seed42; Generator-free Direct FIA.

| Bits | Actual FastICA / GD-ICA / FIA | Status | PSNR dB | SSIM | VGG LPIPS | Attack time |
|---|---|---|---:|---:|---:|---|
| FP32 | 0 / 0 / 0 | Whitening guard stopped execution | unavailable | unavailable | unavailable | unavailable |
| 8 | 4000 / 4000 / 25000 | Completed | 12.054908 | 0.295547 | 0.562428 | 35m 0.3s |
| 4 | 4000 / 4000 / 25000 | Completed existing run | 11.547459 | 0.287770 | 0.805637 | 34m 9.1s |

The saved traceback places the failure in whiten_observation(X,16), before FastICA, GD-ICA or FIA. The input/source provenance gate passes. No final candidate or iteration logs exist. Actual optimization counts: FastICA0, GD-ICA0, FIA0; configured budgets were4000/4000/25000.

Error: observation has insufficient usable covariance rank for source_count. The existing guard rejects nonfinite selected eigenvalues or any selected absolute eigenvalue at or below1e-12. Eigenvalues were not saved; this report does not establish the mathematical rank or whether scaling, numerical precision or data structure triggered the guard.

No reconstruction metrics are available. This execution failure does not establish that FP32 updates cannot be reconstructed. No whitening settings were changed and no optimization was restarted.

New worker total time: 2142.428s (35.707min), including the failed FP32 attempt, INT8 attack and evaluation. The prior INT4 run was reused without rerunning.

All16 final images evaluated for INT8/INT4. Pixel-MSE Hungarian matching is computed once per arm and shared by MSE/PSNR/SSIM/LPIPS. Only truth x is consumed after both attack attempts; no private feedback or intermediate selection.

Visual review: INT8 recovers recognizable outlines in some images, including the child, people holding a fish, lobster, dog and tree scene, but images remain faint and several outputs are dominated by colored textures. The cached INT4 image has poorer recognizable structure. This is a qualitative observation on one batch/seed; no quantitative success cutoff was registered.

INT8 FastICA first crossed tolerance at27, but final residual2.02655792e-6 exceeds1e-6. Exact4000 updates completed with early stopping disabled.

[INT8 original/reconstructed comparison](int8/comparison.png)

[Existing INT4 comparison](../single_client_int4_fast4000_gd4000_fia25000_20261006/comparison.png)

[FP32 failure report](fp32/report.md)
