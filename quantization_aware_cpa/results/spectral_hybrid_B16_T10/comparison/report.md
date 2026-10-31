Completed MP feature-metric comparison: frozen Conv, all FC learn, deterministic uint4, single client, B16/T10.
Identity, flat and MP all use the same hybrid bin/radius objective and private-label update rule. Only the FIA metric treatment differs.
All arms start from the same saved public initialization and run 50 iterations. No additional candidate or iteration is selected using truth.
Identity uses lambda_F=1. Flat's initial image-gradient norm determines one fixed lambda_F shared by flat and MP. Only flat is calibrated to identity's initial feature-gradient norm; MP's remaining norm difference is measured, not independently removed.

| Hybrid metric | PSNR dB ↑ | SSIM ↑ | VGG LPIPS ↓ | Matched label accuracy | Label overlap /16 | Optimization seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 10.85184 | 0.119128 | 0.684643 | 0.4375 | 11 | 117.13 |
| flat | 10.87724 | 0.123372 | 0.694746 | 0.3750 | 10 | 116.84 |
| mp | 10.88154 | 0.125047 | 0.694118 | 0.3750 | 10 | 116.84 |

| Comparison | Δ PSNR | Δ SSIM | Δ LPIPS | Descriptive conclusion |
| --- | ---: | ---: | ---: | --- |
| mp_minus_identity | +0.02969527 | +0.00591884 | +0.00947464 | mixed |
| flat_minus_identity | +0.02539716 | +0.00424430 | +0.01010241 | mixed |
| mp_minus_flat | +0.00429811 | +0.00167454 | -0.00062777 | directional_improvement_in_this_pilot |

| Hybrid metric | Final soft objective | Final hard objective | Hard positive-radius bin violation rate | Exact codes | Exact radius | Exact packet | Closure fit |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| identity | 6.400838 | 108.171854 | 32.092391% | False | False | False | False |
| flat | 6.497621 | 132.974264 | 34.044682% | False | False | False | False |
| mp | 6.491111 | 155.331669 | 33.714317% | False | False | False | False |

Image matching: one pixel-MSE Hungarian assignment per arm, fixed for PSNR, SSIM, LPIPS and label accuracy. Every final image and all six FC observation tensors remain included.
Directional improvement is reported only when LPIPS decreases and both PSNR and SSIM increase. Mixed signs do not establish improvement. These are descriptive results from one fixed pilot, without statistical significance or a reconstruction-success cutoff.
The directional label reports signs only. Small favorable differences, even in all three metrics, do not establish a practically meaningful or robust improvement.
Final soft and hard objectives use each arm's same saved post-step images. They are distinct from pre-step optimization logs. Changing soft probabilities can matter even when hard-label argmax stays unchanged.
The MP weights remain a public spectral heuristic; reconstruction differences do not validate an iid-noise model or establish superiority over other spectral methods.

![All sixteen final images with offline references](all16_comparison.png)
