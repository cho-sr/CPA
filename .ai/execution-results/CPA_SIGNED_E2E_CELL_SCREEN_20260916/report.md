# CPA_SIGNED_E2E_CELL_SCREEN_20260916

Status: **COMPLETED**

The authorized implementation, focused validation, five registered 100-step arms, public endpoint replay, and offline private image evaluation completed. The wrapper exited with status 0. Tests, runner, and evaluator also exited with status 0.

| Stage | Result | Wall seconds |
| --- | --- | ---: |
| Focused and relevant tests | 85 passed, 240 warnings | 11.407795 |
| Public runner | 5 arms completed | 603.019372 |
| Private evaluator | completed | 27.474775 |
| Wrapper total | completed | 641.903975 |

The registered arms were A_FIA, B_H0_low, B_H0_high, B_H1_low, and B_H1_high. H1 used the 12 positive classifier.6.bias code classes and differed from H0. The fixed weights were lambda_base=lambda_low=0.0242029977689154 and lambda_high=0.24202997768915402.

The full-size numeric checks passed: canonical and adjoint terminal deltas were equal, feature gradients with and without grad enabled were equal, and all six FC tensors had zero delta difference and zero fixed-s code difference in the comparison. Exact cell/code feasibility was false for x0 and every final endpoint under both H0 and H1. The public endpoint profiles and per-tensor diagnostics remain in the public output root.

Private evaluation used Hungarian pixel-MSE matching over all 16 images. It recorded `labels_consumed=false`; true labels and true updates were not used for attack, arm selection, or stopping. The measured descriptive means were:

| Arm | MSE | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| A_FIA | 0.078125463 | 11.5162004698 | 0.28166841 | 0.81396208 |
| B_H0_low | 0.078176426 | 11.5206889347 | 0.28166784 | 0.80533356 |
| B_H0_high | 0.078157764 | 11.5218843554 | 0.28277209 | 0.80647491 |
| B_H1_low | 0.078179195 | 11.5127807905 | 0.27631058 | 0.81354740 |
| B_H1_high | 0.078160877 | 11.5217423643 | 0.28270087 | 0.80815700 |
| x0 | 0.078109347 | 11.5173321456 | 0.28489399 | 0.82292460 |

These are paired descriptive batch measurements only. They do not establish reconstruction success, class recovery, feasibility of an unrestricted method, or generalization. Existing protected sources and input artifacts were unchanged. One nonfatal PyTorch scalar-conversion warning is recorded in the runner stderr log; no execution stage failed.

Artifacts:

- Public outputs: `quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916/`
- Private evaluation: `quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916/`
- Execution summary and logs: this directory
