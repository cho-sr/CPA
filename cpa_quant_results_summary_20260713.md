# CPA Quantization Experiments Summary

Archive: `cpa_quant_results_20260713_195111.tar.gz`

This archive stores compact local results for the 4-bit CPA experiments:

- Source/wrapper code used for quantized-update CPA experiments.
- PNG/CSV reports for FP32, naive 4-bit, randomized dequantization, QAS, and QGM runs.
- Small log/summary/iteration result files.
- Large datasets, model checkpoints, update pickles, whitening pickles, and reconstruction pickles are excluded.

Key results:

| Method | CS | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| FP32 baseline | 0.9405 | 22.7606 | 0.1832 | 0.5213 |
| Naive int4 + Direct FIA | 0.6621 | 22.1497 | 0.1410 | 0.6486 |
| Randomized ensemble avg | 0.4412 | 11.2254 | 0.0976 | 0.7458 |
| QAS objective scale-fixed | 0.2147 | N/A | N/A | N/A |
| QGM FIA gm=0.0001 b10 | 0.6591 | 9.7126 | 0.0256 | 0.6894 |

Interpretation:

- The strongest quantized result so far is naive 4-bit dequantization + FastICA/Direct FIA.
- Randomized dequantization and explicit quantization-aware ICA/FIA objectives did not improve recovery in the current formulation.
- Current experiments use 4-bit fake quantization/dequantization only; 8-bit has not been tested yet.
