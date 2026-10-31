# Top-k 50% input audit

All five original updates contain fewer than 50% nonzero entries. Exact global Top-k 50% drops only zero coordinates; decoded observations are numerically identical to the saved actual FP32 updates. This is a lossless sparse representation for these five observations, not a nonzero-value pruning test.

| Batch | Original nonzero fraction | Relative L2 error | Retained energy |
|---|---:|---:|---:|
| 0 | 0.38217110 | 0.0 | 1.0 |
| 1 | 0.39174877 | 0.0 | 1.0 |
| 2 | 0.41586502 | 0.0 | 1.0 |
| 3 | 0.42094787 | 0.0 | 1.0 |
| 4 | 0.38743772 | 0.0 | 1.0 |

The requested Top-k definition and 50% retention are preserved. The receiver is still given a real bitmap + FP32 packet only. A historical 32-bit image reconstruction may differ because its CPA implementation/matching order differ; this input audit does not equate those attack implementations.
