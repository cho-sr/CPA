# M7 consistency audit

All calculations use existing CSV/JSON records under the frozen follow-up amendment. No experiment was rerun.

## Canonical loss versus LPIPS correlation

The main fp32 reference is the 36 actual CPA reconstructions. Global indices 22, 24, 32 and 36 are excluded because their historical fp32 entries used true embeddings. The 4-bit CPA + dequant reference includes all 40 images.

| Condition | Images | Mean LPIPS | Spearman rho | Two-sided p |
|---|---:|---:|---:|---:|
| GIA fp32 CPA, pure36 | 36 | 0.4464729486 | +0.030888030888 | 0.858071088545 |
| GIA 4-bit CPA + dequant, all40 | 40 | 0.5439403489 | -0.590619136961 | 6.02093714009e-05 |

Original per-method CSVs agree exactly with the old aggregation and the filtered current reference. These are descriptive pooled image-level Spearman tests, with scipy's asymptotic p-values; they do not test a paired change in correlation or establish a batch-level causal claim. The methods have different sample counts. Historical alternative-scope values are retained only in `spearman_history.csv` for provenance.

![Canonical loss and LPIPS](loss_vs_lpips_main.png)

## B_eff source-count validation

The 96 unique condition keys and saved condition configuration agree. Predicted counts were independently recounted from the saved eigenvalue/MP-edge ratios (> c_mp=3); measured counts were independently recounted from saved matched source cosines (>0.9). Both agree with every recorded count.

Core grid: n={16,32,64}, bits={2,3,4,8}, seed={0,1,2}, T={1,10}, per-row scaling, batch0. Additional checks: n={16,32}, bits=4, the same seeds and T, per-tensor or per-bucket scaling (bucket size 256). All use ImageNet val and stochastic rounding.

| Subset | Exact / conditions | MAE | Maximum error |
|---|---:|---:|---:|
| all granularities; all T | 81/96 | 0.218750 | 2 |
| all granularities; T=1 | 39/48 | 0.229167 | 2 |
| all granularities; T=10 | 42/48 | 0.208333 | 2 |
| core per-row; all T | 62/72 | 0.180556 | 2 |
| core per-row; T=1 | 30/36 | 0.194444 | 2 |
| core per-row; T=10 | 32/36 | 0.166667 | 2 |

T=1 over all 48 conditions is the primary FedSGD subset. Its 36-condition core per-row subset is reported separately. These are condition cells, not 96 independent datasets.

![Square B_eff plot](predicted_vs_measured_beff_square.png)

## n=32 prediction reconciliation

M6 round0 predictions across batches0-4 are [24, 25, 22, 21, 14]; their arithmetic mean is 21.2. The matching scientific condition is n32, bits4, T1, victim seed0, batch0, round0, stochastic per-row quantization, c_mp3 and ImageNet val. All three source runs predict 24 for that condition. This compares predicted counts with predicted counts.

The quantizer random seeds differ: m5_uia_spectrum=0, m5_uia_beff_validation=294949, m6_round_dependence=293536. Therefore this is count agreement across the listed scientific axes, not identical-input numerical reproduction. The m5 spectrum and m6 source logs identify the same 32 original image paths. M6's round-dependence spectrum is FedSGD/T1; its separately saved T10 UIA dumps were not used in this comparison.

The full axis comparison is in `n32_axis_comparison.csv`; all selected cells, including the other m5 T and seed values, are in `n32_prediction_cells.csv`.

## Reproduce

From `quantization_aware_cpa`, run `python -m qacpa.report_m7_consistency_audit`. The script verifies the frozen protocol SHA-256 and validates source membership before writing results. `input_manifest.json` records every input file hash, dependency versions are in `summary.json`, and output hashes are in `output_manifest.json`.
