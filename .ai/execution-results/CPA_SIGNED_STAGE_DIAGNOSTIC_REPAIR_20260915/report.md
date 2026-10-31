# CPA_SIGNED_STAGE_DIAGNOSTIC_REPAIR_20260915

Status: **COMPLETED**. The authorized diagnostic repair, focused validation, one corrected saved-result evaluation, numerical report, and latent-G design are complete. Original CPA/FIA/AP5, whitening, loss, optimizer, codec, observations, and historical outputs were not rerun or modified.

## What changed

- [`signed_stage_diagnostics.py`](../../../quantization_aware_cpa/qacpa/signed_stage_diagnostics.py) now retains `target_z` separately from signed `S_hat`. `C_fit` and both `C_target` values use the validated nonnegative `target_z`; signed `S_hat` is used only for sign-energy diagnostics. A ReLU/nonnegative-target negative-fit guard, exact FP32 schema checks, zero/invalid reasons, and global 600-second budget are enforced.
- Provenance now calls the completed evaluator's strict `validate_provenance` before private tensor deserialization, then binds comparison provenance, arm final-result hashes and frozen maps. The old diagnostic's cross-arm mapping count is removed.
- Mapping agreement is reported within each arm (`pi_I` versus `pi_E`), and paired outputs are separated into image truth and embedding truth: [`paired16_by_truth.csv`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired/paired16_by_truth.csv) and [`paired16_by_embedding_truth.csv`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired/paired16_by_embedding_truth.csv).
- The test file [`test_signed_stage_diagnostics.py`](../../../quantization_aware_cpa/tests/test_signed_stage_diagnostics.py) includes mixed-sign synthetic measurement, sign-reversal invariance, separate maps and same-truth pairing checks.

## Validation and execution

Focused command exit 0: **57 passed, 240 warnings in 10.27 s**. Evidence: [`tests.log`](tests.log) and [`integrity.json`](integrity.json).

The corrected command ran exactly once in a fresh process (PID 2172); execution record exit 0, callback process exit 0, and callback receipt `01a0a4b2-1738-7101-a8d3-9c9376b44328` were recorded. It evaluated 32 saved final images (16 naive, 16 AP5) with CUDA, microbatch 1, frozen/eval FP32 VGG16 features->avgpool->flatten, no autograd and TF32 disabled. Total feature wall time was 0.7191453 s; peak allocated CUDA memory was 663,822,336 bytes, below 6 GiB and the 600 s global deadline.

Output directory: [`stage_diagnostic_v2_repaired`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired). Machine-readable rows and features are in [`diagnostic.json`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired/diagnostic.json), [`all32_per_candidate.csv`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired/all32_per_candidate.csv), [`features_h.pt`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired/features_h.pt), and [`stage_diagnostic_v2_repaired.png`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2_repaired/stage_diagnostic_v2_repaired.png).

## Corrected measurements

All values are over 16 candidates per arm; each line is mean, median, range, and defined n.

- Naive: C_fit (h,target_z) mean=0.930519005, median=0.938546622, range=[0.854875326, 0.985488689], n=16; C_target_I mean=0.141264282, median=0.0583342026, range=[0.0196123912, 0.711426088], n=16; C_image_I mean=0.117020653, median=0.0380171734, range=[0.0123556402, 0.676861213], n=16; C_target_E mean=0.311605162, median=0.0982583573, range=[0.0196123912, 0.734092965], n=16; C_image_E mean=0.276817226, median=0.0800440801, range=[0.0123556402, 0.677559936], n=16.
- Cell-rank: C_fit mean=0.926128559, median=0.933560647, range=[0.846110915, 0.976098264], n=16; C_target_I mean=0.2358211, median=0.0799475396, range=[0.0239773682, 0.778818169], n=16; C_image_I mean=0.199139135, median=0.048085712, range=[0.00914818993, 0.730050066], n=16; C_target_E mean=0.344761396, median=0.132592051, range=[0.0239773682, 0.778818169], n=16; C_image_E mean=0.310750169, median=0.0822211048, range=[0.0139817965, 0.730050066], n=16.

The corrected C_fit means are **0.930519 (naive)** and **0.926129 (cell-rank)**; the prior negative means were caused by the signed-target data-flow defect and are superseded. All 32 corrected rows are retained. The stored image metrics remain valid independent evaluator results and were not recomputed: naive MSE 0.0781093, PSNR 11.5173 dB, SSIM 0.284894, LPIPS 0.822925; cell-rank MSE 0.0779299, PSNR 11.5045 dB, SSIM 0.281657, LPIPS 0.794168.

Within-arm frozen-map agreement: naive count 2/16 (fraction 0.125; disagreement IDs 0,1,2,3,4,5,6,7,8,11,12,13,14,15); cell-rank count 4/16 (fraction 0.25; disagreement IDs 0,2,3,5,6,7,9,11,12,13,14,15). These are map-consistency descriptors, not identity accuracy.

Geometric intervals (tolerance 1e-10) were defined for all 16 rows in each view, with invalid=0 and violations=0. Mean `c-b`: naive I -0.0242436282, naive E -0.0347879367; cell I -0.0366819654, cell E -0.0340112268. Bounds are algebraic consistency checks and are not gates.

Sign diagnostics are global-sign invariant through the minority-energy and abs-best quantities. Mean negative-energy fraction / minority fraction / abs-best cosine / abs-best relative-L2 were naive 0.620211229 / 0.0266356458 / 0.946728708 / 0.294435629; cell-rank 0.499007374 / 0.0386469737 / 0.922706053 / 0.359444508. Undefined sign rows: 0 in both arms. Descriptive Spearman associations (n=16, no significance/generalization) are in `diagnostic.json`; e.g. C_fit vs MSE is 0.1059 naive and 0.1324 cell-rank, while C_image_I vs LPIPS is -0.2794 and -0.8176.

## Provenance and interpretation

All reused inputs except the expected handoff bookkeeping mutation are byte-identical to the pre-run manifest: method plan SHA `d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d`; checkpoint SHA `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`; public manifest `9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69`; evaluation truth `724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2`; naive/cell candidates, sources, comparison and old diagnostic hashes are recorded in [`integrity.json`](integrity.json). `evaluation_truth.pt` is bundled with x/labels/z; only z was consumed after validation, with `labels_consumed=false` disclosed. Raw updates and true-G were not deserialized or recomputed.

**FACT:** The corrected computation uses `target_z=abs(S_hat)`, separate per-arm maps and same-truth I/E pair tables. The old stage diagnostic signed-target C_fit values, cross-arm map agreement, and cross-map E deltas are superseded; stored image metrics remain valid.

**INFERENCE:** Both arms have high fit cosine to their own nonnegative target vectors, while the cell-rank arm has higher mean C_image in both frozen views. The small sample and map disagreement limit interpretation.

**PROPOSAL:** A future latent-G extension should use the isolated design in [`latent_g_design.md`](latent_g_design.md), with fixed registered whitening first, explicit cell-bound handling and a separately authorized data term.

**UNKNOWN:** The diagnostic cannot determine whether target construction, FIA optimization, or mixed effects caused image differences, and q,s do not identify a unique latent G. The original negentropy implementation remains overwritten to zero; no new objective or optimizer was activated here.
