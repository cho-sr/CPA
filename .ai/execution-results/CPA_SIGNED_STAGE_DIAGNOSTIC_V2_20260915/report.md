# CPA_SIGNED_STAGE_DIAGNOSTIC_V2_20260915

Status: **COMPLETED**. This report covers the authorized implementation, focused tests, and exactly one saved-result diagnostic. No CPA/FIA/AP5 rerun, new arm, seed, ranking, sign flip, or scientific-condition change occurred.

## Implementation and validation

- Added [`signed_stage_diagnostics.py`](../../../quantization_aware_cpa/qacpa/signed_stage_diagnostics.py) and [`test_signed_stage_diagnostics.py`](../../../quantization_aware_cpa/tests/test_signed_stage_diagnostics.py).
- The CLI validates the v2 plan SHA256 before private deserialization, authenticates completed naive/AP5 artifacts and frozen image/embedding maps, validates candidate/source shapes and `target_z == abs(S_hat)`, then loads only `z` from the bundled `evaluation_truth.pt` with `labels_consumed=false` disclosure.
- Features use the existing `official_cpa/src/vgg16.py` VGG16 checkpoint, frozen/eval FP32, ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` exactly once, followed by features -> avgpool -> flatten. No resize, augmentation, extra clamp, autograd, or selection was used.
- Focused command exit 0: **56 passed, 240 warnings**. Evidence: [`tests.log`](tests.log), [`integrity.json`](integrity.json).

## Saved-result run

The exact handoff command ran once in a fresh process (PID 24344), with `device=cuda`, `eval_microbatch=1`, `max_evaluation_seconds=600`, and `max_cuda_allocated_gib=6`. The execution record reports stage exit 0 and 32 feature evaluations; the model-free callback queued its completion notice (callback process exit is recorded as 1 despite the queued receipt and stage exit 0). No retry was made.

- Numerical output: [`diagnostic.json`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2/diagnostic.json)
- Rows: [`all32_per_candidate.csv`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2/all32_per_candidate.csv) (16 naive + 16 cell-rank), [`paired16_by_truth.csv`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2/paired16_by_truth.csv)
- Plot: [`stage_diagnostic_v2.png`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2/stage_diagnostic_v2.png)
- Registered configuration: [`registered_plan.json`](../../../quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/stage_diagnostic_v2/registered_plan.json)

Runtime was 0.412337 s (naive) and 0.127123 s (cell-rank); peak allocated CUDA memory was 663,822,336 bytes, starting at 554,741,760 bytes (delta 109,080,576 bytes), below 6 GiB. All 16 rows in each arm had defined cosines and zero geometric-interval violations.

## Measurements

`C_fit` is cosine(h, abs(S_hat)); `I` uses the frozen pixel-MSE map and `E` uses the frozen embedding map. Values below are mean, median, and min/max over 16 rows.

- Naive: C_fit mean -0.211733097; median -0.838658401; range [-0.965298319, 0.985440718] (n=16); C_target_I mean 0.00468734536; median -0.0194663462; range [-0.544177316, 0.711173033] (n=16); C_image_I mean 0.117020653; median 0.0380171734; range [0.0123556402, 0.676861213] (n=16); C_target_E mean 0.00972657567; median -0.023600964; range [-0.643562243, 0.734000433] (n=16); C_image_E mean 0.276817226; median 0.0800440801; range [0.0123556402, 0.677559936] (n=16).
- Cell-rank: C_fit mean -0.00326143893; median -0.0159988748; range [-0.970389704, 0.975056987] (n=16); C_target_I mean -0.00325324278; median 0.00363521515; range [-0.778661101, 0.762123812] (n=16); C_image_I mean 0.199139135; median 0.048085712; range [0.00914818993, 0.730050066] (n=16); C_target_E mean -0.00502156934; median -0.0198202946; range [-0.778661101, 0.762123812] (n=16); C_image_E mean 0.310750169; median 0.0822211048; range [0.0139817965, 0.730050066] (n=16).

Reused image metrics from the completed comparison (no metric recomputation):

- Naive: MSE mean 0.0781093468; median 0.0685799431; range [0.0394393299, 0.210959022] (n=16); PSNR mean 11.5173321; median 11.6383378; range [6.75801897, 14.0407047] (n=16); SSIM mean 0.28489399; median 0.254379776; range [0.150888179, 0.433928693] (n=16); LPIPS mean 0.822924603; median 0.818700075; range [0.72634238, 0.894607663] (n=16).
- Cell-rank: MSE mean 0.0779298795; median 0.0705553479; range [0.0411402281, 0.207298026] (n=16); PSNR mean 11.5045254; median 11.5149773; range [6.83404834, 13.857333] (n=16); SSIM mean 0.281656965; median 0.244999805; range [0.128168008, 0.502146207] (n=16); LPIPS mean 0.794168193; median 0.796907932; range [0.66968286, 0.877494335] (n=16).

Frozen-map agreement is image 3/16 (disagreement 13/16) and embedding 8/16 (disagreement 8/16), with all IDs retained. These fractions are map agreement descriptors, not identity accuracy or success rates.

Geometric consistency had 16/16 defined rows and 0 violations for each arm/view. Mean observed `c-b` was naive I 0.112333308, naive E 0.26709065, cell I 0.202392378, cell E 0.315771738; tolerance was 1e-10. Bounds are algebraic consistency checks, not optimization or principal-angle gates.

Sign-energy summaries (orientation-dependent negative-energy fraction; global-sign-invariant minority fraction and abs geometry) were:

- Naive: negative fraction mean 0.620211229; minority fraction mean 0.0266356458; abs-best cosine mean 0.946728708; abs-best relative-L2 mean 0.294435629; undefined 0.
- Cell-rank: negative fraction mean 0.499007374; minority fraction mean 0.0386469737; abs-best cosine mean 0.922706053; abs-best relative-L2 mean 0.359444508; undefined 0.

Descriptive Spearman associations (n=16, no significance or generalization claim) are retained in `diagnostic.json`. Examples for C_image_I versus MSE/LPIPS: naive 0.4412 / -0.2794; cell-rank 0.1265 / -0.8176. Ties/constant handling and all undefined reasons are recorded per row.

The historical stored comparison remains unchanged: cell-rank minus naive embedding cosine +0.0331562, MSE -0.000179467, PSNR -0.0128067 dB, SSIM -0.00323702, LPIPS -0.0287564, and true-G relative error -0.0985553. This diagnostic did not recompute true-G or use labels and does not establish a causal explanation.

## Provenance and interpretation

Pre/post SHA256 checks show unchanged reused inputs: method plan `d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d`, naive candidate `fe683f68013896a407a38130ee54ef9dc799a9ca1abf7c4f08efeefea7d91751`, cell candidate `700beaa7b7da8565da68bb782ea67738542695334d09a2910e29f83bf29ab9d6`, naive/cell sources, comparison, public manifest, evaluation truth and checkpoint all match the pre-run manifest. The method-plan hash is `d24c271ae8461bdad3061c957b53f247cf852c7ba1bbf1d63933205511f0d14d`; checkpoint hash is `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`.

**FACT:** The run measured the stated cosine triples, frozen-map joins, geometric intervals, sign-energy quantities and reused image rows over the existing 32 final candidates.

**INFERENCE:** Cell-rank rows have higher mean C_image in both map views and lower mean LPIPS/MSE in the already stored comparison; these are descriptive paired-arm observations only.

**PROPOSAL:** Any follow-up interpretation or new experiment requires a separate authorization and must preserve the registered maps and original CPA/FIA settings.

**UNKNOWN:** These diagnostics cannot identify whether target construction, FIA optimization, or mixed effects caused the observed differences; map disagreement and global sign symmetry prevent treating either view as identity truth. Loading the bundled truth file may materialize `x`/`labels`; only `z` was consumed and `labels_consumed=false`.

The active handoff is now marked COMPLETED with the result files and one Planner delivery receipt recorded in the task directory.
