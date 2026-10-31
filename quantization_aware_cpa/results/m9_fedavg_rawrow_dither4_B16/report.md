# FedAvg direct raw-update-row FIA (no CPA)

Completed all 80 images: VGG16, n=16, T=10. This experiment reuses exactly the cached observation used by the M8 CPA reference.

| Method | Images | LPIPS | PSNR | SSIM | Source absolute cosine > 0.9 |
|---|---:|---:|---:|---:|---:|
| Direct upstream CPA + FIA (M8) | 80 | 0.556121 | 21.937 | 0.2980 | 53/80 |
| Raw update rows + abs + FIA (M9) | 80 | 0.706844 | 21.353 | 0.2300 | 1/80 |

The M9 input is the elementwise absolute value of selected observed update rows. These are mixed-update vectors used as FIA targets, not separated individual-image embeddings.
For each batch, 16 of the 4096 rows are chosen without replacement by CPU torch.randperm with seed=base_seed+batch_index, fixed before FIA. Selection order and zero rows are preserved; no L2 normalization or data-dependent row replacement is applied.
No CPA/ICA, debias, MP truncation or diagnostic, LS mixing-matrix estimate, embedding reoptimization, oracle fill, or new client training is used in M9.
FIA uses the same original Direct implementation, fixed iteration budget and TV coefficient as M8. Original features/images are consulted only after FIA for correspondence and evaluation.
The observation is the existing per-row 4-bit-interval uniform-dither model, not a transmitted integer 4-bit codec. This is raw-row feature inversion, not dequantization-only image reconstruction.

Mean paired LPIPS difference (raw rows minus CPA): +0.150723. This is a descriptive comparison of two feature-input construction methods; it does not isolate every implementation choice or establish a general CPA contribution.
Source absolute cosine is computed between the signed raw update row and the original feature, using the same post-FIA correspondence criterion as M8. The FIA target itself is abs(raw row); its cosine is stored separately as fia_target_cosine. Source cosine > 0.9 is not an LPIPS success threshold and does not establish successful per-image source separation for raw rows.
No invalidated historical no-CPA control, normalized signal percentage, floor-based threshold, or statistical significance claim is used.

FIA runtime: 9769.6 seconds; total batch invocation runtime: 9775.6 seconds.
All batch identities, output hashes, frozen input hashes, all80 image identities, and native-resolution PNG tile pixels were verified before completion.

## Figures

All80 figures contain every image in original index order. Random12 indices were fixed with seed42 before results; not cherry-picked. All panels preserve 224 x 224 pixels per image.

![All80 CPA and raw-row comparison](all80_cpa_vs_rawrow_landscape.png)

![Random12 not cherry-picked](random12_not_cherry_picked.png)

![All80 original and raw-row reconstruction](all80_reconstruction_landscape.png)

Per-image observations, metrics and correspondence are in `per_image_metrics.csv`; paired differences are in `paired_cpa_comparison.csv`; exact grid positions and pixel hashes are in `figure_manifest.json`.
