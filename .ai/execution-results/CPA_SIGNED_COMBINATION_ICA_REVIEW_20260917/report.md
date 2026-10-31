# CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917

Status: **COMPLETED**.

This is one canonical-runtime reevaluation of the four frozen 16-image candidate sets. No optimizer, FC replay, separator, source/FIA run, new arm, tuning, download, or second evaluation was performed.

## Runtime repair and execution

- Parent total wall time: 2330.788675099960528 s of the registered 2700 s cap; remaining before review: 369.211324900039472 s.
- Review wrapper wall time: 44.98513970000204 s; registered added evaluation/finalization limit: 300 s; within limit: True.
- The wrapper recorded runtime before setup and after configure_runtime(42, threads=1). The pre-setup snapshot describes this review process only.
- The unchanged evaluator was invoked exactly once after the public artifact gate path and canonical runtime setup.

| Parent stage | Stored cumulative wall seconds | Independent duration from timestamps |
|---|---:|---:|
| SEPARATE | 33.069726 | 33.068048 |
| INVERT | 2160.407650 | 2127.300794 |
| REFINE | 2308.128908 | 147.659951 |
| PRIVATE_EVALUATION | 2330.781256 | 21.574583 |

The stored parent values after SEPARATE are cumulative wrapper elapsed times. The independent durations above are derived from each stage's own start/end timestamps.

## Provenance and private boundary

- Public gate remained strict and passed before private loading.
- Private truth was consumed as x only; labels_consumed=false and true_updates_consumed=false. Bundled fields may materialize during safe tensor loading.
- The same one-time pixel-MSE Hungarian correspondence was reused for MSE, PSNR, SSIM, LPIPS and the aligned figure; all 16 images were included.
- Before/after hash catalog is in C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\before_after_integrity.json. Parent metadata, public inputs, candidates, protected/shared sources, checkpoint and LPIPS weights remained byte-identical: True.

## Four-set metrics

| Set | MSE | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| original_separator_direct | 0.0781093468343798 | 11.5173321456311 | 0.284893989708307 | 0.822924625128508 |
| original_separator_cell | 0.078160877077264 | 11.52174236431 | 0.282700870678415 | 0.808157000690699 |
| combination_separator_direct | 0.0776844093863262 | 11.5249464319988 | 0.29322348980258 | 0.809157475829124 |
| combination_separator_cell | 0.077736043680349 | 11.5214056622864 | 0.284907516904722 | 0.800327993929386 |

The three predeclared contrasts and the per-image directional counts are recorded in review_results.json. They are descriptive for this fixed batch and seed. The adopted separator package is not an equal-compute or single-prior causal ablation.

## Control residuals

Control reproduction status: **PASSED** with unchanged absolute tolerance 2e-8.
Correspondence equality, all four per-image residuals and LPIPS float32-spacing units are recorded in review_results.json.
- original_separator_direct vs x0: correspondence_equal=True; max absolute residuals={'mse': 0.0, 'psnr_db': 0.0, 'ssim': 0.0, 'lpips': 0.0}.
- original_separator_cell vs B_H1_high: correspondence_equal=True; max absolute residuals={'mse': 0.0, 'psnr_db': 0.0, 'ssim': 0.0, 'lpips': 0.0}.

If both controls pass, the canonical runtime restores the archived comparison. This bundled setup change does not prove that CPU thread count, seed, or an individual backend flag alone caused the earlier discrepancy.

## Endpoint feasibility

The saved endpoint profiles were reused without replay. Exact feasibility still means zero fixed-observed-scale code mismatches and zero bounded-cell violations across all six FC tensors.
| Set | Exact feasible | Code mismatches | Bounded-cell violations | Cell loss |
|---|---:|---:|---:|---:|
| original_separator_direct | false | 1210569 | 1210573 | 22.9916826474 |
| original_separator_cell | false | 996857 | 996857 | 0.204078117165 |
| combination_separator_direct | false | 1225207 | 1225212 | 22.8891772005 |
| combination_separator_cell | false | 995058 | 995058 | 0.192871173711 |

The new final cell endpoint remains exact_feasible=false independently of image-metric reproduction and is not a recovery-success claim.

## Saved optimization observations

- Cell loss decreased from 22.8891772 to 0.192871174.
- Feature-target cosine was approximately 0.930897 at the start and 0.680713 at the final endpoint.
- The initial weighted cell/base gradient-norm ratio was approximately 9508.7; the final gradient cosine inferred from saved norms was approximately -0.9542.
These observations support conflicting objectives in this run but do not uniquely establish the cause of poor source recovery.

## Review artifacts

- Review summary: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\execution_summary.json
- Review results: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\review_results.json
- Review registration: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\review_registration.json
- Runtime before setup: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\runtime_before_setup.json
- Runtime after setup: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\runtime_after_setup.json
- New comparison: C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\signed_combination_ica_B16_T10_20260916\review_20260917\comparison.json
- New report: C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\signed_combination_ica_B16_T10_20260916\review_20260917\report.md
- New figure: C:\Users\조상제\Desktop\CPA\quantization_aware_cpa\results_private\signed_combination_ica_B16_T10_20260916\review_20260917\all16_comparison.png
- Parent bytes archived under: C:\Users\조상제\Desktop\CPA\.ai\execution-results\CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917\before

No additional attempt is authorized or implied by this report.

## Completion delivery

The hidden review wrapper completed before the callback could reserve its wait record. The callback launch failed before reservation because the temporary peer file was written with a UTF-8 BOM; the receipt and stderr are preserved in callback_launch_receipt_bom_failure.json and completion_callback_stderr.log. The peer file was restored byte-for-byte. The completed result was delivered once directly to the verified existing Planner, with no rerun or additional evaluation.
