# CPA_SIGNED_BASELINE_B16_T10_20260914 v1 — Executor report

[FACT] COMPLETED. New signed observation, INT4 naive, FP32 Reference, and all16 offline evaluation finished.

| Arm | PSNR dB | SSIM | VGG LPIPS | Embedding cosine | CUDA peak allocated B |
|---|---:|---:|---:|---:|---:|
| reference | 12.391145 | 0.346884 | 0.459648 | 0.995274 | 3593020928 |
| naive | 11.517332 | 0.284894 | 0.822925 | 0.311605 | 3593020928 |

[FACT] Naive minus reference: {"mean_psnr_db": -0.8738133125594807, "mean_ssim": -0.06198987517853449, "mean_lpips": 0.3632761724293232, "mean_mse": 0.012518882126713121}

[FACT] Same raw local update, frozen Conv/all FC, batch00 B16 T10 seed42; exact ordinary SGD/replay agreement on all six raw tensors. Signed CPU FP32 per-tensor scale, nearest-even; q+7 nibble payload with reserved 15. Original affine observations preserved.
[FACT] Both arms use raw Delta/raw sq without -Delta/lr or T/B transforms; official eig whitening and CPA 25,000, abs(S_hat), Direct FIA 25,000. Separate input fingerprints/caches. Identical FIA initial tensor hashes and actual numeric runtime flags.
[FACT] Public-only naive process has no private truth/raw-update API. Reference receives raw Delta only. Separate offline process loads private truth only after both final artifacts are complete and hashed. No oracle matching/scaling before FIA.
[FACT] One pixel-MSE Hungarian per arm for PSNR/SSIM/VGG LPIPS on all16; distinct embedding cosine Hungarian. Matching used only for final evaluation; paired differences are aligned by truth image in comparison.json.
[FACT] CPU gates: 158 passed; 0 failures/errors/skips. Exact execution commands, timestamps, exit statuses and stdout are in *_execution.json / *_attempt_1.log.
[FACT] Planner review 01 added offline identity/input/source binding and private-output guards, validated by focused CPU rejection tests. No collection or optimized arm was rerun; earlier execution source hashes are preserved with this explicit offline-only revision.
[FACT] New modules: signed_baseline_observation.py, signed_baseline.py, signed_baseline_evaluation.py; new tests: test_signed_baseline.py. Existing full_update_evaluation.py only adds include_labels=False while preserving the original default path; regression gates passed. Existing quantizer/cell_utils/official CPA/FIA/collector/STATE/RELAY unchanged.
[FACT] Communication: {"code_bytes": 61821428, "header_bytes_excluding_scale": 96, "manifest_bytes": 6139, "packet_file_bytes": 61821548, "scale_bytes": 24, "total_file_bytes": 61827687}
[FACT] Raw source/objective diagnostics remain distinct from actual MI/negentropy. MI and actual NE are not computed. Quantization/scale fit is a diagnostic on supplied observation matrices, not an image-update constraint.
[FACT] adaptive_avg_pool2d_backward_cuda emitted a nondeterminism warning under the upstream adapter's warn_only=True setting. Both arms retain identical settings; complete determinism is not claimed.
[UNKNOWN] No success threshold, statistical significance, generalization or causal mechanism established by one batch/seed. This is a newly approved signed per-tensor experiment, not a reproduction of the old affine per-row or -Delta/lr baseline.
[PROPOSAL] No additional run was scheduled. Planner can review the saved evidence.

Evaluation outputs: quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/evaluation/report.md, comparison.json, all16_comparison.png, all16_compact.png.
Public bundle and naive candidates: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/.
Private raw/reference/truth and offline artifacts: quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/.

[FACT] Final one-time Planner delivery: FAILED. codex queue returned no rollout found for the bound Planner thread 01a09ac2-0e8d-7f90-94a8-c45be99f4343 (exit 1). No message was queued; no automatic resend or replacement session was attempted. Scientific job remains COMPLETED. Details: planner_final_delivery.json and planner_final_stderr.log.
