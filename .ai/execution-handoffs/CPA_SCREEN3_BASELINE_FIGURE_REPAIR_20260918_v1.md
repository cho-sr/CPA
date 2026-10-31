# Correct the missing baseline and mislabeled truth in screen3 comparison figures

TASK_ID: CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: USER_REQUESTED_REPORTING_ONLY_FIGURE_REPAIR
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_SHOW_COMPARISON_CONTROL_20260918
PARENT_TASK_ID: CPA_SIGNED_BITS7654_RECOVERY_20260918
ACTIVATION_POLICY: SIDECAR_REPORTING_TASK_PRESERVE_RUNNING_PARENT
RESULT_REPORT_PATH: .ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/report.md

## Request and verified bug

The user requested: "이거 할 때 비교군도 같이 보여주면 좋겠는데". They are viewing the 6-bit final PNG with members 1, 3 and 4. This authorizes a comparison figure from existing saved results and the necessary reporting fix. It does not request another optimizer run or a new experimental control. Planner has not modified the active RUNNING parent handoff or any research source.

Recipient remains the existing designated Luna/xhigh Executor, thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID 14236, start 1789518279030 ms. Confirm actual identity/runtime and serialize this reporting work with your current task. Keep the approved bit sweep running. Do not adopt this sidecar as a replacement for the active research task, kill/restart processes, create an AI session or rebind shared peers. Record this task's claim/status separately under its execution-results directory.

Planner directly viewed:
quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/contact_sheet_b6_25000.png

The figure has only four columns. The first column contains ground truth, but its heading is center_25000. The true center reconstruction is absent. Source evidence: qacpa/signed_int8_screen3_evaluation.py:_contact_sheet initializes displayed=[truth], iterates names[1:], and labels columns with names[j]. The callers in both that file and signed_bits_screen3_evaluation.py pass names beginning with center_5000/center_25000, not a ground-truth sentinel. Thus the first candidate is skipped and ground truth is mislabeled. The same contract affects the 5000 and 25000 figures and callers reusing this helper, including the INT8 repair evaluator; inspect the actual caller chain.

The stored comparison.json includes separate center and member metrics/candidate paths/hashes and image correspondences. Numeric metric calculation iterates all candidate paths independently of plotting; this is a plotting defect, not grounds to rerun metrics or optimization. Verify that independence while implementing.

## Required result

For every completed bit width in the already authorized 8, 7, 6, 5, 4 series, produce corrected full-16-image figures:

- Final 25000 sheet: EXACTLY five columns: Ground truth | Naive dequantization baseline (25000) | Public rank 1 / member ID (25000) | Public rank 2 / member ID (25000) | Public rank 3 / member ID (25000).
- Screen 5000 sheet: EXACTLY ten columns: Ground truth | Naive dequantization baseline (5000) | all eight original member IDs 0..7 (5000).
- State the actual bit width and iteration count visibly. Baseline means same-bit center dequantization -> FastICA -> GD ICA 4000 -> same Direct FIA. Preserve original member IDs and frozen PUBLIC 5000 ranking; do not rename members as ranks or privately reorder them.
- Use readable, non-overlapping headings, with truth and baseline clearly separate. Preserve all 16 rows and reuse the existing per-candidate pixel-MSE Hungarian correspondence. Do not choose favorable images, recompute matching, average images, change pixel normalization, or score/rerank by private metrics.
- Include a short text/Markdown sidecar table with the four final reconstruction sets' EXISTING MSE/PSNR/SSIM/LPIPS values and baseline definition, for clear comparison. Explicitly flag the old PNGs as mislabeled in the new report without deleting them.

Use an isolated report-only renderer or adapter. DO NOT edit shared evaluator/runner/helper sources or registered code hashes while the active sweep depends on them; a small standalone renderer using stored metadata is sufficient. Do not hot-patch the live experiment. Preserve existing PNGs, comparison JSONs, candidate/state/selection files, registrations and source code byte-identical. Write new images under a new figures_with_baseline_20260918 subdirectory of EACH corresponding private evaluation directory, using filenames such as contact_sheet_b6_25000_with_baseline.png and contact_sheet_b6_5000_with_baseline.png. Record their source comparison, candidate hashes and output hashes in a small figure manifest. Do not overwrite an existing populated repair output without authentication.

Before private loading, authenticate the completed comparison's PUBLIC_GATE_PASSED evidence and all required public candidate files/hashes/shape/step/name bindings. Load only the already registered private x for plotting; disclose bundle deserialization separately and do not consume z/labels/updates. Reuse comparison.image_metrics[*].per_image exactly. If a required baseline or matching record is missing, report the missing artifact; never substitute truth or another candidate.

Known completed evaluation locations:
- INT8: quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918
- INT7: quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918
- INT6: quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918
- INT5/INT4: use their actual authenticated completed evaluation directories as the existing sweep finishes. Record pending widths honestly. Fold these figure exports into the current Executor's already registered completion/reporting path without a new polling loop or watcher. Do not delay the running optimization for this visualization request.

Run focused synthetic reporting coverage for distinct truth/baseline/member tensors: assert final five and screen ten column identities, order, labels, all 16 rows and existing correspondence reuse. Inspect at least the corrected 6-bit PNG to confirm truth, baseline and three members are all present with correct headings. No model inference, LPIPS rerun, new Hungarian matching, optimizer steps or private-feedback experiment is needed. Keep this reporting work on CPU.

Report the root cause, fixed figure paths, validation and preserved numeric results/source artifacts. The requesting Planner's existing thread is 01a0a799-d62c-7ab0-9c2a-628acf6d2c95. Return one concrete completion delivery with corrected available figure paths (and pending widths, if any); after the parent sweep finishes, include remaining corrected figures in its existing final report. No additional experiment or model-based monitoring follows. Planner ends after a verified queue receipt.
