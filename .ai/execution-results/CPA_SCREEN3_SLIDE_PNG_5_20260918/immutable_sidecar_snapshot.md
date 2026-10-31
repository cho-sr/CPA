# Export completed INT5 final results in the existing slide PNG format

TASK_ID: CPA_SCREEN3_SLIDE_PNG_5_20260918
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: USER_REQUESTED_REPORTING_ONLY_SLIDE_EXPORT
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_SLIDE_SIZED_PNG_5_20260918
PARENT_TASK_ID: CPA_SIGNED_BITS7654_RECOVERY_20260918
ACTIVATION_POLICY: SIDECAR_REPORTING_TASK_PRESERVE_RUNNING_PARENT
RESULT_REPORT_PATH: .ai/execution-results/CPA_SCREEN3_SLIDE_PNG_5_20260918/report.md
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/slide_png_5_20260918

The user requested "5비트 결과 png 로 만드는데 슬라이드 사이즈로". Produce the same format as the completed INT8/7/6 slide export, now for INT5 only. This request explicitly authorizes rendering from the frozen results. No additional research or permission request is needed.

Use existing Executor thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, start1789518279030ms, Luna/xhigh after actual identity/runtime verification. Track this READY reporting sidecar separately. Keep the active bit4 sweep, its handoff and callbacks intact; do not restart or monitor it or create another AI session.

Completed, evaluated, PUBLIC_GATE_PASSED source:
quantization_aware_cpa/results_private/signed_int5_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json
Frozen public selection: [5,7,6]. Final candidate names: center_25000, rank1_member_05_25000, rank2_member_07_25000, rank3_member_06_25000.

Reuse the proven layout and authentication helpers without altering the old scripts:
- .ai/execution-results/CPA_SCREEN3_SLIDE_PNG_876_20260918/render_slide_pngs.py
- .ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/render_baseline_figures.py
The former main() is hardcoded to8/7/6 and the old output root: do not invoke it unchanged. Use a small isolated INT5 adapter/new task-local entrypoint that calls its existing functions with the correct source/output/selection. Do not modify shared or historical source, registration, metrics, candidate, matching, figure or active handoff files.

Deliver under PRIVATE_OUTPUT_ROOT:
1. int5_25000_overview.png: all16 rows, same overview layout as the previous slide images.
2. int5_25000_page01.png through page04.png: four consecutive examples each (01-04,05-08,09-12,13-16).
3. slide_png_5_20260918.zip containing exactly the five PNGs and index.md, no tensor bundles.
4. index.md and a small slide_manifest.json with source/output hashes and row/role coverage.

All five PNGs must be exactly3840x2160 pixels,16:9. Roles: Ground truth | same-bit naive dequantization baseline | Public rank1 / member05 | Public rank2 / member07 | Public rank3 / member06. All are25000-step reconstructions. Preserve baseline inclusion, original truth order, stored per-image Hungarian correspondence, original member IDs and square image content. Match the previous typography, colors, headings and interpretation footer. No image enhancement, crop, new matching or selection by visual quality.

Authenticate the stored public gate/candidate hashes before registered private-x-only plotting, using the existing strict helper. Reuse frozen numeric metrics if included in the index. No labels/z/updates consumed, optimizer step, model inference, LPIPS run or metric recomputation. Use CPU-only ordinary plotting tools and new output paths. If an output already exists, authenticate it before reuse and do not overwrite blindly.

Verify file count, exact dimensions, member/bit labels, full16 coverage and ZIP contents. Inspect the INT5 overview and one detail page for legibility, image identity and unclipped headings. Reuse existing layout validation; no broad tests or new experiment. Save concise completion/integrity evidence under RESULT_REPORT_PATH.

Send one completion containing the overview path, ZIP path and detail-page directory to existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, including originating user thread01a0a7ab-fa58-7d22-ac52-3247b1dd2255 as before. End after delivery. Planner ends after queue receipt; no polling or added work.
