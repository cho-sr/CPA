# Export the corrected INT8/INT7/INT6 final comparisons as slide-sized PNGs

TASK_ID: CPA_SCREEN3_SLIDE_PNG_876_20260918
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: USER_REQUESTED_REPORTING_ONLY_SLIDE_EXPORT
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_SLIDE_SIZED_PNG_8_7_6_20260918
PARENT_TASK_ID: CPA_SIGNED_BITS7654_RECOVERY_20260918
ACTIVATION_POLICY: SIDECAR_REPORTING_TASK_PRESERVE_RUNNING_PARENT
RESULT_REPORT_PATH: .ai/execution-results/CPA_SCREEN3_SLIDE_PNG_876_20260918/report.md
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/slide_png_8_7_6_20260918

## Authorized task

The user said: "좋은데 슬라이드 사이즈로 이미지 다시 만들어줘 8비트 7비트 6비트 png" after reviewing corrected final baseline comparisons. This is a layout/export request for frozen 25000-step results of exactly bits 8, 7 and 6. Reuse the completed baseline figure repair's authenticated data and matching. No experiment, metric evaluation, source separation, model inference or new image correspondence is requested.

Deliver three overview PNGs, one per bit, each EXACTLY 3840x2160 pixels (16:9), plus four detailed 16:9 PNGs per bit (four consecutive truth images on each page, covering all16 once). Planner has already told the user this arrangement. Put all15 PNGs in one convenient new private output directory with a ZIP for download and a concise index.md. Do not produce a PowerPoint file or add unrelated plots. The three overview files satisfy the user's per-bit request; detailed pages preserve readability.

Existing Executor remains thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, start1789518279030ms, Luna/xhigh. Confirm identity/runtime and handle as a separate READY reporting sidecar; do not replace the active research handoff or interrupt the running bit sweep. No new AI session or shared peer rebinding.

## Sources and semantics

Reuse functions for authentication, private-x loading and existing correspondence from the already completed isolated renderer:
.ai/execution-results/CPA_SCREEN3_BASELINE_FIGURE_REPAIR_20260918/render_baseline_figures.py
Keep that source and its existing outputs unchanged; write any new layout code only under this task's execution-results directory. Regenerate the scientific figure directly from the authenticated saved image tensors with ordinary plotting tools. Do not generatively redraw, enhance, stretch, crop away image content or change image appearance.

Exact evaluation directories:
- INT8: quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918
- INT7: quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918
- INT6: quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918

Read each comparison.json, authenticate its completed public gate and candidate hashes, and reuse its exact image_metrics[*].per_image matching. Load the registered private x only after the public authentication. Keep labels/z/true-updates unconsumed; disclose bundle deserialization as before. Preserve original truth row order and original member IDs. Expected selections: b8 [1,7,5], b7 [0,5,3], b6 [1,3,4], verified from the corresponding stored selection.

For every example, show the same five roles together:
Ground truth | Naive dequantization baseline | Public rank1 / original member ID | Public rank2 / original member ID | Public rank3 / original member ID.
All reconstructed images are at25000 steps. Baseline means same-bit center dequantization -> FastICA -> GD ICA4000 -> Direct FIA. Clearly distinguish truth from baseline. The existing long portrait PNG had this fixed already; do not reintroduce the original missing-baseline bug.

## Presentation layout

- White or very light neutral background, restrained typography, no decoration competing with the images.
- Each title includes the bit width and25000 steps. Short, readable column headings with line breaks as needed; member IDs must remain legible. Avoid displaying filesystem/internal task labels in the figure.
- Overview: all16 examples, preferably two side-by-side panels of eight rows and five columns, each panel repeating its own headings. Row labels clearly identify examples01-08 and09-16, in order. Preserve square image aspect and enough margins for unclipped text. Do not simply squeeze the tall old PNG into16:9.
- Details: four pages per bit, rows01-04,05-08,09-12,13-16; four rows and five columns, with much larger images. Label page1/4 etc. Use exactly the same rows and correspondence as the overview. No selected "best" subset.
- Use a concise footer such as "Rows matched by pixel error; source identity is not verified" because the user just asked why a leopard-like result could appear on the fish row. This explains the figure's interpretation without changing matching.
- Every exported PNG must remain exactly3840x2160. Avoid tight-bounding-box export that silently changes the canvas ratio. A standard plotting canvas at the desired dimensions is appropriate. Do not alter contrast, normalize each image differently or apply sharpening.
- Suggested files: int8_25000_overview.png; int8_25000_page01.png through page04.png; corresponding int7 and int6 files. ZIP: slide_png_8_7_6_20260918.zip. Store the ZIP in the same new private root, containing the PNGs and index; never include private tensor bundles in the ZIP.

## Validation and finish

Use CPU-only rendering and preserve all original sources, figures, candidates, registrations, comparison JSONs, metrics and the active parent handoff. No optimizer steps, metric recomputation or matching change. No broad test suite or experiment is needed for a presentation export.

Verify actual PNG dimensions, file counts, row coverage, 5-role identity/order, bit/member labels and source hashes. Open at least the6-bit overview and a detailed page to inspect legibility and ensure labels do not overlap. If the overview is dense, the detailed pages remain the readable companion; do not silently omit rows. Record comparison/candidate hashes, row ranges, filenames, output hashes and zero optimizer/metric/matching work in a small manifest/report. Leave original files byte-identical.

After the artifacts exist, send one final completion with the three overview paths, ZIP path and per-bit detailed page folders to the existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95. Mention the originating user thread01a0a7ab-fa58-7d22-ac52-3247b1dd2255 in context as before. No additional work, polling or research follows this reporting request. Planner ends after a verified queue receipt.
