# Complete scientific comparison review using frozen candidates and the archived evaluation runtime

TASK_ID: CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: EVALUATION_RUNTIME_REPAIR_AND_FROZEN_RESULT_REVIEW
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_CONTINUE_SCIENTIFIC_COMPARISON_AND_PRIOR_FINALIZATION_SCOPE_20260917
PARENT_TASK_ID: CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916
REVIEW_REFERENCE: PLANNER_STATIC_EVALUATOR_RUNTIME_AND_SAVED_METRIC_REVIEW
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917/execution_summary.json
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/review_20260917
NEW_OPTIMIZER_STEPS: 0

## Authorization and exact scope

The user authorized the parent experiment with "실험 ㄱㄱ" and "루나에게 전달해줘". Its registered scope included final comparison and routine evaluation/reporting repairs from frozen artifacts. After the completed-with-review notification, the user requested "과학적 비교 검토". Planner identified the runtime omission below and described a single frozen-candidate reevaluation. The user briefly interrupted with an unfinished alternative; no job was sent, and then explicitly said "이어서 과학적 비교". Continue the review and ordinary evaluation completion without asking again.

This task permits exactly ONE offline evaluation of the existing four sets of 16 saved images under the archived reference evaluation runtime, plus necessary small mocked adapter validation and a superseding scientific-review report. No separator run, feature inversion, FC replay/endpoint regeneration, image optimizer step, new candidate, label change, new seed/arm, tuning, extra runtime sweep, private-feature diagnosis, or research rerun. Keep all scientific settings and the predeclared 2e-8 mean-metric control-reproduction tolerance unchanged. Do not run the parent wrapper or historical finalizer (the latter includes replay); use only its runtime setup as the reference.

Existing designated Executor: Luna/xhigh thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID 14236, process creation 1789518279030 ms. Return to existing Planner thread 01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID 24972, creation 1789518271364 ms. Verify actual role/runtime/identity and this READY handoff hash before claim. Do not change model/bindings or create another AI session.

## Static evidence already established

1. Both archived control candidates authenticate. Matching is identical for all 16 images in both controls. Recorded LPIPS metric provenance (model/calibration hashes, versions, CPU FP32 and normalization) is identical between the previous finalized comparison and the parent comparison.
2. The direct control's LPIPS differs in only five images: candidate indices 0, 2, 3, 5, 10. Differences are -1, -1, -1, -2, -1 units of float32 spacing respectively; the largest absolute per-image difference is 1.1920928955078125e-7. Mean difference is -2.2351741790771484e-8, slightly beyond 2e-8. The cell control's mean difference is -3.725290298461914e-9 and passes. Other metrics differ only around 1e-15 or less; correspondence is unchanged.
3. signed_e2e_nofia_finalizer.py:345-346 calls configure_runtime(42, threads=1) before its computation, and its private LPIPS evaluation occurs later in the SAME process (:447-460). That helper seeds Python/NumPy/Torch, sets one intra-op CPU thread, disables TF32/benchmark, and enables deterministic algorithms.
4. signed_combination_ica_evaluation.py:evaluate/main never calls that runtime setup. The parent wrapper launches the evaluator as a fresh Python process. Torch thread/algorithm settings applied in the separate public child processes do not configure this evaluator process. The parent did not record evaluator thread settings, so its historical actual thread count is UNKNOWN; do not invent it.
5. The missing canonical evaluation setup is a concrete difference. Whether restoring it resolves the discrepancy is still unverified. The fact that differences are tiny does not itself authorize passing the gate or loosening tolerance.

## Frozen artifacts and pins

Scientific parent report directory: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916.
Original experiment registration.json SHA256: b5aed69767b22a2074f2fbd76064ea34076fc2283789142fa5a781573a53c81a.
Original private comparison: quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/comparison.json; SHA256 d765fc3c9292bd41df2ac8a286165850906b5f22b5b7ac9df50f89de6b341369.
Archived reference comparison: quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916/comparison.json; SHA256 545f3d422ab54121826abd412065f53cc74d37c4d4b346cdbd633042e3b3704e.

Four candidate hashes, unchanged from the parent's registration and public gate:
- original_separator_direct: fe683f68013896a407a38130ee54ef9dc799a9ca1abf7c4f08efeefea7d91751
- original_separator_cell: b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a
- combination_separator_direct: 4efc29ad08b1ea8b0cee6991de6f185839708c2c72aaafb4fedfb307b5e680ce
- combination_separator_cell: 01c3b2b9289c055feb9c1a0bf496855fb7232a0bf406ba05bd161e679535d708

Original new evaluator signed_combination_ica_evaluation.py SHA256: dec3fd4189db5665378d8be80f724472c14f64b025ce6f38d95d9364394f4dd8. Preserve it and all original core/runner/experiment/wrapper bytes and code hashes. The original strict validate_public_gate must continue to pass against the original registration; never rewrite that registration to accept changed code.

All public inputs, actual-update profiles, source/target/candidate tensors, labels, logs, protected/shared sources and historical private results remain byte-identical. Archive the parent's completed handoff/report/summary/audit and original comparisons under this task's before/ directory; record before/after hashes for all consumed metadata, candidates, sources, checkpoint, packet and evaluator weights. Do not rewrite the original NEEDS_REVIEW result; new records explicitly supersede it if justified.

## Minimal finalization-only implementation and execution

Create an isolated task-local wrapper/finalizer and, only if needed, focused mocked tests under the new task results directory. Do not modify the existing scientific modules. The helper uses C:/miniconda/envs/DLG/python.exe with the existing project paths and imports the unchanged evaluator. Before any evaluation, establish the existing required environment and invoke qacpa.deterministic_hybrid_setup.configure_runtime(42, threads=1). Record actual runtime before setup and after setup, including intra/inter-op threads, deterministic/warn-only flags, CPU backend flags, relevant environment and versions. The pre-setup snapshot describes THIS new process only; it cannot reconstruct the old process.

Run the existing evaluator.evaluate with:
- registration_path = parent's unchanged registration.json;
- public_output = quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916;
- private_truth = unchanged signed_int4_baseline_B16_T10/observation/evaluation_truth.pt under results_private;
- output = the NEW review_20260917 private subdirectory named in the header, which must not already exist.

Validate/hash the new wrapper separately in a review registration written before evaluation. This runtime wrapper does not bypass the original evaluator's public gate or change any of its logic, hashes, matching, metric parameters or control tolerance. The new scientific output keeps the parent scientific task ID; the outer review report records this review task ID, original and new comparison hashes, runtime repair provenance and zero optimizer/replay steps.

If a focused mock test is needed, verify setup precedes evaluation/private loading, the original strict gate remains active, paths point only at frozen artifacts, and the evaluator is invoked at most once. Do not repeat the full unchanged test suite. The real evaluation is CPU metric computation only; no model training or FC update replay. Preserve x-only consumption, labels_consumed=false, true_updates_consumed=false and truthful disclosure of bundled fields materialized. Use the same MSE Hungarian correspondence once per set and reuse it for all metrics and the figure. No private feedback to attack.

## Review criteria and report

- Both archived controls must reproduce all four registered mean metrics within the UNCHANGED 2e-8 absolute tolerance. Also report per-image differences, correspondence equality and the float32 spacing differences. If the canonical-runtime evaluation passes, report that restoring the archived runtime resolved the reproduction failure. A single bundled setup change does not separately prove CPU thread count, seed or a specific backend was the sole cause.
- If it fails, retain NEEDS_REVIEW and the precise residuals. No searching over threads/precision/seeds/backends, rounding the metric to pass, tolerance extension or extra evaluation attempt. Finish with an explicit unresolved finding.
- Report the four-set metrics and the predeclared three comparisons: new direct versus old direct; new cell versus new direct; new cell versus old cell. State magnitudes and per-image directions, and whether the direction changes from the original evaluation. Conclusions apply only to this batch/seed and the adopted separator package, not an equal-compute or individual-prior ablation.
- Reuse the existing exact endpoint profiles without replay. The new final endpoint has 995058 code mismatches and bounded-cell violations; exact_feasible remains false regardless of image evaluation reproduction. Do not convert evaluation completion into a recovery success claim.
- Discuss the saved optimization evidence as observations: cell loss 22.8891772 -> .192871174; feature-target cosine about .930897 -> .680713; initial weighted cell/base gradient norm ratio about 9508.7; final gradient cosine inferred from saved norms about -.9542. This supports conflicting objectives in this run but does not uniquely establish the cause of poor source recovery. No unrequested feature-truth diagnostic is part of this task.
- Correct timing presentation in the NEW review report: the parent's per-stage wall_seconds values after SEPARATE are cumulative wrapper elapsed time, not independent durations. Derive independent durations from the stored stage start/end timestamps; preserve original records and total2330.7886750999605 seconds.

## Budget and delivery

Parent total real time was 2330.7886750999605 seconds under a 2700-second cap, leaving369.2113249000395 seconds. Allow at most300 seconds of new real evaluation/finalization computation, charged to that same budget; development/mock tests and idle review time are excluded as before. Record cumulative actual time. No budget extension, optimizer/replay work, downloads or additional runs. Prepare a single hidden/persistent wrapper and the existing one-time non-model completion callback; perform one launch check and end the Executor turn. No model polling.

Write report.md and execution_summary.json, runtime snapshots, before/after integrity, exact validation/evaluation commands and exits, original/new control residuals and final scientific-review disposition under .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REVIEW_20260917. Preserve original reports/figures and link the new comparison/figure. Ensure referenced files exist, send one final result to the verified existing Planner, then end. The Planner reports delivery once and does not wait for completion.
