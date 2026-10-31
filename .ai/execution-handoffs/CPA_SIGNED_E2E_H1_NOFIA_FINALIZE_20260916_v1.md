# Repair endpoint authentication and finish the existing noFIA comparison

TASK_ID: CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: ROUTINE_VALIDATOR_REPAIR_AND_SAVED_CANDIDATE_FINALIZATION_ONLY
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_PRIOR_USER_GO_H1_ABLATION_INCLUDES_REPAIR_AND_FINAL_EVALUATION_20260916
PARENT_TASK_ID: CPA_SIGNED_E2E_H1_NOFIA_20260916
REVIEW_REFERENCE: PLANNER_CODE_REVIEW_OF_PARENT_PARTIAL_FAILURE
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916/execution_summary.json
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916

## Authorization and immutable research scope

This is completion of the user's already-authorized H1 FIA-weight1-versus0 ablation (user said `ㄱㄱ`), including ordinary implementation repairs and required final evaluation explicitly covered by the parent handoff. It is not a new experiment. The parent 100-step C_H1_high_noFIA optimization finished successfully; only endpoint authentication/reporting/evaluation remain. The completion notice's PARTIAL/NEEDS_REVIEW has now been reviewed by Planner. Do not request duplicate approval or stop at another proposal for these routine fixes.

Read the full parent plan at .ai/execution-handoffs/CPA_SIGNED_E2E_H1_NOFIA_20260916_v1.md and its partial completion receipt archived beside this plan. Scientific conditions and criteria remain exactly as registered. No optimizer steps, candidate changes, resumed training, rerun of the original wrapper/run_ablation, control retraining, tuning, new labels/arms/seeds, budget extension or weaker scientific feasibility test are authorized. Reuse the existing final candidates only.

Assigned existing Executor: Luna/xhigh thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, creation1789518279030ms. Existing Planner return thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID24972, creation1789518271364ms. Reverify role, current runtime, process identity and delivery hash; no model/binding changes or new CLI.

## Confirmed failure mechanism and additional review findings

FACT from source review: signed_e2e_nofia_runner._endpoint_profile calls the original canonical_profile then adds seven fields: label_hypothesis, feature_target_cosine_loss, tv_normalized, pixel_min, pixel_max, pixel_mean, x0_displacement_l2. The old archived control profile has none of these fields. _same_profile first requires identical dictionary key sets, and the caller passes the entire enriched profile against the archived canonical profile. This guarantees a keys mismatch before any numerical values are inspected. The actual common-field numerical reproduction is still unverified and must be checked after fixing the schema comparison.

FACT from Planner's read-only audit: both previous registration.json and integrity.json are valid UTF-8 JSON, parsed successfully by json.loads(Path.read_text(encoding='utf-8')). The new _raw_metadata_authentication rationale about invalid historical Windows JSON is unsupported. Authenticate these structured records and actual input/source hashes, not just substring status markers.

FACT from current hashes: parent registration's 10 input hashes, 13 protected source hashes and 5 shared source hashes all match disk; parent candidate still hashes to the expected value below. Parent optimization_log.json contains100 rows. These are evidence for preserving the candidate, not substitutes for final endpoint validation.

The current _same_profile treats counts and booleans as tolerance-based floating numbers. Required integer counts, tensor cardinalities, booleans, names/shapes and metadata must compare exactly. Do not weaken existing floating comparison thresholds to make mismatches pass.

The new evaluator's figure currently plots raw candidate order alongside truth order, despite Hungarian matching being used for metrics. The final figure must use each candidate set's already-computed metric correspondence so each row refers to the same truth image; do not compute a new correspondence for the figure or imply row identity without it. This is a reporting correction, not a change to optimization or image metrics.

## Required repair and regression validation

Modify only parent-task new files signed_e2e_nofia_runner.py, signed_e2e_nofia_evaluation.py and test_signed_e2e_nofia.py as necessary; add an isolated saved-candidate finalizer and its focused tests if useful. Preserve the original screen modules signed_e2e_cells.py/signed_e2e_runner.py/signed_e2e_evaluation.py, all shared replay/codec/baseline/AP5/FIA/CPA code and all old experiment outputs. Archive bytes and hashes of every repaired parent-task source before changes, and report the exact source diff. Do not alter the optimizer implementation or objective to repair a reporting gate.

Compare a complete, explicitly defined canonical profile schema against the archived profile. Either keep canonical and added diagnostics separate or validate all required canonical keys and compare that exact substructure; allow only the documented extra reporting keys. Never silently take a dictionary intersection, ignore missing required fields, or bypass the archived-control gate. Preserve all six per-tensor fields. Counts/booleans/string metadata must match exactly. Report each mismatch with path, expected and actual values; record float differences before applying the preexisting tolerance. A real numerical or provenance discrepancy still stops private evaluation and is reported explicitly.

Replace raw-marker authentication with structured UTF-8 JSON/schema checks and actual byte hashes for the previous registration, integrity and public inputs. Compare to the parent registration and prior recorded input/source hashes, normalizing path spelling only as needed; never normalize file contents or substitute self-generated evidence for an expected hash. Preserve the private-load phase boundary. Check the evaluator's candidate, x0, control, label, coefficient, completed-step, profile and integrity guards against the actual finalized schema.

Meaningful regression tests: valid canonical archived profile plus allowed added diagnostics passes; changed required value, omitted tensor/key, changed integer/boolean, and unexpected extra schema fail; differing float values report their actual discrepancy under unchanged bounds; valid Windows paths parse as UTF-8 JSON and changed input hashes fail authentication; the finalizer cannot call optimization and refuses changed/non100-step candidates; private loading is not invoked if public completion/provenance fails; the figure uses the recorded correspondence. Reuse relevant existing tests and rerun only checks needed for the repairs. Do not use real private fixtures in tests.

The log's pre_step_base_loss currently includes diagnostic FIA even though lambda_F=0; preserve the recorded optimization log byte-for-byte and document this naming issue. The actual weighted total and optimizer-gradient exclusion are separately recorded. Do not rewrite old log values to improve presentation.

## Finalize from the frozen candidates

Required saved parent candidate:
quantization_aware_cpa/results/signed_e2e_h1_nofia_B16_T10_20260916/C_H1_high_noFIA/candidate.pt
SHA256: 5a20b87bf69473bf7696e46d9054c0f3d04a4c02f98ffde6ee3969374bf4e525

Required stored FIA=1 control:
quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916/B_H1_high/candidate.pt
SHA256: b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a

Before any completion work, authenticate candidate/state/result/log consistency, exact100 completed steps, H1/order, lambda_F0, lambda_TV0.1, lambda_C0.24202997768915402, x0/target/public checkpoint/input hashes, previous numerical preflight and actual current runtime. Confirm parent public source/inputs stayed unchanged across the completed optimization using its recorded hashes. Disclose which hashes are historical evidence and which are current repair records. The original candidate, state, result, log, registration and preflight must remain byte-identical.

Use a finalization-only entry point that loads frozen candidates and cannot reach _run_arm/run_ablation/Adam or any image optimization. Regenerate only the three previously requested canonical H1 endpoint profiles: x0, stored B_H1_high, existing C_H1_high_noFIA. The parent already computed three profiles in memory before losing them at the schema failure; count this repeat diagnostic work honestly (six endpoint passes across original+repair if exactly three are repeated). No extra calibration, independent training trajectory or optimization is needed.

Persist the repaired endpoint profiles and detailed archived-control authentication before private evaluation. Verify core control profile reproduction first; added control pixel/FIA/TV metrics can additionally be checked against its existing result.json. Authenticate and save integrity after these checks. Exact scientific cell/code feasibility remains zero bounded-cell violations AND zero fixed-s code mismatches over all six unprojected actual replay tensors. Keep candidate scale diagnostics separate.

Preserve parent failure.json, parent execution logs/report and initial failed summary. Before updating parent summary.json for truthful final public completion, archive its exact failed bytes/hash under the new finalization task's `before/` directory. Record both stages: original optimization complete, original wrapper exit1 due validator, finalizer exit and evaluation exit. Do not rewrite the parent execution record to pretend its wrapper succeeded. New top-level finalizer report should supersede the partial scientific comparison while linking the immutable original failure record.

The scientific public/private artifacts retain parent task ID CPA_SIGNED_E2E_H1_NOFIA_20260916 because they describe that same registered experiment; add finalization_task_id and recovery provenance explicitly. The new execution receipt uses CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916. This distinction must be explicit, not falsified to pass a validator. New endpoint/integrity/finalized summary files may be placed in the existing partial public root after the above archival; do not overwrite original registration/candidates/states/results/logs/preflight.

Only after public authentication passes, run the originally authorized offline evaluator once on all16 images for x0/control/new. The private root did not previously exist; verify before creating it. Use the original pixel-MSE Hungarian matching once per set, reused for PSNR/SSIM/LPIPS and the aligned figure. Only private x is consumed, labels_consumed=false; no true updates or labels in attack/evaluation decisions. Prefer the prior machine-readable metrics for control reproduction if available; if reading rounded Markdown, disclose precision and do not loosen a guard to hide a real mismatch. This task may repair ordinary evaluation/reporting defects without changing scientific methods.

Report endpoint feasibility, full tensorwise cell/code profiles, image metrics, control reproduction, protected input/source/candidate integrity, timings, warnings and actual source changes. Parent proposal decision rules remain unchanged. Preliminary optimization losses are not a final scientific comparison; do not claim FIA removal helped based solely on unfinished results. No new experiment follows automatically.

## Budget and completion delivery

Retain12GiB CUDA peak allocated cap and the parent's45-minute real-compute cap, charging the already measured155.46602060005534 seconds public real phase plus the repair's replay/evaluation wall time. Idle time awaiting review is not computation. Record repeated diagnostic costs; no budget extension is needed or authorized.

Result/log directory: .ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_FINALIZE_20260916. Record tests, exact commands, source before/after diff/hashes, archived failed summary, all execution/exit evidence and final report. If a true data/numerical mismatch remains, save a precise mismatch report and leave PARTIAL/NEEDS_REVIEW; never bypass the gate.

This is ordinary repair/completion within prior user scope, not renewed experimental authorization. Complete the saved-candidate comparison rather than merely acknowledging the task. Follow WORK_MODE: if using a background wrapper, launch hidden/persistent, register one non-model exit callback, end the turn, finish reporting on exit, ensure all referenced report files exist, send one final result to the verified existing Planner and end. No polling or extra agents. Planner ends after one verified delivery receipt.
