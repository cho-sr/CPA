# Repair continuation export, recover the saved25000 endpoint and finish the registered INT8 screen3 experiment

TASK_ID: CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: ORDINARY_IMPLEMENTATION_REPAIR_AND_REMAINING_AUTHORIZED_COMPLETION
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_FIX_ERROR_AND_PRESERVED_SCREEN3_COMPLETION_20260918
PARENT_TASK_ID: CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917
REVIEW_REFERENCE: VERIFIED_CONTINUATION_WRITER_BUG_AND_USER_FIX_REQUEST
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_int8_randomized8_screen3_repair_B16_T10_20260918
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_int8_randomized8_screen3_repair_B16_T10_20260918
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918/report.md

## Authorization and intended completion

The user said "오류 수저해" after receiving the parent PARTIAL result. Correct the implementation failure and finish the SAME already-authorized experiment from its saved state. User authorization for all8 ICA -> all8 FIA5000 -> public top3 to TOTAL25000, center control, and final evaluation persists. Do not ask again. No extra experiment, sample, seed, tuning, reranking or budget extension is authorized or needed.

Planner remains CODEX_PLANNER and makes no research edits/tests/runs. Existing designated Executor is Luna/xhigh thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab,PID14236,start1789518279030ms; return to existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95,PID24972,start1789518271364ms. Validate actual role/runtime/identity and this READY handoff. No replacement AI/session/model or shared peer rebinding.

Read immutable parent plan .ai/execution-handoffs/CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917_v1.md SHA2567a9b09314f77458e1526cd318d99bef427633567f042cac975c5dfea341b5c4b and its inherited original INT8 plan. All fixed settings, target/order conventions, loss/ranking rule, no-private-feedback boundary, metric definitions,13-set evaluation and original12600-second total real-work budget remain binding. THIS repair authorizes only necessary local code/schema/provenance fixes, export from a completed checkpoint, two remaining registered continuations, and final gate/evaluation/reporting.

## Verified state and exact failure

Parent public root: quantization_aware_cpa/results/signed_int8_randomized8_screen3_fia_B16_T10_20260917.
Parent execution evidence: .ai/execution-results/CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917/{report.md,execution_summary.json,final_audit.json}.
Parent wrapper23720 finished; at Planner's inspection no parent Python process was active. Recheck once before launch to prevent duplicate work. The parent's COMPLETED handoff means Executor closed its turn; RESULT_STATUS=PARTIAL and the full scientific experiment is incomplete. Preserve that distinction in the archive.

Completed: sender INT8 conversion, center+8 ICA separations, center FIA25000 with5000 prefix, all8 member FIA5000, immutable public top3 selection [1,7,5], and member01 continuation to TOTAL25000. Executed85000/125000 FIA updates. Remaining optimizer work is ONLY member07:5000->25000 and member05:5000->25000, exactly40000 updates combined. No private evaluation has begun. No need to load raw private updates or generate observations again.

Root cause in qacpa/signed_int8_screen3_fia.py: run_trajectory initializes prefix_checkpoint_path=None; it writes a local5000 checkpoint only for a fresh25000 trajectory with parent_checkpoint=None. A valid child resumes at5000 and therefore intentionally creates no new5000 prefix. After saving checkpoint_25000.pt, the writer unconditionally requires prefix_checkpoint_path not None and raises StagedFIAError("25000 trajectory did not produce its 5000 checkpoint"). This is an output/provenance bug after successful optimizer computation.

Current failed driver SHA256ab36aef2990fc7c0c11f4a0abcd1bc162e6a5896c91e7bbb3e71176c55849a95; runner SHA2562b718a7203ca47e2b09773bdb344b7ee8b074ca5429d37a7cf58dced1f6b23e2. Authenticate them and preserve original source copies/diff before repair. Original registration.json SHA256d974ad6075bdf3dbb5607ec914d992957a26d537a83483e98f70c370242ea3fb. Original selection.json SHA2569c6a24bd73d0d9fb4ea752316b3a77e05779b6ae4152c090c229ca2b1d2a8683.

Critical immutable checkpoints:
- member01 prefix fia_member_01_5000/checkpoint_5000.pt SHA2568932029334060d05b9f50a75b8a4a3dd816d8726a1b41bbcbdd8bd470f033cb5.
- member01 final fia_member_01_25000/checkpoint_25000.pt SHA256e48fee1e339d2f47c25cd308f3f78cad0ff44e2b6e6d78ea72d8669190f14353; completed_iterations25000 and parent hash above. Candidate/result export missing.
- member07 prefix fia_member_07_5000/checkpoint_5000.pt SHA2565e8f6ead3dd7a3e2f486fb6adb587c55411ebdaf2f28b40d1d41d8fa21c154da.
- member05 prefix fia_member_05_5000/checkpoint_5000.pt SHA2567bdccb705c2edb911ce33f283e68167cb41c160262e336d667f3fdd7f1ce3908.
- center checkpoint_5000.pt SHA256ae7942cfce971091a9cb0ca357ebde184fe9e9f03977cd4c7a2a16273e5be1b6; center checkpoint_25000.pt SHA2562e6f075d2097d0c663648564359aae729895b86ebf090eae107a2931f0c9c9fe.

Authenticate every inherited observation/source/target/candidate/checkpoint/result/ranking and protected source using parent records and a fresh BEFORE inventory. No numerical private data may enter these checks. Preserve all existing parent public files, metadata, raw checkpoints and failure logs byte-identical; write repair artifacts under NEW roots with explicit references. Do not edit old hashes/registrations to pretend they were produced by the repaired code.

## Required code repair and regression coverage

Make a narrow correction to the isolated staged FIA driver: fresh25000 trajectories still require their own newly saved5000 prefix; resumed children require and record the authenticated parent5000 checkpoint and its provenance instead. Child outputs must not invent a newly generated prefix or silently omit source/parent validation. Preserve Direct.step, optimizer settings, scheduler horizon25000, pixel parameter, loss, seed and numeric path. Official/upstream/ICA/model/metric/INT4 sources remain unchanged.

Separate completed-checkpoint finalization from optimizer execution. Implement a strict exporter/finalizer for an authenticated TOTAL25000 checkpoint which performs ZERO optimizer steps and does not rewrite the checkpoint. It should produce the exact existing Direct.get_rec/_candidate pixel conversion and bound result metadata using the checkpoint's real history, not a newly initialized image. Validate checkpoint schema/step/config/target/source/parent/raw-image hash and state contents. Never relax the existing5000 resume validation to accept arbitrary step counts just to reuse run_trajectory.

Provide an isolated repair orchestration entrypoint and, as needed, a repair manifest/public gate adapter. Do not call old run_public from its beginning: it attempts to recreate the center and screens and refuses occupied folders. No skipping via broad exists checks; reused outputs must authenticate as completed with the correct target/source/config/step. Do not remove no-overwrite guards globally.

Source provenance needs an explicit migration: the old evaluator gate compares original registration code_hashes to CURRENT sources, so a legitimate driver patch will otherwise cause a second failure after the remaining hour of work. BEFORE real continuation, prepare a separate repair registration with the exact old registration/source hashes, the permitted source diff, the new source hashes and an explicit map of inherited versus newly produced artifacts. Authenticate old artifacts under their old source revision and new work under the repaired revision. Do not bypass hash gates, monkeypatch imports to make them agree, widen metric tolerance or overwrite original registration. Reuse unchanged metric functions through a strict repair-aware gate/evaluator adapter if necessary. Preserve source-role separation and PUBLIC_GATE_PASSED before private x loads.

Mandatory focused regression checks before real execution:
1. Exercise the PUBLIC run_trajectory writer branch for a fresh25000 trajectory, fresh5000 prefix and a child continuation from5000. A child must complete export while correctly linking its parent. Use a small synthetic driver/feature fixture; test the actual writer path, not only checkpoint payload serialization.
2. Export-only recovery of a completed checkpoint creates matching pixels/result with ZERO step calls; wrong step/hash/parent/target/config is rejected. Verify state and original checkpoint bytes remain unchanged by export.
3. Retain/extend existing serialized split-versus-continuous original-Direct validation for raw image, Adam state, schedule and loss. Test no reset on resumed children. No full-size validation rerun is authorized.
4. Mock end-to-end repair orchestration from a fixture shaped like this exact failure: center+all8prefixes+frozen selection+one final raw checkpoint; it exports member01, invokes ONLY member07 and05 continuations for20000 steps each, then passes the strict repair provenance gate to the13-candidate evaluator. No re-separation, re-screening or reranking.
5. Validate final candidate path/step/source/parent bindings, code revision migration, before/after integrity, budget accounting and gate-before-private-loading. Mutated inherited artifacts and unauthorized source changes fail. Run only relevant focused tests, py_compile and CLI help; no broad unrelated suite or added research run.

Ordinary isolated implementation/reporting fixes within this exact scope are already authorized and should be completed autonomously. If another output/gate defect occurs after optimizer work, preserve the successful checkpoint and repair export/gate from it rather than stopping permanently or recomputing the optimizer trajectory. Genuine data corruption, changed scientific conditions or exhausted budget must be reported honestly.

## Exact recovery sequence

1. Archive closed parent handoff and failure evidence; claim this task; snapshot inputs/old code. Verify no active duplicate process and authenticate all inherited artifacts. Freeze the repaired-source manifest and complete relevant validation before real continuation.
2. Load member01's existing final25000 raw checkpoint and its original5000 parent; validate the chain and export missing candidate/result under the repair public root. No ICA/FIA optimizer calls for this member. Record optimizer_steps_executed_by_recovery=0 and keep original file hashes unchanged. Metadata distinguishes original computation runtime from new export time; do not fabricate unavailable historical measurements.
3. Restore member07's original5000 raw image/Adam/scheduler/RNG state and run20000 additional steps to TOTAL25000 under the same original full25000 cosine schedule. Persist parent hash and all completion outputs in a new branch folder. Then do exactly the same for member05. The selection order1,7,5 remains frozen; no replacement branch or new target.
4. Assemble complete public outcome/provenance manifest for inherited center and all8 prefixes, recovered member01 and new member07/member05 final outputs. Record old85000+new40000=125000 total FIA updates. Verify old file bytes and protected sources unchanged, legitimate repaired source diff registered, all13 candidates frozen/valid, and PUBLIC_GATE_PASSED.
5. Run the originally authorized canonical-runtime offline x-only evaluation and two PNG contact sheets. Same13 sets: center5000 + eight members5000 + center25000 + public ranks1/2/3 at25000. One pixel-MSE Hungarian assignment per candidate reused for all MSE/PSNR/SSIM/VGG LPIPS and figures. Primary contrast remains PUBLIC rank1 member01 versus center25000; report other selected members separately, no private best-of3 or image averaging. No labels/true features/true updates consumed by repair or evaluator; disclose bundled deserialization separately. No new private feedback to attack. Actual-image cell feasibility remains NOT_EVALUATED.
6. Final audit/report includes prior failure, root cause and patch, regression evidence, old/new source revisions, member01 zero-step export, restored-state evidence for the only40000 new updates, frozen ranking,13-set metrics/figures, byte-preservation and elapsed/budget/resource accounting. Parent failure evidence remains intact; completion is reported as completion of the parent scientific experiment through this repair task.

## Existing budget and process policy

Parent execution_summary records7200.393110799952 seconds real execution under12600 total. Remaining original allowance is5399.606889200048 seconds, about90 minutes. Deduct any previously unaccounted real completion audit time if applicable. The two remaining20000 continuations should take roughly53-55 minutes at the observed rate, plus authentication/export/evaluation; this is an estimate. No new12600-second budget. Development/focused synthetic tests and idle time are excluded as before. CUDA allocated cap12GiB stays unchanged. Record both incremental and cumulative stage durations; no silent cap extension, precision/device/schedule/step change.

Prepare a complete sequential repair/export -> remaining public continuations -> public gate -> private evaluation wrapper before launch, using C:/miniconda/envs/DLG/python.exe. Snapshot-source/authentication setup and full pipeline tests must catch export/gate errors before consuming the remaining real budget. Hidden persistent launch, exact PID/start verification, one non-model OS completion callback to the SAME Executor, one startup receipt then END turn. No polling/new agent or restarted completed optimization.

Use BOM-free UTF8 task-local transport and preserve shared peers. On process exit, complete only remaining authorized reporting/evaluation fixes from saved artifacts and send ONE truthful final result to existing Planner, then end. Planner ends after one verified queue receipt and does not wait for claim/completion. Do not automatically launch any further experiment.
