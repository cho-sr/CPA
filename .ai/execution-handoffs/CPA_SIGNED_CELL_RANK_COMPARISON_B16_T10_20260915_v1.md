# Run the repaired signed INT4 cell-rank comparison once

TASK_ID: CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: SINGLE_APPROVED_GPU_COMPARISON_AND_FINAL_OFFLINE_EVALUATION
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_RUN_COMPARISON_VIA_EXISTING_LUNA_20260915
FULL_EXPERIMENT_AUTHORIZATION: EXPLICIT_USER_RUN_AFTER_VALIDATION_REPAIR
PLANNER_SESSION_ROLE: CODEX_PLANNER
PARENT_TASK: CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915
CONSENSUS_REFERENCE: RELAY_TURNS_48_TO_52_AND_USER_AUTHORIZED_REPAIRS_AND_RUN
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915/report.md
MODEL_DECISION: User-designated existing Luna Executor; no model or binding change

## 1. Actual authorization and immediate action

[FACT] The three validation repairs completed: 43 focused CPU tests and 109 regression tests passed; protected source hashes unchanged. Their FULL_EXPERIMENT_NOT_RUN and GPU exclusion describe that completed repair task only.

The user subsequently asked whether the comparison was ready, then explicitly instructed:
> 루나한테 돌리라해
> 아니 루나한테 실행코드 던지라고

These latest direct requests authorize the previously described real comparison: AP5 preprocessing, original CPA25,000, original FIA25,000, and final comparison against the stored signed naive arm. Repeated messages mean ONE run. This new authorization supersedes the earlier repair-only GPU exclusion for THIS comparison, without authorizing unrelated experiments. Execute the commands below; do not substitute another review-only task or ask for the same execution permission again.

Recipient: existing Luna/xhigh thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID18272, creation /Date(1789362262182)/, workspace C:/Users/조상제/Desktop/CPA. Read required project docs, the full current handoff, latest repair report and signed_cell_rank_comparison.md. Verify own identity/runtime, delivered pre-claim SHA, READY/unclaimed status, and no duplicate run, then claim with your thread ID and mark RUNNING. Historical Astra/high bridge wording does not override the user's Luna designation. No new AI/CLI/worker or global model/binding/STATE/RELAY changes.

## 2. Exact commands, repository cwd, serial exit-code guards

Command 1: public AP5 preprocessing.

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_runner preprocess --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5 --ap-iterations 5 --projection-device cuda

Command 2: only after successful finalized preprocessing.

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_runner attack --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --preprocessed quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5 --checkpoint "C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth" --output quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --expected-ap-iterations 5

Command 3: only after successful finalized CPA and FIA, final offline evaluation.

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_evaluation --naive quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive --cell-rank quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --private-observation quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5

Use existing implementations. Stop the sequence on command, provenance, finite, code or cell validation failure; report the exact failure. No silent fallback, changed scientific settings or extra arm. Routine same-scope path/logging/execution plumbing fixes are allowed and recorded; no research algorithm rewrites or weakened validation to make the run succeed.

## 3. Fixed scientific and information contract

- Same existing VGG16 Conv-frozen, all six FC parameters trained, single-client batch00 B16/T10 same batch/order, seed42, SGD lr0.001, momentum0, weight decay0, dropout0. Reuse observations; no recollection, training replay, new seed or batch.
- Signed deterministic INT4 [-7,7], CPU int8 q and scalar CPU FP32 per-parameter-tensor s=max_abs/7. Existing nearest-even rounding, zero q0/s1, bounded/representable cell semantics. No affine codec, dither or iid-uniform noise assumption.
- Attack layer classifier.0.weight [4096,25088], raw delta theta_after10_minus_theta_start. No sign/lr/T/B conversion.
- Public six q/s packets and checkpoint/rules only in preprocessing and attack. Original FP32 update, images and embeddings are final offline evaluation inputs only. Private labels are never used for attack/replay/selection or image matching.
- Original official CPA code, eig/whitening, loss, optimizer, seed and Direct FIA are unchanged. Whitening is not a new loss. External uncentered rank16 gram_eigh preprocessing is separate from canonical CPA.
- Exactly five rank/cell alternating steps; low-rank M and cell-feasible V remain distinct. Final V only -> original CPA25,000 -> abs(S_hat) -> original FIA25,000. Final-only selection preserved.
- Scale metric-only. No scale anchor/loss/rescale, midpoint attraction, principal-angle gate, standard MP threshold, quantization-aware FIA loss, joint G optimization, weighted score or update reranking.
- Reuse completed signed naive arm after strict binding/runtime/initialization checks. No rerun of naive or FP32 Reference; no AP0/additional arm.
- Preserve actual baseline runtime including deterministic_warn_only=true, TF32 disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8, threads1 and original stage-specific cudnn flags. Projection uses documented CUDA FP32 path with TF32 disabled. Preserve execution evidence; fail honestly on mismatch.
- Unrecorded round/global client IDs stay Unknown/null; do not fill them.

## 4. Validation, provenance and evaluation

Before execution save exact commands, configuration, task/version, source/input/checkpoint hashes and outputs in a new run manifest. Check protected hashes against the completed repair results; signed_cell_rank.py remains 7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb. Preserve original CPA/FIA, baseline, codec/cell utilities and past results. Recheck protected hashes after execution. Do not rerun already-passed 43+109 tests without a new failure/edit requiring it.

Use repaired registration/result/independently declared AP5 checks before CPA, and required-field/type/SHA checks before evaluation. Missing metadata is never evidence of equality. Final V must pass observed-scale code and bounded-cell checks. Log finite status, movement from sq, M/V residuals and scale drift; scale drift is not a new constraint.

Evaluate all16 final images using existing per-arm pixel-MSE Hungarian matching reused for LPIPS/PSNR/SSIM. Report means, per-image scores and deltas. Report embedding-oracle cosine separately. Matching and embedding oracle are evaluation-only, never candidate/iteration selection.

Load true-G from raw_updates.pt only in the final offline evaluator after finalized arm/public/provenance checks, binding the same observation/sender/checkpoint/layer/raw delta convention. Report naive sq and final V absolute Frobenius errors, reference norm and relative errors. A zero reference norm is JSON null plus UNDEFINED_ZERO_REFERENCE_NORM, with separate absolute error; no epsilon, Infinity or NaN. No early true-G access for choosing preprocessing/reconstruction.

No invented success threshold. Report improved, regressed or mixed metrics as observed. One batch/seed is limited evidence; code/bin agreement alone is not image recovery.

## 5. Outputs, finite budget, process lifetime and failure

Allowed outputs are the three new command output directories, .ai/execution-results/CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915/ (execution-only wrapper, manifest, stage logs/status/report/results), active handoff claim/status/result fields and scoped completion/delivery records. Do not overwrite existing results or delete partial output. If a path exists, prevent duplicates; reuse only a demonstrably finalized stage of this same run. Preserve foreign/conflicting output and report it.

Budget: one local GPU AP5 + CPA25k + FIA25k, followed by one final offline evaluation. No repeat, additional seed, tuning or GPU benchmark. Record actual stage runtime and peak VRAM from existing instrumentation or execution wrapper. Full-shape wall time/peak VRAM are Unknown until measured. Stop on OOM/NaN/validation/command failure without auto-restart or changed settings.

A minimal execution-only wrapper may serialize commands, retain logs/exit codes/run records and preserve process lifetime after the model turn. Background helpers must be hidden. Follow WORK_MODE.md: once the process and approved remaining sequence are prepared, register at most one model-free process-exit callback via .ai/completion_notify.py using exact PID/start time, then end the model turn. No sleep/status/write_stdin monitoring loop or monitoring agent. The callback may wake this same Executor once to finish approved evaluation/validation/report.

## 6. Final report and one result delivery

Write report.md and results.json in this task's result directory: actual stage status/exit codes, input/source/output provenance, settings, numeric image/embedding/true-G results versus reused naive, code/cell/movement/scale diagnostics, runtime/peak VRAM (Unknown if unavailable), execution-plumbing changes and remaining issues. Distinguish FACT/INFERENCE/PROPOSAL/UNKNOWN. FULL_EXPERIMENT_COMPLETED requires preprocessing, CPA, FIA, evaluation and validation completion; otherwise state completed stages and exact failure.

Keep old repair reports unchanged. Update this active handoff with status/result paths. Deliver final completion or failure once to existing Planner thread 01a09ac2-0e8d-7f90-94a8-c45be99f4343, PID12728, after verifying its pinned process identity. Use existing queue protocol; no new CLI/rebinding, blind retries, courier reply or peer research loop. The existing user-authorized single completion notification persists. Planner displays it once and does not monitor.
