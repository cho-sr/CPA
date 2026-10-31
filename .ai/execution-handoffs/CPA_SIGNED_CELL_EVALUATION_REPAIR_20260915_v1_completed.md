# Repair final evaluator and finish the existing comparison

TASK_ID: CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915
PLAN_VERSION: 1
STATUS: COMPLETED
TASK_KIND: EVALUATOR_REPAIR_REGRESSION_TESTS_AND_OFFLINE_EVALUATION_ONLY
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
COMPLETED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
RESULT_STATUS: COMPLETED
FULL_EXPERIMENT_RESULT: FULL_EXPERIMENT_COMPLETED
PLANNER_FINAL_DELIVERY: FINAL_RESULT_HANDOFF_CONFIRMED
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_FIX_AFTER_EVALUATION_ONLY_PROPOSAL_20260915
RECONSTRUCTION_EXECUTION_AUTHORIZATION: NONE_IN_THIS_TASK_REUSE_COMPLETED_ARTIFACTS
PLANNER_SESSION_ROLE: CODEX_PLANNER
PARENT_TASK: CPA_SIGNED_CELL_RANK_COMPARISON_B16_T10_20260915
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915/report.md
MODEL_DECISION: Same existing user-designated Luna/xhigh Executor

## 1. Actual user authorization and recipient

[FACT] The parent comparison completed AP5 preprocessing and original CPA25,000/FIA25,000 and saved all16 candidates. Final offline evaluation failed before private truth loading: signed_cell_rank_evaluation.py line297 references legacy_naive in _check_arm, whose parameter is named naive. No evaluation output was created. Protected original sources matched.

[FACT] The Planner read the saved result, traceback and source, then proposed:
> 변수명 오류와 해당 경로의 테스트를 보완한 뒤 저장된 결과의 평가 단계만 실행하면 됩니다. AP5·CPA·FIA를 다시 돌릴 필요는 없습니다.

The user explicitly replied:
> 보완해줘

This authorizes that concrete evaluator repair, regression coverage, and completion of the already-requested comparison using existing artifacts. Do not ask for this same scope again. It does not authorize a new attack, scientific method or scale constraint.

Recipient: existing Luna/xhigh thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID18272, creation /Date(1789362262182)/, workspace C:/Users/조상제/Desktop/CPA. Read AGENTS.md and required role/protocol docs, this full handoff and parent report. Verify identity/runtime, delivered pre-claim SHA and READY/unclaimed state; claim with your thread ID and mark RUNNING. Historical Astra/high wording does not override the user's Luna selection. Do not change bindings/settings or create another CLI/model/worker. Do not act on Planner RELAY turns.

## 2. Minimal repair with meaningful regression coverage

Inspect current code and the actual saved naive JSON metadata without modifying those artifacts. In _check_arm, replace the undefined compatibility variable with the function's existing naive flag (or an equally minimal explicitly bound equivalent). Keep _completed(..., legacy_naive=naive) and its distinct parameter intact. Do not perform a blind global name replacement.

The intended existing contract is unchanged: the older completed naive result may omit its duplicated result-level checkpoint hash and use its explicitly validated registration hash; new cell-rank results must supply the result-level hash. If a result-level hash exists for either arm, validate its type/format/value rather than falling back on malformed or mismatched input.

Add focused tests that exercise the actual compatibility branch, rather than fixtures that always populate the missing field:
- Valid legacy naive result without result-level checkpoint_sha256, with a valid matching registered checkpoint, reaches successful public/final-arm validation. A mock evaluator integration test for this exact schema reaches the intended later stage without NameError; retain actual schemas and existing contract.
- Non-legacy cell arm missing that field raises the expected ProvenanceError before private ground-truth loading.
- Missing/malformed/mismatched registered checkpoint still fails. Present malformed or wrong result-level checkpoint fails for both arms; no permissive fallback.
- For the rejecting cases, use a private-load sentinel/mock to prove failure before private true-G/images/embeddings are deserialized. Do not patch away the function/branch under test.

Demonstrate the specific regression reproduces on pre-fix behavior, then passes after the fix. Run relevant existing signed_cell_rank pipeline and validation tests, plus syntax/import checking. Existing original-adapter/codec tests need rerunning only if a relevant change or new concern justifies it. Preserve all three prior validation repairs, including required presence/types/SHA validation, independent AP arm registration checks and true-G evaluation boundary/zero-reference convention.

Allowed research edits: quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py; quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py; quantization_aware_cpa/tests/test_signed_cell_rank_validation.py; an optional focused new test file in that tests directory; necessary correction notes in quantization_aware_cpa/docs/signed_cell_rank_comparison.md. Execution-only wrapper/log/report files belong in this task's result directory. No research code outside this evaluator scope. Routine evaluator defects encountered during this authorized completion may be fixed minimally with regression evidence and without changing scientific semantics or weakening guards; a real provenance mismatch must be reported rather than bypassed.

## 3. Preserve completed attacks, provenance and scientific conditions

Reuse exclusively:
- public observation: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation
- stored naive: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive
- AP5 preprocessing: quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5
- completed cell arm: quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5
- evaluation-only private observation: quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation

Preserve all existing input/artifact bytes and histories, including failed parent report/status. Snapshot relevant code and metadata hashes before edits and recheck original protected sources and input/attack artifacts after evaluation. Document evaluator before/after hashes separately; do not rewrite recorded execution fingerprints to match newly edited evaluator code. signed_cell_rank.py must remain 7ae6deb49738914f4cd1f9328397400d89835ede4cd6a462b2ea260a77ae4acb. Parent final V file hash: 7b4850a13dc725017eab7da451019b1a2c3e90bd6d58d8d4a0db3181fcd010cf. Cell candidate file hash: 700beaa7b7da8565da68bb782ea67738542695334d09a2910e29f83bf29ab9d6. Authenticate remaining hashes from existing finalized manifests and parent run evidence.

Keep original official CPA, eig/whitening/loss/optimizer/FIA and adapters untouched. No AP5/CPA/FIA rerun, no FP32 reference or naive rerun, no local training/replay/data generation, no extra arm/seed, no scale anchoring/loss/rescale, no principal-angle gate, no new ranking/selection, no MP/noise assumption, no labels.

Fixed signed INT4 per-parameter scalar FP32 scale, q[-7,7], nearest-even/zero/bounded-cell rules; VGG16 Conv frozen and FC trained, single-client B16/T10 seed42, same SGD lr0.001/momentum0/weight_decay0/dropout0 and attack layer classifier.0.weight [4096,25088]. Raw delta theta_after10_minus_theta_start unchanged. AP5 uncentered rank16 gram_eigh and final-only candidate policy remain historical executed facts. Preserve all original runtime/initialization validators, including deterministic_warn_only=true. Unrecorded round/global client IDs remain Unknown/null.

Scale drift 7.142846% and code ratio1.0 are already-recorded diagnostics. Do not change the candidate to enforce scale or reinterpret fixed-s code agreement as complete max-abs scale agreement. This task evaluates the saved method as executed.

## 4. Run only final offline evaluation after tests pass

From repository cwd:

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_evaluation --naive quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive --cell-rank quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --private-observation quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5

Verify that the target is absent before creating it. If it has since appeared, inspect finalized state/provenance and avoid duplicate evaluation; never overwrite/delete existing results. Preserve failed partial output and report exactly what completed. Public and completed-arm provenance, settings and initialization equality must pass before private truth use. Run the actual evaluator with the saved outputs, not a mock substitute for final metrics.

Use evaluation-only true-G bound to the same observation/sender/checkpoint/layer/delta convention for naive sq and final V absolute/reference/relative Frobenius errors. Zero reference norm remains null with UNDEFINED_ZERO_REFERENCE_NORM plus separate absolute error. No epsilon/NaN/Inf. Original images and oracle embeddings may be loaded only in final evaluation. Private labels are never read/indexed/used for reconstruction or matching.

Use all16 final images and the existing per-arm pixel-MSE Hungarian mapping for LPIPS/PSNR/SSIM. Keep embedding-oracle cosine separate from image metrics, with raw/mean/per-image comparisons and correct metric directions. Existing comparison figures are allowed as outputs of the evaluator. No ground-truth selection/tuning and no new scientific success threshold. State improved/regressed/mixed evidence honestly; one batch/seed is not a general claim. Retain numerical JSON if plotting fails and report plotting separately.

## 5. Budget, execution lifecycle and reporting

Budget is the focused CPU repair tests and completion of the single stored-result offline evaluation (metric GPU work is permitted if already used by the evaluator; no reconstruction/benchmark). Actual runtime/peak memory are measured or Unknown, not guessed. Stop on scientific/provenance mismatch, OOM/NaN or any failure requiring a condition change. No automatic experiment restart or hyperparameter tuning. Same-scope evaluator correction and its necessary regression/validation are authorized; do not stop at a new routine evaluator error merely to ask for the same permission.

Follow WORK_MODE.md: no repeated model-based progress polling. If work needs background execution, use a hidden execution-only wrapper, retain PID/start-time/exit/status/logs, register at most one scoped non-model process-exit callback to the same existing Executor, then end the model turn. A fast task may complete in the current turn; no waiting loop or new monitoring agent. Finish approved evaluation/report when notified.

Write .ai/execution-results/CPA_SIGNED_CELL_EVALUATION_REPAIR_20260915/report.md and results.json with changes and rationale, pre-fix reproduction, test counts/commands/exits, real evaluator command/exit, provenance checks, all requested numerical metrics and differences, actual output/figure paths, runtime/VRAM where available, source/input protected hashes, and explicit stage statuses/limitations. Distinguish FACT/INFERENCE/PROPOSAL/UNKNOWN. Do not claim scientific improvement from code/cell consistency or all-tests-passed alone.

Mark this repair/continuation completed only when required repair/tests/evaluation/validation/report finish. Preserve the original failed parent record, and explicitly link this task as its evaluation-only continuation. Final success or failure is sent once to existing Planner thread 01a09ac2-0e8d-7f90-94a8-c45be99f4343 PID12728 after verifying pinned process identity. Use existing queue protocol with receipt, no blind retry/rebinding/new CLI/RELAY discussion. Planner displays the result once and ends its turn. Existing single final-result notification authorization applies.
