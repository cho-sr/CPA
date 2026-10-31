# Repair exactly three remaining cell-rank validation holes

TASK_ID: CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: THREE_VALIDATION_HOLES_AND_SMALL_CPU_TESTS
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_THREE_VALIDATION_HOLES_REPAIR_20260915
PLANNER_SESSION_ROLE: CODEX_PLANNER
FULL_EXPERIMENT_AUTHORIZATION: EXPLICITLY_EXCLUDED_BY_USER
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915/report.md
PARENT_TASK: CPA_SIGNED_CELL_RANK_REPAIR_20260915
CONSENSUS_REFERENCE: RELAY_TURNS_48_TO_52_WITH_CODEX_SCOPE_CORRECTIONS
MODEL_DECISION: Continue in the same user-designated existing Luna Executor

## 1. Actual user instruction, scope and recipient

[FACT] Latest direct user instruction in the Planner conversation:

> 현재 비교 구현은 NEEDS_REPAIR 상태다. 새 알고리즘이나 실험 설정 변경 없이 아래 3개 validation hole만 수정하라.
>
> 설정 동등성 검사 전에 필드 존재/타입/SHA256 형식 검증을 추가하고 양쪽 누락 및 한쪽 누락 negative test를 추가한다. 기존 실제 baseline runtime contract는 유지한다.
> _read_preprocessed에서 registration metadata, result metadata, declared arm을 교차검증하여 rank16, centering, gram_eigh, 등록 AP iteration, AP0/AP5 구분이 일치하지 않으면 CPA 호출 전에 실패시킨다.
> raw_updates.pt의 true-G를 final offline evaluation에서만 사용하여 동일 observation/sender/checkpoint/layer에 binding한 뒤 relative error를 계산한다. denominator-zero 규약과 작은 synthetic boundary test도 추가한다.
> 이 repair에는 principal-angle gate, loss, scale constraint, label, seed, reconstruction 실행 변경을 포함하지 마라. 수정 후 unit/mock/negative test 결과와 변경 파일을 보고하고, 실제 GPU comparison은 실행하지 마라.

This explicitly authorizes these three repairs and their unit/mock/negative tests. It explicitly excludes a real GPU comparison. Original FP32 G evaluation-only access was already allowed by the user; do not ask for it again or describe it as a threat-model change. In THIS task, implement that offline path and test it on synthetic fixtures only, without loading real private tensors or running real evaluation.

Recipient: existing Luna/xhigh thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID18272, creation /Date(1789362262182)/, workspace C:/Users/조상제/Desktop/CPA. Verify your identity/runtime and delivered task/version/snapshot before claiming READY as RUNNING. The user's prior designation of Luna persists for this continuation; historical Astra/high wording does not change that. Do not change model/settings/bindings, launch another CLI or worker, or act on Planner relay turns.

Read AGENTS.md and required project documents. Read the full parent implementation plan .ai/execution-handoffs/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914_v1.md and the completed repair report .ai/execution-results/CPA_SIGNED_CELL_RANK_REPAIR_20260915/report.md for compatibility. Current research review is NEEDS_REPAIR; parent implementation-task COMPLETED does not mean all these newly reviewed gaps were fixed.

## 2. Preserve these boundaries

Preserve original CPA/whitening/loss/optimizer/FIA, original signed codec and cell utilities, fixed VGG16 Conv-frozen/FC-trained single-client B16/T10 seed42 conditions, raw-delta convention, attack layer classifier.0.weight, existing original CPA25k/FIA25k configurations and abs(S) adapter. Keep signed deterministic INT4 [-7,7], scalar per-parameter FP32 s and the existing rounding/zero/edge semantics. Original implementation uses external uncentered rank16/gram_eigh plus exact cell projection, AP5 default, AP0 explicit control and final-iteration-only candidate selection. None of those algorithms or scientific settings changes in this repair. Scale stays metric-only.

No principal-angle gate, new loss, scale anchor/constraint/rescaling, labels, replay, additional seed/arm, iteration increase, candidate ranking change, new observation, GPU operation, reconstruction, real-image metrics, downloads, benchmark or experiment launch. No cleanup of unrelated workspace modifications or existing results. Do not alter the completed parent artifacts.

Allowed research edits are confined to:
- quantization_aware_cpa/qacpa/signed_cell_rank_runner.py
- quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py
- quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py
- optional focused new test file quantization_aware_cpa/tests/test_signed_cell_rank_validation.py
- quantization_aware_cpa/docs/signed_cell_rank_comparison.md

Keep signed_cell_rank.py projection core unchanged; helper functions can be local to the runner/evaluator. Also write this task's result directory and update the active handoff's claim/status/result fields. Do not modify global role/bridge files, STATE or RELAY.

## 3. V1: evidence validation before equality

Before any settings/initialization equality comparison, require the corresponding objects/fields to exist and have the declared type. Validate initialization SHA256 as exactly64 hexadecimal characters (and consistent normalization if accepted). Reject missing/null/empty/wrong-type/invalid-format evidence on either side. Python bool/int equality must not make a bool pass as an integer or vice versa. Validate seed and initialization shape as required by the actual contract, and preserve existing final-artifact/source/checkpoint/path guards.

The actual existing naive/fia_initial.json contains:
- tensor_sha256: 6f6ec493923b7081284f0fc405f76f60b583cc41a5977393751f96bb21147e59
- shape [16,3,224,224], seed42
- cudnn_benchmark=true, cudnn_deterministic=true, deterministic_algorithms=true, deterministic_warn_only=true
- matmul_tf32=false, cudnn_tf32=false, cublas_workspace_config=:4096:8, threads=1

Read that existing JSON and the original adapter/config to confirm schema. Preserve these actual runtime values; notably do not impose warn_only=false because old mock fixtures used it. Require each compared RUNTIME_FIELDS member, type-check it, and check fixed values against the declared baseline contract as well as cross-arm equality. Never overwrite baseline metadata to make validation pass. Keep legacy naive schema compatibility explicit, with no permissive fallback that silently accepts missing required evidence. Report registered/executed-settings-valid only after every required check passes.

Tests: valid populated fixture passes; remove each required field from both sides and from just one side -> rejection; wrong type/null/empty/bad SHA256/equal invalid fixed runtime -> rejection. Include integer/bool confusion and retain a valid baseline-shaped fixture. Tests may read existing completed JSON but must not load real private tensors.

## 4. V2: registration/result/declared-arm cross-check before CPA

In _read_preprocessed and the attack entry path, validate presence/type/value and cross-consistency of preprocessing registration and preprocessing result, and compare them with an independently declared expected arm contract. Required: attack layer, format/schema as applicable, rank16, centered=false, rank_backend=gram_eigh, projection dtype/device as declared, final_iteration_only selection, intended AP iteration and scale metric-only semantics. Preserve q/s/observation/hash/shape/code/cell checks.

Do not derive the expected contract exclusively from the metadata being validated, or from output-directory names. If the existing attack API lacks the caller's expected AP setting, introduce the smallest validation-only explicit expected-AP argument/CLI option: existing AP5 remains the default intended comparison, and AP0 is an explicit identity control. Do not silently substitute AP0 data for an AP5 arm or vice versa, change the number of projection steps, add a new experimental arm, or reinterpret a mismatched artifact as valid. Keep registrations/results and documentation synchronized. Added path/hash references needed to bind the final V for V3 must be truthful, recorded before the attack and checked; do not rewrite previous result files.

Tests: consistent AP5 and explicit AP0 fixtures accepted; registration-vs-result mismatch and both-metadata-vs-declared-arm mismatch rejected, including missing/invalid rank, centering, backend, AP count, and swapped AP0/AP5. Spy on original run_upstream_cpa to prove rejected artifacts never invoke CPA. Exercise real validator logic on tiny fixtures, not a mocked constant validator. Core projection and original CPA remain unchanged.

## 5. V3: bound evaluation-only true-G error

Implement only inside final offline evaluation after both arms' finalization, public/final artifact checks and all V1/V2 provenance checks. Attack/preprocess code must have no private-path input or trueG loader. The raw update's existing generating schema at signed_baseline.py:172-174 is {updates, delta_convention, checkpoint_sha256}; validate the actual loaded schema/types/finite FP32 tensor and exact requested attack-layer shape instead of assuming it. The raw convention is theta_after10_minus_theta_start; do not divide by SGD lr/steps/batch or flip its sign.

Bind the actual raw_updates.pt file hash to sender_validation.json, the sender to the same public manifest and checkpoint, the raw payload checkpoint/delta convention and updates[ATTACK_LAYER] to that observation's layer, and each compared estimate to the same finalized arm input. Retain path separation and no-overwrite rules. Validate the public packets/manifest and actual sq input fingerprint; bind final V using its recorded preprocessing/result/file/tensor and canonical CPA input hashes. If the current schema lacks the V path/reference, add only the minimal public provenance reference needed for this binding and validate it. Missing or inconsistent binding must fail before computing/reporting the metric. Do not select a different artifact, recompute training, or use GT to repair any candidate.

For each of naive sq and final V, report:

    absolute_l2_error = ||estimate - G_true||_F
    reference_l2_norm = ||G_true||_F
    relative_l2_error = absolute_l2_error / reference_l2_norm   when reference_l2_norm > 0

Denominator-zero convention for this repair: if reference_l2_norm==0, relative_l2_error is JSON null with status UNDEFINED_ZERO_REFERENCE_NORM, including when both tensors are zero. Still record absolute_l2_error (zero when both are zero) and reference_l2_norm. Do not add epsilon, invent a finite ratio, serialize NaN/Infinity or turn that case into a scientific success threshold. Reject nonfinite/shape/type-invalid tensors. Use explicit norm definitions and numerically sensible accumulation (e.g. chunked CPU float64 sums for large future inputs) without changing attack tensors or projection precision.

Store these values as clearly evaluation-only per-arm update metrics, with layer/convention/provenance identifiers. Preserve numerical results if plotting later fails. No metric feeds candidate selection, ranking, tuning, stopping or reconstruction. private_loaded=false describes the prior validation stage only; add truthful offline-loading status if useful instead of mislabeling the whole evaluator. Existing image/embedding matching and final16 candidate policy remain unchanged.

Synthetic tests: independently known positive-denominator norm result for both sq/V; zeroG with zero estimate and zeroG with nonzero estimate; strict finite JSON; invalid shape/dtype/nonfinite; wrong checkpoint/layer/convention/sender/public/raw/V hash rejected; incomplete/unbound arms reject before a private torch.load. A mock full pipeline test verifies attack/preprocess cannot load the private sentinel and the offline evaluator loads synthetic trueG only after all final checks, saves the metrics and does not change selected candidates. Never load actual private tensor artifacts during this implementation task.

## 6. Execution and validation budget

Use C:/miniconda/envs/DLG/python.exe for small synchronous CPU unit/mock/negative tests only. Synthetic projection/update matrices at most32x64; shape metadata/mocked candidate tensors may retain16x3x224x224 to exercise the existing adapter contract without VGG computation. Mock expensive CPA/FIA/model/image-metric calls; keep binding validators, canonical manifest/hash logic, strict JSON writing and new norm computation real. No CUDA test, full-size observation decode/projection, real trueG load, image reconstruction/LPIPS or actual comparison.

First capture regression failures for the new gaps, implement focused fixes, then run appropriate focused tests and existing codec/original-adapter/provenance regressions. Suggested commands (include the optional test file only if created):

    C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py quantization_aware_cpa/tests/test_signed_cell_rank_validation.py -q
    C:/miniconda/envs/DLG/python.exe -m pytest quantized_update_cpa/tests/test_cell_utils.py quantization_aware_cpa/tests/test_upstream_naive.py quantization_aware_cpa/tests/test_signed_baseline_provenance.py -q

Inspect tests before execution and narrow only as needed to honor the synthetic CPU scope, reporting the actual commands. Retain stdout/stderr, exit codes or JUnit evidence in this task directory. Do not count repeated/overlapping passes as unique tests. Record protected-source hashes before/after against the parent implementation results.json's10 protected_source_hashes_current values, and also preserve the current signed_cell_rank.py hash. Report unexpected differences instead of modifying originals. No new testing/experiment expansion after relevant checks pass.

## 7. Completion report and delivery

Write .ai/execution-results/CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915/report.md and results.json. Include [FACT]/[INFERENCE]/[PROPOSAL]/[UNKNOWN], changed files, V1/V2/V3 acceptance evidence, denominator-zero schema/examples, actual test commands/counts/durations/exit codes, protected hashes, schema/CLI compatibility, and remaining limitations. Status FULL_EXPERIMENT_NOT_RUN must remain explicit. Mark the handoff COMPLETED only after these three repairs, necessary tests and documentation are complete; otherwise NEEDS_REVIEW/FAILED with exact remaining work. Do not claim real restoration improvement from synthetic tests.

Display the final report summary in the existing Luna CLI, send one authorized final completion/failure notification to the verified existing Planner, and end. The current discussion CODEX binding is thread01a09ac2-0e8d-7f90-94a8-c45be99f4343/PID12728; recent Turns47/50/51 deliveries reached it despite a historical failed queue invocation. Do not assume permanent staleness from that old failure. Verify the current peer process/thread/store before a one-time notification; use the installed C:/Users/조상제/AppData/Local/Programs/OpenAI/Codex/bin/codex.exe queue command if valid. No rebind, blind retry, alternate/new session, monitoring loop, callback or research restart. If final delivery cannot be verified, retain the report and state delivery unconfirmed. Planner will not poll while you work.
