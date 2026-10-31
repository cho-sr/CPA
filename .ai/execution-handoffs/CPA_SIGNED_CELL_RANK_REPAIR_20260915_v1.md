# Signed INT4 cell-rank implementation repairs

TASK_ID: CPA_SIGNED_CELL_RANK_REPAIR_20260915
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: IMPLEMENTATION_REPAIR_AND_SMALL_CPU_TESTS
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_LUNA_REPAIR_HANDOFF_AND_EXECUTION_20260915
PLANNER_SESSION_ROLE: CODEX_PLANNER
FULL_EXPERIMENT_AUTHORIZATION: NOT_GRANTED_BY_THIS_REPAIR_REQUEST
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_CELL_RANK_REPAIR_20260915/report.md
PARENT_TASK: CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914
MODEL_DECISION: User explicitly selected the existing Luna Executor for these repairs

## 1. Authorization and recipient

[FACT] After Planner reviewed the implementation and proposed four concrete repairs plus integration tests, the user instructed "luna 에 넘겨", followed by "루나에 실행 맡겨". Execute those implementation repairs and the necessary small CPU tests. This is not authorization to launch the real B16/T10 preprocessing/reconstruction/evaluation experiment.

Recipient: existing thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID 18272, process creation /Date(1789362262182)/, workspace C:/Users/조상제/Desktop/CPA. User-selected Luna/xhigh supersedes the historical Astra/high wording for this task. Do not change the model, global configuration, peer bindings, or session. Do not spawn another AI or worker.

Planner verified the same live process identity, saved CLI thread model/effort gpt-5.6-luna/xhigh, and latest persisted turn_context at 2026-09-14T14:25:37.543Z with the same settings. Recent ThreadSettings rows were unavailable in logs_2.sqlite. This is saved-session evidence, not a claim of a freshly observed model setting event. Verify your actual runtime and identity before claiming; report any mismatch without executing research work.

Read AGENTS.md, its required project documents, the FULL current handoff, and the parent immutable plan .ai/execution-handoffs/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914_v1.md. Verify this task/version and delivered snapshot before changes. Claim only READY/unclaimed work, with your exact thread ID; mark RUNNING. Parent implementation was reported COMPLETED; this new task repairs the reviewed defects. Do not rerun that original task.

## 2. Fixed conditions and allowed edits

Preserve the scientific contract of the parent plan: deterministic signed INT4 [-7,7], observed q and scalar per-parameter-tensor FP32 s, existing nearest-even/zero/edge codec rules; VGG16 Conv frozen, all FC trained, single client, B16/T10, seed42; classifier.0.weight raw update; external uncentered rank16/cell projection, AP5 default, final V into original CPA25k and unchanged abs(S) into original FIA25k. No new observation, labels, replay, selection, loss, tuning, seed, arm, or iteration change. Scale stays a metric only; no midpoint/scale loss or rescaling. No MP threshold or noise-model change.

Original CPA source, whitening, loss and optimizer, existing FIA, baseline, codec and cell utilities remain byte-identical. Preserve all existing code/results outside the following six new implementation files:

- quantization_aware_cpa/qacpa/signed_cell_rank.py
- quantization_aware_cpa/qacpa/signed_cell_rank_runner.py
- quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py
- quantization_aware_cpa/tests/test_signed_cell_rank.py
- quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py
- quantization_aware_cpa/docs/signed_cell_rank_comparison.md

Also write this task's dedicated execution-results directory and update only claim/status/result fields of the active handoff. Preserve the immutable plan and parent report. No cleanup/revert of unrelated workspace changes. Do not modify RELAY, STATE, global role files or bridge bindings.

## 3. Required repairs

### R1: Canonical CPA input hash

[FACT/code review] signed_cell_rank.tensor_sha256 hashes the dtype string torch.float32, while upstream_naive._tensor_hash hashes numpy dtype float32. Runner line267 currently registers core.tensor_sha256(value), but actual CPA manifest uses _tensor_hash. Evaluator compares these distinct encodings and rejects the input fingerprint.

[PROPOSAL/authorized repair] Use the EXISTING upstream_naive._tensor_hash(value) in the NEW runner for the canonical CPA input registration/result. Do not edit upstream_naive. Keep preprocessor tensor/file hashes under their existing separate meanings; do not silently change historical hash formats or conflate file hashes and CPA fingerprints. Correct all relevant new-code readers/writers consistently.

Test acceptance: same input produces identical registered and actual canonical CPA manifest fingerprints. A modified tensor or forged/mismatched fingerprint is rejected. Exercise the actual canonical manifest/hash logic with the costly optimizer/model mocked; do not hide this defect by mocking both hash producers to an invented constant.

### R2: All16 figure layout and preservation of numerical results

[FACT/code review] evaluator._write_figures allocates compact (4,9), but offsets (truth_index % 4)*3 require 12 columns. At truth_index=3 the first index is9, out of bounds. evaluate currently calls plotting before writing comparison.json/report.

[PROPOSAL/authorized repair] Use compact (4,12), preserving four groups of original/naive/cell-rank per row and all16 images. Preserve the existing (16,3) full layout and fixed image pairing. Save completed numerical results atomically before plotting; record figure/report state separately. If plotting fails, retain the metrics and failure detail, return an appropriate failure/nonzero status, and do not label the overall evaluation COMPLETED. Mark full completion only after required artifacts exist. Never swallow plotting errors and claim success. Keep output separation/no-overwrite protections.

Test acceptance: both images render all16 synthetic examples without indexing failure. Injected savefig failure leaves readable numerical results and a truthful partial/failed status; no false overall success. Test actual matplotlib rendering on tiny synthetic images using Agg, without LPIPS downloads or real private data.

### R3: Correct consecutive-iterate movement

[FACT/code review] current loop passes a lagging previous variable into _record_step. From step2 onward movement_from_previous_V compares against a value two steps back.

[PROPOSAL/authorized repair] Compute consecutive movement with _movement(after, before), i.e. V_(t+1)-V_t. Remove redundant previous state if appropriate. Keep movement_from_V0 separately, with existing norm/count/zero-denominator semantics. Do not change the projection update itself.

Test acceptance: a multi-step (at least3) nontrivial tiny fixture matches explicitly retained consecutive iterates for relative L2, maxabs and changed-coordinate count. Initial/AP0 movement stays zero. Demonstrate finite/code/cell feasibility and no change to projected V solely from this logging repair.

### R4: Stage-specific execution status

[FACT/code review] runner.attack completes actual CPA/FIA but hardcodes full_experiment_status=NOT_RUN. The existing mocked test enforces that contradictory runtime output.

[PROPOSAL/authorized repair] Introduce consistent attack_status/evaluation_status fields in new runtime artifacts and their consumers. Completed attack awaiting evaluation means attack_status=COMPLETED, evaluation_status=NOT_RUN. Distinguish incomplete/failed stages and successful full evaluation. Keep implementation-task reporting FULL_EXPERIMENT_NOT_RUN because this repair task runs no real experiment. Mock test fixtures must not be presented as real research results. Preserve readability of existing naive baseline artifacts; adapt new evaluator schema handling explicitly without rewriting old files. Do not loosen completion/provenance checks to make tests pass.

Test acceptance: completed, pending and failed states distinguish actual stage outcomes; incomplete attacks cannot reach private evaluation or be accepted by the evaluator.

### R5: Complete integration and configuration consistency checks

[FACT/code review] new pipeline tests do not currently traverse a successful evaluator/figure/report path. _executed_settings checks only seed and iterations although the parent plan requires matching registered AND executed CPA/FIA settings.

[PROPOSAL/authorized repair] Add a successful CPU/mock path from preprocessing through mocked CPA/FIA artifacts to real provenance validation and final report/figure storage. Mock heavy model/optimizer/LPIPS operations only; keep canonical hash generation, validator, state handling and file writing real. Validate relevant executed CPA/FIA parameters against their registered expected settings using the original adapter's existing schema/mappings, including learning rate and loss/configuration values; retain seed, source/checkpoint/input hash, initialization hash and runtime checks. Missing required evidence must fail explicitly, not be guessed.

Add focused negative cases for input/hash tampering, registered/executed learning-rate mismatch, incomplete arm, and plotting failure. Retain private-load-before-validation and path-separation tests. Preserve all16 label-free offline matching and separation of evaluation-only truth from attack inputs. This repair does not authorize label-based replay or a new oracle selector.

## 4. Validation scope and commands

Execution environment: C:/miniconda/envs/DLG/python.exe. Small synchronous CPU tests only: synthetic matrices at most32x64, small temporary artifacts and all16 tiny synthetic RGB images; mocked full CPA/FIA and LPIPS computations. No real observation decompression/preprocessing, private tensor loads, large matrix diagnostics, GPU tests, training, reconstruction, benchmark, downloads or experiment callback. Existing source/JSON/log reads are allowed. Bound costs by those tiny fixtures; if full-size execution is needed, stop and report rather than expanding scope.

First reproduce each reviewed defect in the relevant regression/integration test; then repair and run the focused suite. Record actual pre-fix failures and post-fix results rather than asserting tests were run without retained output. Suggested targeted commands from repo root:

    C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py -q
    C:/miniconda/envs/DLG/python.exe -m pytest quantized_update_cpa/tests/test_cell_utils.py quantization_aware_cpa/tests/test_upstream_naive.py quantization_aware_cpa/tests/test_signed_baseline_provenance.py -q

Inspect test scope first and document any needed narrowing to stay CPU-only. Store stdout/stderr or JUnit reports under this task's result directory. Do not count overlapping test invocations as unique passes. Compare protected-source hashes before/after to the parent results.json protected_source_hashes_current values; discrepancies are NEEDS_REVIEW, not permission to rewrite originals. Do not broaden checks after relevant tests pass without new concerns.

## 5. Report and stopping rule

Write .ai/execution-results/CPA_SIGNED_CELL_RANK_REPAIR_20260915/report.md and results.json, including [FACT]/[INFERENCE]/[PROPOSAL]/[UNKNOWN], R1-R5 disposition, changed files, retained test evidence, actual commands/counts/runtime, protected hashes, status schema, numerical-result/plot-failure behavior and remaining limitations. Full research experiment status remains FULL_EXPERIMENT_NOT_RUN; image quality and full-size resource costs remain UNKNOWN.

Mark this task COMPLETED only when the bounded repairs, appropriate tests and documentation are finished; otherwise FAILED/NEEDS_REVIEW with specific outstanding items. Display your conclusion in the existing Luna CLI and end your turn. Do not monitor, spawn a worker, start the full comparison, or reopen the Claude relay.

The parent task's final notification to the historical Planner binding failed with "no rollout found". Do not blindly retry that stale binding, change it, or start a replacement. Follow the authorized one-time completion policy only with an independently verified current Planner destination; otherwise preserve the result report, state notification unconfirmed, and finish. Planner will review on user request; there is no periodic monitoring.
