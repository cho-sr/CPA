# Repair the completed ICA-only implementation's executable integration

TASK_ID: CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: ROUTINE_IMPLEMENTATION_REPAIR_AND_SYNTHETIC_INTEGRATION_TESTS
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_PRIOR_ICA_ONLY_ADOPTION_INCLUDES_IMPLEMENTATION_REPAIRS_20260916
PARENT_TASK_ID: CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916
REVIEW_REFERENCE: PLANNER_STATIC_REVIEW_AFTER_COMPLETION_NOTICE
FULL_RESEARCH_RUN_STATUS: NOT_AUTHORIZED
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916/execution_summary.json

## Authorized scope

The user authorized adopting ICA-GA's ICA stages only while keeping the model: "우리도 저기 방법 차용하자 Generator model 은 버리고 ICA 쓰는 방식만, 모델은 유지". The parent implementation was reported complete. The user subsequently said "다 했대", and Planner reviewed readiness. This review found an unconditional integration failure and missing explicitly required checks. Repairing them and testing the actual staged interfaces is ordinary completion of the already-authorized implementation, not a new method or research experiment. Do not ask for repeat implementation approval.

Recipient remains the existing user-designated Luna/xhigh Executor thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236/start1789518279030ms. Return one final report to existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID24972/start1789518271364ms. Verify actual role/runtime/identity and handoff hash before claim; do not change models or bindings or create a CLI/agent.

Read the full parent scope at .ai/execution-handoffs/CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916_v1.md. All its scientific settings, no-generator/model-preservation contract and research-not-run boundary remain unchanged. Only synthetic CPU validation and mock compute adapters are allowed. No real B16/T10 separation, FIA, image replay, private tensor loading/evaluation, tuning, downloads, new seed/arm or real research run.

## Confirmed findings requiring repair

1. **Wrong feature dimension blocks inversion.** In signed_combination_ica_runner.py:82,116,215, the schema uses [16,4096*25088]. The input update is [4096,25088]; after source separation, each of16 sources has25088 coordinates. The correct output and FIA target are [16,25088]. The core produces that correct shape, but `_source_registration` declares the wrong one and `_load_sources` rejects the actual tensor. Therefore the documented separate->invert sequence cannot work. Derive one canonical FEATURE_DIM from the actual attack-layer input width or existing signed_e2e_runner.FEATURE_DIM, and use it consistently in writer, loader, result and registration checks. Reject invalid shapes without allocating the erroneous6.6GB-shaped fixture.

2. **The required actual-update endpoint checks are absent.** `refine` invokes `compare_replay_paths` and `run_arm`, but never `canonical_profile`. `run_arm` returns image feature/TV metrics and the candidate; it does not perform final code/cell feasibility assessment. Implement the parent-required start and final canonical profiles over all6 FC tensors, preserving exact mismatch/violation counts and separate scale diagnostics. Do not substitute preflight terminal equality for feasibility. Exact infeasibility is a scientific outcome, not a software failure, and must be reported honestly. This task tests the orchestration with mocks only.

3. **Preflight evidence is persisted too late.** Currently `preflight.json` is written after the full100-step refiner returns. Persist a successful preflight immediately before optimization; persist the start profile before optimization and final profile immediately after final-candidate authentication. Preserve existing evidence if a later stage fails.

4. **Label metadata is misreported.** Passing name `INT4_combination_ica` to unchanged `run_arm` makes its `name.split('_')[1]` label field equal `combination`, even though the actual label rule is the coverage-based hypothesis. Use an explicit compatible arm identifier such as `B_H1_combination_ica` while keeping top-level method `INT4_combination_ica`, or an equally explicit truthful adapter metadata solution. Do not edit historical run_arm. Labels themselves must still be recomputed from the NEW public candidate and then frozen.

5. **Runtime/configuration and integrity guards need completion.** `separate` records `deterministic_expected: true` but does not configure that runtime, and `invert` does not establish the same pre-adapter TF32/CUBLAS runtime as the signed baseline. Explicitly establish seed42 and the existing deterministic FP32 setup before computation, and record actual flags. Preserve Direct FIA's own documented runtime choices after entering its unchanged adapter; do not claim the flags stayed constant when that adapter changes them. Bind the scientific separator/FIA presets and completed-stage schemas during loading, and reject unsupported/mutated configurations rather than accepting any `config` dictionary. Keep flexible small-fixture configuration only in the pure tensor core.

The parent required hashes of protected/shared sources and inputs before/after each future stage. These are presently missing from the stage completion path (the implementation-time hash audit is a different check). Add consistent pre/post stage integrity checks, including the consumed source/candidate artifacts and declared checkpoint. A mismatch must prevent a successful completion record. Retain the known public observation manifest SHA2569b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69 and checkpoint SHA256397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 as the authorized input identity. Validate them at the appropriate boundary, not merely record whichever input happens to be there.

## Why the passed tests missed this

The retained logs do show20 focused tests and5 existing upstream tests passed. However test_signed_combination_ica_pipeline.py mainly invokes `invoke_direct_fia` and `invoke_actual_update_refinement` with fake callbacks, bypassing the real artifact writer/loader and full `separate`, `invert`, `refine` orchestration. Thus those tests establish argument forwarding, not the documented pipeline's executability. Keep that distinction in the correction report; do not delete or rewrite the old pass logs or pretend the real experiment failed.

## Required regression validation

- Valid signed source tensors shaped [16,25088], exact abs targets and real registration/result serialization must round-trip through the actual new writer and loader. A synthetic tensor of this source size is small and allowed; the huge observation matrix and VGG forward are not needed. Check incorrect source dimensions and nonfinite/negative/mismatched targets.
- Exercise the real separate->invert orchestration on temporary synthetic public fixtures by mocking only expensive packet computation/ICA/FIA adapters as needed. The actual artifact validation and stage schemas must remain active. Confirm the next stage consumes the precise saved target. Include missing and changed hashes, method/stage/status/config, incomplete result, and output-collision failures.
- Exercise the real refinement orchestration with synthetic candidate artifacts and mocked model/device/replay compute boundaries. Retain production CUDA/numerical guards; do not weaken them to make tests pass. Assert ordered calls and persisted files: runtime setup -> authenticated inputs -> new-candidate coverage assignment -> successful preflight saved -> start profile saved -> unchanged actual-image refiner -> final candidate authenticated -> final profile saved -> integrity verified -> successful result. No private or generator dependencies.
- Assert all6 tensors and exact code/cell fields survive into the final report, and both feasible/infeasible mocked outcomes are represented accurately. A false feasibility outcome must not become a claim of success, while successful software completion is still possible.
- Inject a post-preflight/refinement failure and confirm earlier evidence persists and status does not become COMPLETED. Inject changed public/source/candidate/protected-source hashes and verify rejection.
- Verify the lower-level arm's label hypothesis metadata correctly identifies the actual frozen coverage-based assignment while method identity remains the adopted ICA method.
- Keep the previously passing core numerical and unchanged-adapter tests; add focused regression coverage and run only relevant synthetic suites. Use C:/miniconda/envs/DLG/python.exe. Do not run GPU/research/private validation to prove this fix.

## File boundaries, provenance and completion

Allowed edits: parent-added signed_combination_ica_runner.py, its pipeline/core tests only as necessary, and signed_combination_ica.md. The ICA method, coefficients, iterations, whitening, old signed_e2e/FIA/codec/replay implementations and all historical research artifacts remain unchanged. The numerical core is expected not to need changes for these findings; if an ordinary interface issue truly requires a small core edit, document it explicitly without changing scientific behavior. Archive exact pre-repair bytes/hashes of edited parent files and the parent completed handoff/report under the new task's before/ directory. Preserve the old completion notice/report as historical evidence and write a new superseding readiness report.

Current pre-repair reference hashes:
signed_combination_ica.py: 1c7482e1ed77ca4641cd8822ebb6046c5150e794c3838aacc3e3d575f3d992d6
signed_combination_ica_runner.py: 00394ff2d5f056156d3b572bb02e6e817b2a9f39b5ab343be7442044cdea6020
test_signed_combination_ica_pipeline.py: 049b318b3ba04449a9bbb7224f5a79590e87a0bd0693d4bd96391b9254684354

Correct docs to describe supported executable stages and their exact outputs. No scientific quality claims. Report precise fixes, regression commands/results, protected/source/input integrity and remaining limitations. Continue ordinary repair until the accepted implementation scope is complete; no permission question is needed for these defects. A separately authorized real experiment remains future work. Follow WORK_MODE: no polling, no new models; send one final result to the verified Planner after all referenced files exist, then end.
