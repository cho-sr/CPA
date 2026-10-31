# Compare 4000-step ICA prior refinement followed by FIA against the frozen 1000-step result

TASK_ID: CPA_SIGNED_ICA4000_FIA_20260917
PLAN_VERSION: 1
STATUS: COMPLETED
TASK_KIND: ONE_ITERATION_ABLATION_WITH_DIRECT_FIA_AND_OFFLINE_EVALUATION
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_ICA_4000_THEN_FIA_20260917
REVIEW_REFERENCE: USER_SELECTED_ITERATION_COMPARISON_AFTER_SAVED_LOG_REVIEW
PARENT_TASK_ID: CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_ICA4000_FIA_20260917/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_ICA4000_FIA_20260917/execution_summary.json
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_ica4000_fia_B16_T10_20260917
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_ica4000_fia_B16_T10_20260917

## Explicit authorization and scope

The user questioned whether 1000 gradient-descent ICA steps were sufficient, then requested "FIA 사용해봐". After clarification that the old run already used FIA25000, the user explicitly selected "경사하강 ICA를 4,000회로 늘린 뒤 FIA 적용". This authorizes one new 4000-step separator followed by unchanged25000-step Direct FIA, necessary isolated adapters/tests and the final paired offline comparison. Do not ask again or stop at a proposal. The scientific independent variable is ONLY CombinationICAConfig.refinement_iterations:1000 -> 4000.

Do not run image-cell optimization in this task. Do not invoke FC replay, label inference/search, endpoint feasibility profiling, a generator, other seed/arm, new observation, tuning, intermediate-image selection or extra continuation. Use the archived1000-step Direct FIA image set as the control; no control retraining. This experiment evaluates separator iteration count followed by FIA. It makes no new actual-update cell-feasibility claim.

Existing user-designated Executor: Luna/xhigh thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, creation1789518279030ms. Return to existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID24972, creation1789518271364ms. Verify role/runtime/process identity and this READY handoff hash before claiming; no model/peer rebinding or new AI session.

## Fixed settings and evidence

Preserve the existing model and observation: pretrained VGG16 checkpoint, frozen convolutional features, observed single-client B16/T10 FC-only SGD update, deterministic per-tensor signed symmetric INT4. Local training rules and packet are unchanged. Input to separation is exactly classifier.0.weight decoded as q*s, shape[4096,25088], source count16. Do not regenerate observations or change codec/scale/rounding/sign conventions.

Observation: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation.
Manifest SHA256:9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69.
Checkpoint:C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth.
Checkpoint SHA256:397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0.
Authenticate all packet/source/checkpoint hashes using the completed parent registration and repair integrity record before computation, and verify unchanged afterward.

Separator: seed42, CPU FP32, one CPU thread, configure_runtime(42,threads=1); unchanged single shared whitening basis and corrected symmetric FastICA, max200/tolerance1e-6/identity initialization. Reuse the UNCHANGED pure core signed_combination_ica.separate_observation with CombinationICAConfig(refinement_iterations=4000), device='cpu'. All other config fields must equal the original default dictionary exactly. Plain SGD lr.001, momentum0, weight_decay0, no scheduler. Keep the four weights .0935827857091549/.2513674104161625/.05811961446881231/17.94138146247377 and existing numerical formulas. No early stopping or private selection; use the4000th iterate. Finite FastICA max-iteration behavior remains the core's existing contract, while nonfinite/unusable-rank results fail.

Output: native signed S_hat[16,25088] and EXACT target_z=abs(S_hat), contiguous CPU FP32. No extra sign/permutation alignment, normalization, standardization, rank/cell preprocessing, learned mixing model or truth input in attack.

Direct FIA: unchanged upstream_naive.run_upstream_fia / original Direct implementation; use the actual new target, seed42, CUDA FP32,25000 Adam iterations, lr.1, existing cosine scheduler, feature cosine weight1, normalized TV weight.1, log_every5000. Establish the same deterministic pre-adapter runtime as the old run, preserve and record the adapter's own runtime changes separately. No changed inversion initialization or extended FIA budget.

## Control and provenance pins

Control root:quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916.
- registration.json SHA256:70f48861153e5155c5d9fe1c4a20851dc8b4ca3ee9317dbef362148bf6d3d484
- result.json SHA256:80946a388a649ebb8a99bfcbdb9301659788d89e1cb7562c441dca43f6b4e36e
- sources.pt SHA256:408099e99bb6ad1d8b721ff518c040ec9bec332827f0f36d3b9715ffeb4af5a2
- refinement_log.json SHA256:c20f6a62b2e5161c63fc89fcfcbe9e2dee3221eb3956262c8dc7c6a7cf8d9d2f
- direct_fia/candidate.pt SHA256:4efc29ad08b1ea8b0cee6991de6f185839708c2c72aaafb4fedfb307b5e680ce
- direct_fia/result.json SHA256:91c05c281135ffff7dc2fb40fa14e213675d371f72c528036ac8b3e69515a6da
- direct_fia/registration.json SHA256:9b4d66d5c35588df5d9e4e7fae5362643569908177b65514348e32584c239657

Use the successfully reviewed metric control ONLY in the offline phase:
quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916/review_20260917/comparison.json, row combination_separator_direct;
SHA256:bc07e948dada0aba09e14b4b7ff8fa65d7ba988315fcaa1801c43cb7c7880d04.
Do not use the earlier evaluator's unmatched runtime or private metric values for attack decisions.

Preserve core SHA2561c7482e1ed77ca4641cd8822ebb6046c5150e794c3838aacc3e3d575f3d992d6 and old staged runner SHA256a7ca10cb59f59c76e50c3c20a564ca6d13ee1795385fade62b302a44854a124f. All previously protected/shared/official CPA/FIA/codec/replay/evaluator files and historical artifacts remain unchanged. Snapshot all reused and new execution sources before/after, not only the old protected list.

## Necessary isolated integration

The old staged runner intentionally authenticates only its1000-step preset and old task identity. Do not loosen its validators, monkeypatch its globals, or falsify the new artifact config to pass them. Add an isolated explicitly4000-step runner/evaluator/wrapper, for example qacpa/signed_ica4000_fia_runner.py, signed_ica4000_fia_evaluation.py and focused tests, plus task-local launch/report files. Reuse the pure core's existing configurable iteration field and generic unchanged FIA/metric helpers. No mathematical core change is needed; its docstring mentioning1000 does not override the implemented config loop.

The new schemas must bind this task/variant, exactly4000 refinement steps, all unchanged numerical settings, output shapes/finite values/exact abs target, file/tensor hashes, completed-stage statuses and the source->FIA target->candidate chain. Distinguish new4000 artifacts from old1000 artifacts. No-overwrite new output roots. Register the full method/config/paths/source-input-control hashes/budget/evaluation rules before real computation.

Meaningful focused synthetic validation before launch: only refinement_iterations differs between canonical configs; the real pure core honors a supplied iteration count on a small fixture; finite[16,25088] signed/abs artifact round trip; actual writer/loader/bridge passes the precise new target to the unchanged FIA adapter; changed config/hash/shape/status/output collision fails; no cell/replay entry point is reachable in orchestration; public gate precedes private loading; evaluator establishes the same canonical runtime. Mock expensive model/packet/metric compute. Run only relevant new/affected checks, py_compile and CLI help. No extra full-size pilot or repeated unchanged suite without need.

Prepare the complete sequential separator -> FIA -> offline comparison wrapper before launching. Use C:/miniconda/envs/DLG/python.exe and record exact commands.

## One real run and prefix consistency check

Call the unchanged separator once for4000 steps from the same FastICA initialization rules. Before launching FIA, compare the first1000 saved refinement log records to the authenticated old1000 records, including iteration, all loss terms and gradient norm, parsed as structured values. The old experiment used the same deterministic CPU core/runtime and stopped solely at the iteration budget. Require exact equality of this common prefix; report FastICA convergence/iterations/residual against the old result as well. A mismatch stops dependent FIA/evaluation and is reported as an execution-comparability problem; do not silently align, round or retry. This is a read-only check of the new continuous run, not another1000-step arm or checkpoint selection. Do not claim1000-step tensor equality unless an actual corresponding tensor was saved and compared.

Persist the4000-step result and complete log, all four component losses and gradient norms, exact source/target hashes and prefix check before FIA. Only the final4000 target is inverted. Complete the single25000-step FIA, freeze/hash candidate.pt pixels[16,3,224,224] in[0,1], logs, settings and integrity. Stage failure prevents later stages; preserve completed artifacts and truthful exits. Ordinary wrapper/evaluation repairs are authorized without rerunning successful optimizations or changing the method.

## Offline paired evaluation and decision rules

After public source/config/prefix/FIA completion/integrity gates pass, start a separate evaluator process. BEFORE metric computation invoke the unchanged configure_runtime(42,threads=1) and record actual settings; this is the canonical runtime established in the completed scientific review. Private x-only truth is the same results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt. No private labels, true updates or true feature tensors are consumed. Disclose bundled deserialization honestly. Private outputs stay under the new private root; no private feedback to attack.

Evaluate exactly two frozen sets, all16 images each: old GD1000+FIA25000 and new GD4000+FIA25000. Use unchanged pixel-MSE Hungarian matching once per set, then reuse those correspondences for MSE/PSNR/SSIM/VGG LPIPS and an aligned truth/control/new figure. Preserve cached model/metric weights and normalization; no downloads. Per-image results and mean values are both required.

Re-evaluated old1000 control must reproduce its reviewed row's four mean metrics within the existing absolute2e-8 tolerance; report correspondences and per-image residuals too. A failure preserves metrics with NEEDS_REVIEW; do not change tolerance, repeat runtime settings or rerun attack to pass. Frozen-image reporting-only repairs remain within scope if needed, with honest budget and provenance.

Scientific comparison is new4000 minus old1000. Favorable directions: MSE and LPIPS lower, PSNR and SSIM higher. Report differences and number of improved/worsened images. A lower ICA prior loss alone does not establish better separated private features or images. Consistent image improvement supports that the old iteration budget mattered for this fixed setup; mixed/no improvement does not establish iteration count as the principal bottleneck. No invented success threshold, generalization, full convergence guarantee, cell-feasibility assertion or claim about extending FIA/cell steps. Do not pick a preferred checkpoint using private results.

Report the separator's full loss trajectory with1000/2000/3000/4000 values (label pre-step versus post-step correctly), gradient norm trend, FastICA convergence and runtime, final target statistics, FIA target fit and image metrics. These observations answer whether extra ICA optimization meaningfully translates to reconstruction quality.

## Budget and completion

This is a NEW explicitly authorized experiment, with a fresh45-minute cap for sequential real separation/FIA/private evaluation and12GiB CUDA allocated cap. Development/focused tests and idle time are excluded. Record stage durations separately from cumulative elapsed time, resource peaks and all exits. Do not extend budgets, change device/precision/threads/iterations to rescue a failure, add arms or automatically restart. The previous FIA took about35.4 minutes; treat it as a cost anchor, not a guarantee.

Launch hidden and persistent. Verify exact wrapper PID/start, register one existing non-model process-exit callback to the SAME Executor, verify its receipt once and end the turn. Prior callback failure was UTF-8 BOM in temporary peer JSON: use valid BOM-free UTF-8 for any task-local transport file and preserve shared bindings. No model polling or new monitoring agents.

Write complete report.md and execution_summary.json under .ai/execution-results/CPA_SIGNED_ICA4000_FIA_20260917 with test evidence, source changes, registration/hashes/integrity, prefix check, actual4000/25000 completion, resource/timing/exit status, both-set metrics, reproduction result, limitations and figure paths. Preserve old reports and archive the previous completed handoff. Mark scientific quality evaluated only after actual evaluation. Ensure all files exist, send one final result to verified Planner and end. No additional experiment follows automatically.
