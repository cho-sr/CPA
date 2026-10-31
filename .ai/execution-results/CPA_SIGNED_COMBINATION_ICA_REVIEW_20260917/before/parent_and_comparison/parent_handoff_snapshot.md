# Run the repaired ICA-only pipeline and compare frozen image results

TASK_ID: CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: ONE_REGISTERED_REAL_PIPELINE_AND_OFFLINE_COMPARISON
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_RUN_COMBINATION_ICA_EXPERIMENT_20260916
REVIEW_REFERENCE: PLANNER_READINESS_REVIEW_OF_COMPLETED_IMPLEMENTATION_REPAIR
PARENT_TASK_ID: CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916/execution_summary.json
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_combination_ica_B16_T10_20260916

## Explicit authorization and roles

The user asked whether experiments or further validation were needed. Planner reviewed the repaired sources, actual test logs (31 focused and 5 unchanged contracts passed), schemas and path checks, and described: FastICA -> prior-based gradient-descent feature separation -> Direct FIA -> real numerical preflight -> 100 image steps with actual-update cell loss -> comparison with existing results. The user then explicitly said "실험 ㄱㄱ". This authorizes that concrete real experiment, necessary bounded execution/evaluation adapters and focused validation, final offline image evaluation and reporting. Do not request this permission again. The implementation-only no-run restriction in the parent handoffs is superseded ONLY for this registered run.

This is one new separator run, one new 25,000-step Direct FIA run, and one new 100-step image-cell refinement. Reuse the two archived controls. No control retraining, extra arms/seeds, hyperparameter search, calibration, label swaps, new observation, generator, model change, randomized dequantization, source-selection experiment, extra continuation or automatic rerun.

Assigned existing user-designated Executor: Luna/xhigh thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID 14236, process creation 1789518279030 ms. Return to existing Planner thread 01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID 24972, creation 1789518271364 ms. Reverify exact identity/role/runtime and handoff hash before claiming. No new AI session or peer/model rebinding. Do not reopen closed Claude discussion.

Read the two preserved implementation/repair plans and the repair report. Their scientific settings remain unchanged; execute the repaired staged interfaces. Planner confirmed all six new-file hashes and 13 protected / five shared source hashes still match the repair completion record and both new output roots do not exist.

## Registered scientific conditions

- Same public VGG16 starting checkpoint, frozen convolutional extractor, all six FC parameter tensors trained, one client's local update, batch of 16 images repeated in the same order for 10 local SGD steps, learning rate .001, momentum/weight decay/dropout zero. Delta means theta_after10 minus theta_start. This is not a multi-client aggregated update.
- Same deterministic per-tensor signed symmetric INT4 packet, integer codes -7..7, existing scale and round/tie/cap/zero-cell rules. No regeneration or codec conversion.
- Public observation: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation. Manifest SHA256: 9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69. Bind all six packet files against the repair integrity record and existing screen registration, not just the manifest name.
- Checkpoint: C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth. SHA256: 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0. No downloads.
- Separator input is q*s for classifier.0.weight [4096,25088], with 16 sources. Run the registered default CPU FP32 separator, seed 42, one CPU thread, one shared whitening basis, corrected parallel FastICA max 200 / tolerance 1e-6, followed by 1,000 plain SGD steps at .001 and the four existing prior weights. Preserve the exact core and canonical config. Finite max-iteration FastICA nonconvergence is flagged and may proceed under the existing core contract; nonfinite/unusable-rank errors fail. No changing precision, whitening, device, initialization or retries to rescue a result.
- Save native signed S_hat [16,25088] and exact target_z=abs(S_hat). No truth alignment or generator standardization in the attack.
- Direct FIA: unchanged upstream adapter, CUDA, seed 42, 25,000 Adam steps, learning rate .1 with existing cosine schedule, feature cosine weight 1, normalized TV weight .1. Preserve its own recorded runtime choices and exact new target.
- Image-cell refinement: unchanged run_arm, CUDA, 100 pixel Adam steps, learning rate .001 -> .0001, feature weight 1, TV .1, fixed cell coefficient .24202997768915402. The constraint acts through candidate image -> frozen features -> actual 10-step FC update -> all six unprojected update tensors -> cell loss. Do not substitute a surrogate matrix constraint.
- Recompute the existing public coverage-based label assignment from the NEW Direct FIA candidate and observed positive final-bias codes; then freeze it. Keep the existing B_H1_combination_ica identifier internally; explain its meaning in ordinary language in user-facing reports. Never attach old slot labels to a new source permutation. No true labels in attack or evaluation decisions.

## Frozen comparison sets and identities

Evaluate exactly these four sets of 16 images, with clear human-readable labels:

1. original_separator_direct: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive/candidate.pt; SHA256 fe683f68013896a407a38130ee54ef9dc799a9ca1abf7c4f08efeefea7d91751.
2. original_separator_cell: quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916/B_H1_high/candidate.pt; SHA256 b352c3bde7db87ff00ecbbbd4715c5c865ada662c4dcf88b1e116f5369ec9d5a.
3. combination_separator_direct: new public root/direct_fia/candidate.pt.
4. combination_separator_cell: new public root/actual_update_refinement/B_H1_combination_ica/candidate.pt.

Old screen registration.json SHA256 25ae8a5ef4d2e38110709b1fd003d1aa09b823dd63ed2c0a41391d49f79bbe69; summary.json SHA256 dca2618457914fc38298041dd0270c6f113fb899f5c7f88af6a48278c8557a00; integrity.json SHA256 a98f0b2e5526fe305a147f69c5e51d13478252eda927a8349b7f12f6fa530d65. Authenticate their structured contents and actual candidate/input hashes. Preserve all archived controls and metadata byte-for-byte. Old canonical start/final coverage-label profiles are already in that summary; reuse them, no extra control replay or optimization.

Source pins at registration:
signed_combination_ica.py: 1c7482e1ed77ca4641cd8822ebb6046c5150e794c3838aacc3e3d575f3d992d6
signed_combination_ica_runner.py: a7ca10cb59f59c76e50c3c20a564ca6d13ee1795385fade62b302a44854a124f
test_signed_combination_ica_pipeline.py: 41e207ab5f3bff106df19ab15a8dacb3a14b6744f3c973acbffddc57cb50d401
All other expected source/input hashes are in .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916/source_integrity.json. Snapshot the complete actual source, checkpoint, packet and consumed-control hashes before execution and verify after the public stages and after evaluation, including the new experiment/evaluator code. A mismatch prevents a successful report.

## Prepare complete execution and evaluation before launching

Use C:/miniconda/envs/DLG/python.exe. Preserve repaired core/runner and all historical algorithms/evaluators. Add only isolated orchestration/evaluation files such as qacpa/signed_combination_ica_experiment.py, qacpa/signed_combination_ica_evaluation.py, a narrowly scoped test file and task-local wrapper/report files. Ordinary adapter/schema/reporting defects may be repaired within this scope with archived before bytes and focused tests; do not change the scientific method or silently rerun completed optimizations. If the accepted method itself would need changing, preserve evidence and report the blocker.

The staged runner intentionally retains its implementation TASK_ID CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916. A new experiment-level registration must explicitly bind this outer execution ID to those stage schemas, source hashes/configs and actual artifacts. Do not rewrite stage IDs or pretend a new artifact is an old baseline in order to pass an existing evaluator. The isolated offline adapter must validate the new method and its actual explicit schema.

Before any real computation, prepare the complete sequential wrapper, public provenance gate, offline evaluator and final reporting. Register paths, exact configs, source/input/control hashes, metrics, comparison directions, stop rules and resources before seeing new results. New output roots must be absent; preserve no-overwrite behavior. Put experiment-level records in the task results directory so the separate command can still require its public root not to exist.

Necessary focused synthetic validation of NEW wrapper/evaluator logic only: strict completed-stage/source/target/candidate/config/100-step binding; public gates run before private loading; six-tensor preflight/profile fields accepted in their real schema; exact infeasibility still permits honest completed evaluation; changed hash/missing stage/numerical-path failure blocks private entry; stage failure prevents subsequent optimization; metric correspondence reused in figures. Reuse tested metric primitives. Do not repeat the entire unchanged passing suite without a concrete change requiring it. No extra real-data pilot or optimizer step is needed before the registered run.

## Execute the three stages once

Set PYTHONPATH to workspace/quantization_aware_cpa plus workspace; working directory is the workspace. Use the documented commands in separate processes, in this order, with the usual deterministic environment established before CUDA initialization:

    C:/miniconda/envs/DLG/python.exe -m qacpa.signed_combination_ica_runner separate --device cpu --output quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916

    C:/miniconda/envs/DLG/python.exe -m qacpa.signed_combination_ica_runner invert --device cuda --source-root quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916 --output quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916/direct_fia

    C:/miniconda/envs/DLG/python.exe -m qacpa.signed_combination_ica_runner refine --device cuda --source-root quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916 --candidate-root quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916/direct_fia --output quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916/actual_update_refinement

Record each exit/status/time. Authenticate each completed stage before the next. Actual refinement preflight must pass feature grad/no-grad equality and canonical/adjoint terminal-delta equality for all six FC tensors. Save successful preflight and canonical start profile BEFORE the 100 steps; authenticate the final candidate and save canonical final profile afterward. Preserve these incremental artifacts on downstream failure. Record separation convergence/losses, target statistics, feature-target fit, image/feature norms and cell profiles already emitted by the existing stages.

Preflight numerical mismatch, invalid/nonfinite artifacts, changed hashes, resource exhaustion or incomplete expected iterations stops dependent work. False exact cell feasibility is a measured scientific outcome, not a software error or a reason to suppress offline evaluation when the public completion/provenance gates otherwise pass.

## Offline image evaluation and interpretation

Only after all new public stages complete and candidates/profiles/config/steps/integrity authenticate, freeze and hash the four candidate sets and enter the private evaluator. Attack processes must finish before the evaluator starts. Private truth path is quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt. Consume x only; labels_consumed=false and true_updates_consumed=false. Disclose bundled fields materialized by torch.load as in the existing evaluator; do not claim they were never deserialized if the file bundles them. No private feedback returns to attack, candidate selection or parameter choices.

Use unchanged existing evaluation semantics: pixel-MSE Hungarian matching ONCE per 16-image candidate set, reuse that exact correspondence for per-image MSE/PSNR/SSIM/VGG LPIPS and the aligned figure. Evaluate every image; no best-looking subset, rematching per metric or tuned image processing. Reuse metric helpers from the existing evaluation modules while keeping their method-specific validation intact. Use the existing cached metric models, no download.

Compare the two re-evaluated old controls to the existing machine-readable comparison at quantization_aware_cpa/results_private/signed_e2e_h1_nofia_B16_T10_20260916/comparison.json, rows x0 and B_H1_high. This is private evaluation-only information; do not load it into attack or public run registration. Keep the established absolute mean-metric reproduction tolerance 2e-8 for each of MSE, PSNR, SSIM, LPIPS. Report actual discrepancies; do not loosen tolerance or tune matching. If reproduction fails, preserve the computed metrics and mark the comparison NEEDS_REVIEW rather than making a scientific conclusion or rerunning the attack.

Save private comparison.json, a concise report, per-image values/correspondence and a figure showing truth and all four sets in the SAME truth row order. Candidate hashes and matching provenance must be included. Private output stays under the new private root; the public attack artifacts never gain truth images/labels or oracle source alignment.

Predeclared comparisons: (a) new direct versus old direct measures adoption of the complete separator package; (b) new final versus new direct measures the observed effect of the fixed cell refinement from that new start; (c) new final versus old final compares the complete pipelines. Report all mean/per-image metrics and differences. MSE/LPIPS lower and PSNR/SSIM higher are the favorable directions; mixed or negligible results are reported as such. No invented reconstruction-success threshold or generalization claim. This is one fixed batch and seed; the separator iteration/optimizer packages differ, so it is not an equal-compute or single-prior causal ablation. New candidate-derived label slots may differ; disclose the assignments and avoid claiming cross-pipeline cell differences isolate separation alone.

Report exact code mismatch counts/fractions and bounded-cell violations for ALL six tensors and overall, plus penalty and scale diagnostics, for old and new start/final profiles. Exact feasibility means zero fixed-observed-scale code mismatches AND zero bounded-cell violations across all six. Candidate-scale equality remains a separate diagnostic. Image quality and feasibility are independent conclusions; a smaller cell penalty alone is not successful private-image recovery.

## Resources, failure handling and completion

Register a total 45-minute wall-clock cap for the actual sequential public computation plus private evaluation, excluding development/tests and idle time awaiting review. The old naive separator+25,000-step FIA took 2137.2704 seconds; this is a historical anchor, not a guaranteed duration for the new path. Retain the existing 12 GiB peak CUDA allocated cap; record CPU/GPU time and peak allocated/reserved memory when available. Do not change device, thread count, iterations or precision to fit the cap. The wrapper must enforce its total budget while retaining the existing per-refinement Clock/resource guards. Use controlled termination of only owned experiment children if the registered hard cap is reached, record partial outputs, and do not restart or extend the budget. No model polling.

Launch the prepared wrapper hidden and persistent so ending the CLI turn does not kill it. Record PID/start and register the existing one-time non-model process-exit callback to the SAME Executor. Do one initial launch/receipt check, then end the Executor turn. On exit, finish the authorized validation/reporting. A reporting-only repair may use authenticated frozen artifacts without new optimizer steps; never rewrite a failed process exit as a successful one. On scientific/numerical failure report the precise last successful stage and evidence without inventing a conclusion.

Results directory: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_EXPERIMENT_20260916. Required report.md and execution_summary.json include exact commands/tests, source diffs if any, phase exit codes/timings/budget, hashes/integrity, actual completed iterations, numerical preflight, all endpoint feasibility outcomes, private phase status and four-set image metrics/figure paths. Scientific quality becomes evaluated only when real evaluation occurred; COMPLETED means required execution and reporting completed, not recovery success. Ensure referenced files exist before sending one final notification to the verified Planner, then end. No further run follows automatically.
