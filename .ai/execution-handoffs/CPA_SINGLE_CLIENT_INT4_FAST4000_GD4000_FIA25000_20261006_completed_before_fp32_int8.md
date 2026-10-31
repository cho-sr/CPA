# Single-client INT4 update: FastICA4000 -> GD-ICA4000 -> Direct FIA25000

TASK_ID: CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006
PLAN_VERSION: 1
STATUS: COMPLETED
ASSIGNED_ROLE: CODEX_EXECUTOR
EXECUTOR_TARGET: CURRENT_CONVERSATION_EXPLICIT_USER_EXCEPTION
REQUIRED_MODEL: CURRENT_SESSION_UNCHANGED
CLAIMED_BY: CODEX_EXECUTOR:01a11025-fc1c-7e53-bea5-3cc32a1bae01
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_RUN_THIS_EXPERIMENT_HERE_20261006
DELIVERY_STATE: DIRECT_USER_ASSIGNMENT_NO_LUNA_QUEUE

## User instruction and role exception

The user requested "ICA 각 4천번씩 FIA 25k, 4비트 업데이트에 대해서 해줘 single client update 에 대해서", then explicitly instructed "그냥 여기서 실험 넣어줘" after being informed that the historical Luna execution session was closed. This conversation is CODEX_EXECUTOR for this concrete task; no model settings or historical peer bindings are changed. No repeat approval is needed. No automatic Planner relay response or extra research is authorized.

## Fixed scope

One existing single-client B16/T10 FC-only SGD update: VGG16 pretrained checkpoint, Conv frozen, all six FC tensors trained, seed42, SGD lr0.001/momentum0/weight_decay0/dropout0, same16 examples in each local step. Signed symmetric deterministic nearest-even per-tensor INT4 packet, no new collection or codec change. ICA consumes classifier.0.weight decoded q*s [4096,25088]. Source count16. Other tensors remain preserved.

Public observation: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation
Manifest SHA256: 9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69
Checkpoint: C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth
Checkpoint SHA256: 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0

FastICA performs exactly4000 unchanged fixed-point updates, disabling only the convergence break. Report the first convergence iteration and final residual. One existing whitening basis is shared with exactly4000-step GD-ICA using existing coefficients, plain SGD0.001 and existing stable prior/math helpers. Save full residual/loss logs, signed sources and target_z=abs(S_hat) in native estimated order. An isolated task-local loop avoids changing historical core defaults or protected sources.

FIA: unchanged official CPA Direct adapter, exactly25000 steps, Adam0.1, cosine scheduler, fi1, TV0.1, seed42; all16 final images. No generator, gradient-matching/cell objective, private feedback, intermediate selection, new seed or comparison arm.

## Implementation and checks

Task-local runner: .ai/execution-results/CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006/run_experiment.py
Task-local tests: test_experiment.py (4 focused synthetic tests passed: exact4000 despite immediate convergence, GD4000/abs target, nonfinite abort, shortened-run gate before loading pixels).
Preparation completed: source/input/checkpoint/LPIPS weight hashes sealed in public registration.json; at least8GiB free GPU memory; CUDA available. Reused production sources and old artifacts unchanged.

Command: C:/miniconda/envs/DLG/python.exe -B .ai/execution-results/CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006/run_experiment.py run

Launch one hidden persistent process. Refuse duplicate start. No automatic research restart. Numerical or provenance failure stops dependent stages; preserve successful outputs. Estimated35-45min on RTX5060Ti16GB; actual timings/resource peak recorded.

## Evaluation and completion

The wrapper automatically launches a separate evaluator after public provenance/count/candidate gates pass. Evaluate the final frozen16-image set only, using existing pixel-MSE Hungarian matching once for MSE/PSNR/SSIM/VGG LPIPS. Private bundle is deserialized only during evaluation; x alone is consumed and bundled keys are disclosed. No user-provided numerical success cutoff exists. Save metrics, original/recovered comparison PNG and report. Software completion is not proof of successful reconstruction or actual-update cell feasibility.

Public output: quantization_aware_cpa/results/single_client_int4_fast4000_gd4000_fia25000_20261006
Private report: quantization_aware_cpa/results_private/single_client_int4_fast4000_gd4000_fia25000_20261006/report.md
Execution record: .ai/execution-results/CPA_SINGLE_CLIENT_INT4_FAST4000_GD4000_FIA25000_20261006/execution_record.json

Register task-local notify_exit.py once against verified worker PID/creation time. It waits on the OS exit event without model calls and attempts one completion message to this explicitly assigned current conversation. Do not use stale Luna/Planner peers. After one launch/status verification, end the model turn; no model progress polling. Completed reports are written by the experiment wrapper itself.

WORKER_PID: 18052
WORKER_CREATED_MS: 1791275487701
CALLBACK_PID: 16376

## Completed result (2026-10-06)

All optimization completed once: FastICA4000 and GD-ICA4000 have saved contiguous
logs1..4000; FIA settings n_iter25000 and final zero-based log iter24999 match.
The original worker exited1 only because its offline evaluator compared the
source-data TRUTH_SHA to the derived evaluation bundle. The unchanged bundle SHA
724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2 matches both
the original sender_validation.json and archived20260917 evaluation record.
Task-local repair_evaluation.py corrected only the expected file identity in
memory and reused the immutable evaluator/public gate on frozen artifacts.
Reporting repair exited0. No optimization restarted, no new experiment or poll,
and no historical Luna/Planner binding received a message.

Attack wall time2049.116s (34min9s); separation45.662s; FIA2002.924s.
Peak CUDA allocated memory3526829568B (3.285GiB).
All16-image means: PSNR11.547459dB, SSIM0.287770, VGG LPIPS0.805637,
MSE0.077835116. Visual comparison shows no clearly recognizable reconstruction;
no numerical success threshold was registered and no causal claim is made.
Final FastICA residual1.54972076e-6 exceeds tolerance1e-6 despite first crossing
at84; early stopping remained disabled and exactly4000 updates completed.

Original worker record and failure are preserved in the task directory. Final
execution_record.json distinguishes original worker exit1 from reporting exit0.
Saved source/input/artifact hashes passed the unchanged gate before and after
evaluation. Final metrics/image/report paths are the registered private outputs.
