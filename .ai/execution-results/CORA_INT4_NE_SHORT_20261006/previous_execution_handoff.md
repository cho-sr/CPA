# Same-client FP32 and INT8: FastICA4000 -> GD-ICA4000 -> Direct FIA25000

TASK_ID: CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006
PLAN_VERSION: 1
STATUS: FAILED
ASSIGNED_ROLE: CODEX_EXECUTOR
EXECUTOR_TARGET: CURRENT_CONVERSATION_EXPLICIT_USER_ASSIGNMENT
REQUIRED_MODEL: CURRENT_SESSION_UNCHANGED
CLAIMED_BY: CODEX_EXECUTOR:01a11025-fc1c-7e53-bea5-3cc32a1bae01
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_FP32_AND_INT8_REQUEST_20261006
DELIVERY_STATE: DIRECT_USER_ASSIGNMENT_NO_HISTORICAL_PEER_QUEUE

The user explicitly assigned this conversation CODEX_EXECUTOR and requested the preceding INT4 run here. Following its completed result, the user now instructs "32비트랑 8비트에 대해서 돌려줘". This authorizes two new bitwidth arms of that same experiment; all other conditions remain fixed. No reapproval, historical Luna/Planner dispatch, model change, relay response, tuning, extra seed/arm or additional scientific change is authorized.

## Fixed experimental conditions

Same archived single-client B16/T10 local update and original16 images. VGG16 pretrained checkpoint; Conv frozen, all six FC tensors trained10SGD steps; same16 samples each step; lr0.001/momentum0/weight_decay0/dropout0/evalmode; seed42. ICA consumes classifier.0.weight [4096,25088], source_count16. The only scientific change is public update representation: unquantized FP32 or signed symmetric deterministic nearest-even per-tensor INT8 [-127,127], scale=float32(absmax/127). CPU FP32 arithmetic is used after decoding INT8 q*s. No new local training or victim data collection.

Archived raw update SHA256: ef9aefc66ab9fcfd0b023fdfc064dbadc27567bd3ef3508bc14ebda9af28ba4c
Archived sender validation SHA256: 6e78ec5ff2376b01c23a3606fa18b3d9e60d1209f55504211aaf811919d2e244
Existing INT4 manifest SHA256: 9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69
VGG16 checkpoint SHA256: 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0
Evaluation bundle SHA256: 724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2

Sender preparation authenticates archived raw update and sender record, verifies all six original INT4 codes/scales are exactly reproduced, exports all six FC updates as public FP32 and existing strict INT8 packets, then seals manifests/packets/checkpoint/code hashes. Images, labels and oracle features are not deserialized during sender preparation or either attack. INT8 attack sees only its public q/s packets; FP32 attack sees its authorized unquantized public update.

Exactly4000 FastICA fixed-point updates without convergence break, followed by exactly4000 existing GD-ICA steps with unchanged priors/coefficients/plain SGD0.001 and one shared whitening basis. Reuse the task-local tested fixed4000 loop from the completed INT4 runner; original source files stay byte-exact. Save full1..4000 logs, signed S_hat and target_z=abs(S_hat), native estimated order. No randomized members or private selection.

Unchanged official CPA Direct FIA: exactly25000 steps, Adam0.1, cosine scheduler, fi1, TV0.1, seed42, VGG16 adapter. No generator, cell/replay/gradient-matching objective or intermediate candidate selection. Run FP32 and then INT8 once each, sequentially, in isolated processes. An independently failing arm is preserved; no restart is authorized. Provenance failure before dispatch stops dependent execution.

## Implementation and verification

Task-local directory: .ai/execution-results/CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006
run_experiment.py orchestrates prepare/run/attack/evaluate and reuses unchanged ICA/FIA/evaluation helpers. test_experiment.py5focused synthetic contracts passed: FP32 tiny values retained without quantization; INT8 reader consumes only public q/s; scalar absmax and nearest-even ties; shortened84-step result rejected before pixel loading; unknown arm rejected. Existing fixed4000/GD4000 tests were already passed on the immutable reused INT4 helper.

Allowed writes: this task-local directory, registered new public/private result roots, this active handoff, new task plan and byte-exact archive of completed prior handoff. Protected production and prior result files remain unchanged.

Preparation command: C:/miniconda/envs/DLG/python.exe -B .ai/execution-results/CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006/run_experiment.py prepare
Worker command: C:/miniconda/envs/DLG/python.exe -B .ai/execution-results/CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006/run_experiment.py run

Expected total70-90min based on INT4 measured34min9s; actual timings and CUDA peak are saved per arm. At least8GiB CUDA free required. Nonfinite values or input/source integrity violations abort affected dependent stages; no automatic research restart.

## Evaluation and completion

Only after both attack attempts have ended, authenticate public source/input/artifact hashes and actual exact counts. Evaluate all16 final images with the same previous frozen evaluator: pixel-MSE Hungarian matching once per arm for MSE/PSNR/SSIM/VGG LPIPS. Only x is consumed from the correctly registered evaluation bundle; disclose deserialized keys. No images/labels/features are supplied to the attacks. No numerical success cutoff is registered. Original evaluator is imported unchanged; supply task paths and the verified evaluation-bundle SHA in memory. Update inherited LPIPS marker after actual LPIPS computation. No completed optimization is rerun for reporting repair.

Produce per-arm metrics.json/comparison.png/report.md and aggregate comparison.json/report.md, including the cached previously completed INT4 metrics without rerunning INT4. Seal cached INT4 metric file hash before attacks. Software completion does not establish reconstruction success or actual-update cell feasibility.

Public root: quantization_aware_cpa/results/single_client_fp32_int8_fast4000_gd4000_fia25000_20261006
Private root: quantization_aware_cpa/results_private/single_client_fp32_int8_fast4000_gd4000_fia25000_20261006
Execution record: .ai/execution-results/CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006/execution_record.json

Launch one hidden persistent master worker, register task-local notify_exit.py once on verified PID/creation time and current thread01a11025-fc1c-7e53-bea5-3cc32a1bae01. The callback uses the OS process-exit event without model calls, then queues one completion notice to this same explicitly assigned conversation. No historical peer binding is used. Confirm launch/callback once and end the model turn; no model progress polling, monitoring agents or new AI session. Worker already performs evaluation/reporting; the one completion turn verifies results or repairs reporting only and reports to this user.

WORKER_PID: 33376
WORKER_CREATED_MS: 1791280140715
CALLBACK_PID: 27508

## Verified partial completion (2026-10-06)

Original worker33376 exited1. INT8 attack and all16-image evaluation exited0,
and saved logs authenticate exactlyFastICA4000/GDICA4000/FIA25000. INT8 attack
time2100.293s (35min0.3s); separation45.814s; FIA2053.879s.
Means: PSNR12.054908dB, SSIM0.295547, VGG LPIPS0.562428, MSE0.070204774.

FP32 failed in the existing whiten_observation(X,16) guard with
"observation has insufficient usable covariance rank for source_count" before
FastICA/GD-ICA/FIA. Actual optimization counts0/0/0; no candidate or metrics.
Configured rank_floor1e-12 and source_count16 were preserved. Eigenvalues were
not saved; mathematical rank and the cause of the guard are not established.
This failure does not prove FP32 reconstruction is impossible. No optimization
restart, whitening change, additional experiment or polling was performed.

Cached INT4 was reused unchanged: exactly4000/4000/25000; attack2049.116s;
PSNR11.547459dB, SSIM0.287770, LPIPS0.805637. This one-batch comparison improves
all three image metrics at8bit, but provides no FP32 quality result.
Visual review finds some recognizable outlines at8bit, with faint images and
several texture-dominated outputs. No quantitative success cutoff is claimed.

Reporting-only finalizer authenticated sealed inputs/code/attack/evaluation
artifacts, retained original worker record/report/comparison and failure,
added FP32 failure to the aggregate report and verified both completed counts.
Task status remainsFAILED because the requested FP32 arm did not complete;
completion=PARTIAL_COMPLETION_VERIFIED and reporting_exit_status=0 are recorded.
New worker total2142.428s (35min42.4s), including failed FP32, INT8 and evaluation.
Report: quantization_aware_cpa/results_private/single_client_fp32_int8_fast4000_gd4000_fia25000_20261006/report.md
Verification: .ai/execution-results/CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006/completion_verification.json
No historical Luna/Planner message was sent; completion is reported here only.

