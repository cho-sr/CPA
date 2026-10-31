# Scheduled 7 -> 6 -> 5 -> 4 bit repetitions after the current INT8 completion

TASK_ID: CPA_SIGNED_BITS7654_SCREEN3_20260918
PLAN_VERSION: 1
STATUS: RUNNING
ACTIVATION_POLICY: AFTER_SUCCESSFUL_PARENT_COMPLETION_EXECUTOR_OWNED
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_SCHEDULE_BITS_7_6_5_4_AFTER_CURRENT_WORK_20260918
DEPENDS_ON: CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918
BIT_ORDER: 7,6,5,4
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_BITS7654_SCREEN3_20260918/report.md

## User request and ownership

The user directly requested in thread 01a0a7ab-fa58-7d22-ac52-3247b1dd2255:
"이거 다하면 7 6 5 4 비트도 실행 스케줄 걸어줘".
This explicitly authorizes scheduling and executing four repetitions of the current
experiment, changing the quantization bit-width in that order after current work
finishes. No repeat permission is required for this scope. The request authorizes
the necessary isolated multi-bit integration and its focused validation.

The CURRENT claimed task identifies the existing designated Executor as
thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID 14236, start 1789518279030 ms,
Luna/xhigh. Use that actual authorized recipient, not the obsolete Astra/high
binding in the generic executor bridge. Preserve all shared bindings. This
schedule's author is forwarding the user's explicit request, not taking over the
existing Planner or Executor role. Do not create an AI session.

Do not modify the active RUNNING handoff or any imported research source until
the parent experiment and its reporting are finished. Its wrapper is PID 9068,
start 1789663302825 ms. A task-local OS exit callback queues this instruction
once to the same Executor; exit alone is NOT proof of scientific completion.

## Activation gate

First finish the currently authorized INT8 repair, public gate, all 13-set offline
evaluation and final reporting. Verify its real execution_summary, integrity,
outputs and report; a handoff marked COMPLETED without successful scientific
completion is insufficient. If an ordinary parent output/report repair remains,
finish it within its existing authorization and budget using saved artifacts.
If the parent fails or cannot be completed, record this schedule as
BLOCKED_BY_PARENT and do not launch a lower-bit run. Do not silently continue on
an incomplete control. Do not overwrite any other newly claimed task.

Once the dependency is successful and no other experiment owns execution, archive
the completed handoff and adopt this immutable plan as the active task as its
designated Executor. Keep a durable schedule status with four ordered jobs and
the parent's completion evidence. Existing completed compatible output is reused
only after exact authentication; never overwrite or relabel historical INT4 data.

## Inherited experiment, with bit-width as the only scientific change

Read these immutable plans completely, with this schedule taking precedence only
for bit-width, ordered repetitions, output roots, activation and aggregate budget:

- .ai/execution-handoffs/CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917_v1.md
  SHA256 7a9b09314f77458e1526cd318d99bef427633567f042cac975c5dfea341b5c4b.
- .ai/execution-handoffs/CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917_v1.md
  SHA256 318a67e7e62a021ea73865cc2917e909c78810b1505509d131563f9bb0388092.

Retain B16/T10, frozen VGG16 Conv, same six-FC archived raw update, checkpoint,
data, normalization, training rules, seed42, eight noise streams
1234+1000003*m, separator, loss, alignment, ranking and evaluation definitions.
The screen3 plan supersedes the older averaged-feature plan: never average
features/images. Do not carry an INT8 or previous-bit image, target, selected
member list or optimizer state into a different bit's fresh run.

For each b in [7,6,5,4], define Qmax=2**(b-1)-1 (63,31,15,7), symmetric codes
[-Qmax,Qmax], CPU FP32 s=absmax/Qmax, nearest-even division/rounding/clamping,
and the same all-zero convention. A separate sender converts the SAME
authenticated archived raw update directly at each b; never quantize an already
quantized update. The attack sees only the corresponding public codes/scales.
Preserve exact FP32 cell endpoint/tie/cap and public extremal-anchor semantics
with the correct Qmax. Record logical bits, real payload storage and metadata
bytes separately; int8 tensor storage is not proof of b-bit packed bandwidth.
Use a new, versioned codec/reader with explicit bit metadata and roundtrip tests;
do not mutate INT8 or historical INT4 codec constants in place.

Per bit, execute one center and eight independently sampled member separations
with the unchanged FastICA + GD4000 method. Each gets its own whitening. Align
members to that bit's center using the same public permutation/sign-only rule.
Run all eight members for 5000 FIA updates on the original TOTAL25000 cosine
schedule; rank by the PUBLIC post5000 objective feature distance + 0.1*TV with
member-index tie break. Freeze that bit's top3 before private loading, then
continue those three by 20000 updates each to TOTAL25000. Center remains a
separate 25000 control retaining its genuine 5000 prefix. Preserve common random
initialization and all checkpoint/Adam/scheduler/RNG and parent hashes.

Use the repaired continuation writer/export behavior proven by the parent;
never recompute a successful optimizer stage to repair a missing output file.
Authenticate inherited versus new source revisions explicitly. Implement any
necessary bit-generic integration in isolated modules, protecting original
ICA/FIA/model/metric and historical experiment sources and outputs. Validate
codec endpoints/roundtrip, same-bit binding, boundary separation and the actual
fresh/prefix/resume/export/public-gate orchestration on focused small fixtures.
No extra real-data pilot, new seeds, method tuning or additional branches.

For each bit freeze all 13 candidates and the public gate, then use the same
separate x-only offline evaluator and pixel-MSE Hungarian assignment for all
MSE/PSNR/SSIM/VGG LPIPS metrics and both contact sheets. Primary contrast is that
bit's public rank1 at25000 versus its center25000. No oracle winner selection or
private feedback to any later bit. Pre-register all four configurations before
reading their private outcomes. Final cross-bit reporting uses the same metrics.
Actual-image cell feasibility remains NOT_EVALUATED.

## Outputs, budgets and serial execution

New per-bit public roots:
quantization_aware_cpa/results/signed_int{b}_randomized8_screen3_fia_B16_T10_20260918_sweep
Private counterparts use results_private with the same suffix. Check collisions
before execution, preserve existing bytes, and record every artifact binding.
Schedule records and final report belong under this task's execution-results root.

Exactly four fresh bit experiments, 125000 FIA updates per bit, 500000 total;
36 separation calls total. Retain the parent's per-experiment real-work ceiling
12600 seconds and CUDA allocated cap 12 GiB. Aggregate ceiling is 50400 seconds
(14 hours) of real work, excluding parent work, development/tests and idle time.
The roughly 3-hour per-bit estimate is inherited, not a measured promise.
Keep bit jobs strictly sequential: parent -> 7 -> 6 -> 5 -> 4. Do not overlap GPUs.
Stop dependent jobs on unrecoverable error/budget failure, preserve successful
checkpoints, and report exact remaining states instead of skipping a failed bit.

Before launching, prepare and validate the whole ordered persistent wrapper,
including completion gates/evaluation/reporting and per-bit budget accounting.
Use C:/miniconda/envs/DLG/python.exe for research work. Verify one startup receipt,
register a single non-model OS exit callback to this same Executor, then END the
turn. No model polling or additional agent. On exit finish authorized reporting
from saved outputs, record all four statuses, and send one final result to the
existing Planner thread 01a0a799-d62c-7ab0-9c2a-628acf6d2c95. Do not schedule any
bits, seeds, retries of completed optimization or experiments beyond these four.


## Activation evidence

PARENT_TASK_STATUS: COMPLETED_AFTER_REPORTING_REPAIR
PARENT_FINAL_REPORT: .ai/execution-results/CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918/report.md
PARENT_PUBLIC_GATE: PUBLIC_GATE_PASSED
PARENT_PRIVATE_EVALUATION: COMPLETED; 13 candidates; x only
PARENT_OPTIMIZER_REPAIR_STEPS: 0 after saved continuation
PARENT_FINALIZATION_SUMMARY: .ai/execution-results/CPA_SIGNED_INT8_SCREEN3_REPAIR_CONTINUE_20260918/finalization_summary.json
SCHEDULE_CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
SCHEDULE_OUTPUT_ROOTS_COLLISION_CHECK: PASSED
