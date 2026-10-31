# INT4 original CPA gradient-only separation after the current bit sweep

TASK_ID: CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918
PLAN_VERSION: 1
STATUS: COMPLETED
TASK_KIND: USER_REQUESTED_ORIGINAL_CPA_GD_ONLY_COMPARISON
ACTIVATION_POLICY: AFTER_SUCCESSFUL_PARENT_COMPLETION_EXECUTOR_OWNED
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_GD_ONLY_INT4_AFTER_CURRENT_EXPERIMENT_20260918
DEPENDS_ON: CPA_SIGNED_BITS7654_RECOVERY_20260918
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_int4_gdonly_randomized8_screen3_fia_B16_T10_20260918
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918/report.md

## Authorization, interpretation and sequencing

The user directly requested: "지금 실험 끝나면 FastICa - GD ICA 이 구조에서 GD ICA 만 돌리는 기존 방법으로, 4bit 해줘".
Originating user thread: 01a0a7ab-fa58-7d22-ac52-3247b1dd2255.
This authorizes one new 4-bit condition after the current experiment, with
necessary isolated integration, focused validation, evaluation and reporting.
Do not ask again for this scope. Preserve every other fixed scientific condition.

Planner's declared interpretation of "기존 방법" is the original CPA gradient-only
backend and its existing 25000-step configuration in signed_baseline.CPA_CONFIG,
not merely replacing FastICA initialization inside the current GD4000 algorithm.
An optional clarification presented those alternatives; no answer had arrived
when this plan was prepared. Respect any subsequent explicit user correction
before launch; do not run both alternatives. Current replacement changes the
whole separator backend (whitening, optimizer, prior and iteration count), so
this is NOT a controlled ablation of FastICA initialization alone.

Use only the existing designated Executor: thread
01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236/start1789518279030ms,
actual model gpt-5.6-luna, reasoning xhigh. No new AI session or peer rebinding.
Planner destination is 01a0a799-d62c-7ab0-9c2a-628acf6d2c95.

This is a DEFERRED SIDECAR while the parent is RUNNING. Do not overwrite
.ai/EXECUTION_HANDOFF.md, stop its children, change its registered sources, or
start the new real experiment concurrently. Record this pending job durably and
include its activation check when handling the parent's already-registered exit
callback. If a one-shot OS exit scheduling notice is necessary, use one exact
PID/start-bound, model-free notice to this same Executor with duplicate protection;
do not add model polling or an additional research process. Do not wait in a
model turn. A queue receipt is not a scientific launch receipt.

At planning time the parent wrapper is PID25160/start1789703008341ms, and its
one-shot callback PID25904 waits for that exit (attempt4). INT7/6/5 are complete;
INT4 is running. These observations are historical pins, not permission to assume
a process is still alive. Authenticate actual ownership/identity before use.

First complete the parent's INT4 public gate, offline evaluation, integrity and
final reporting; preserve the already-authorized same-scope repairs and exact
resume policy. A process exit or stale status alone is insufficient. If a parent
error cannot be repaired inside its existing authorization/budget, mark this job
BLOCKED_BY_PARENT and report rather than launch against an incomplete control.
Once successful and no other task owns execution, the Executor archives the
completed active handoff, adopts this plan, and launches this single condition.
Do not infer additional permission from successful results.

## Frozen inputs and randomized dequantization

Read the inherited screen/ranking/resume contracts in:
- .ai/execution-handoffs/CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917_v1.md
  SHA256 7a9b09314f77458e1526cd318d99bef427633567f042cac975c5dfea341b5c4b
- .ai/execution-handoffs/CPA_SIGNED_BITS7654_SCREEN3_20260918_v1.md
  SHA256 f3f8c60e36f6fd1cece6d30acf6429a15ce0dc14a0ad5bac744f743f6d751612
This plan overrides only separator backend, single-bit scope, output roots,
dependency and comparisons. Preserve the repaired fresh/prefix/resume/export
behavior from the completed parent. Do not revive the earlier averaged-source
plan or perform extra bits.

Use the current parent's authenticated PUBLIC INT4 observation directly:
quantization_aware_cpa/results/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep/observation
Its manifest.json SHA256 is
e1519866632bce490c687420ce6f7f61afd9fb05cbe1fc25ff63c8fa51788fa9.
Verify every payload hash bound by that manifest. No new sender conversion,
raw-update load, local training or new private data acquisition is needed.

Same signed symmetric nearest-even FP32 codec, q in [-7,7], per-tensor scalar
scale absmax/7, exact endpoint/tie/cap and public extremal-anchor semantics.
Same B16/T10, VGG16 checkpoint/frozen Conv, FedAvg update definition, model/data
normalization and six-FC observation metadata. Checkpoint SHA256 remains
397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0.

Exactly one center matrix s*q and eight randomized matrices, members0..7,
seeds1234+1000003*m. Use the unchanged signed_bits_randomized.sample_matrix
with bits=4 and chunk_rows=32; reproduce and compare all nine input tensor hashes
against the parent's separation records before optimizing each corresponding
branch. The public observation and sampled matrices are shared; recovered sources
and FIA states are newly computed for the new method. Do not substitute a new
noise draw, average, filter members or recycle parent FastICA/GD source outputs.

Parent registration.json legitimately changes status at public completion.
Do not pin its current RUNNING/REGISTERED bytes as the future final hash. Pin
immutable scientific configuration/input projection now and snapshot the final
registration, candidates, gate and comparison only after successful completion.
Do not "repair" a real input mismatch by updating an expected hash.

## Original gradient-only ICA, precisely

Call the unchanged qacpa.upstream_naive.run_upstream_cpa for each matrix:
- n_components=16, device="cuda", seed=42, separate output/cache per branch.
- config is an exact copy of qacpa.signed_baseline.CPA_CONFIG:
  cpa_ica_max_iter=25000, cpa_ica_lr=0.001, cpa_ica_optimizer="adam",
  cpa_ica_scheduler="none", cpa_ica_decor=5.3, cpa_ica_T=7.7,
  cpa_ica_ne=1.0, cpa_ica_nv=0.13, cpa_ica_l1=5.0, cpa_ica_tv=0.1,
  cpa_ica_log_every=5000.
- Original official_cpa/src/gradient_inversion.py:CocktailPartyAttack,
  identity W initialization and original row normalization; zero FastICA calls.
- Each input has its own original cov/(d-1), torch.linalg.eig/top-absolute-eigenvalue
  whitening. No whitening or fitted unmixing state imported from the parent.
- Use the existing signed-baseline canonical runtime seed42/threads1 and the
  upstream wrapper's original device/runtime behavior. Record actual settings.

Preserve original code semantics, including two scientifically relevant details:
the original implementation calculates then overwrites its negentropy loss with
zero, and feature TV is inactive when nv=0.13 is nonzero. Do not silently fix these
legacy behaviors or claim they were active regularizers. Log configured and
effective objective terms separately. No generator, learned image prior,
mixing-matrix fitting, cell penalty, label search or actual-update replay.

Record per-branch native signed S_hat, settings, actual iterations, losses,
checkpoint and provenance; completed budget does not establish convergence.
Do not fabricate FastICA convergence metadata to satisfy an old schema. Native
sources must be finite FP32[16,25088]. For each member use unchanged public
absolute-cosine Hungarian permutation/sign alignment to THIS NEW GD-only center,
without row-norm rescaling. Store native/aligned sources, permutation/signs and
hashes. FIA target is exactly abs(aligned_S_hat), or abs(S_center) for center.

## FIA screen and continuation

Same original Direct FIA/VGG16FeatureAdapter and exact current pipeline settings:
CUDA FP32, common actual image initialization from seed42, Adam lr0.1,
cosine horizon25000 from the start, feature weight1 and normalized TV weight0.1.
Reset construction/runtime as in the proven upstream driver, and bind the actual
initial image parameter hash. No parent source/target/image/optimizer carryover.

Run the new center to25000 while retaining its genuine5000 prefix. Run all eight
new member targets to5000 on the25000-step scheduler. Compute the unchanged PUBLIC
post5000 score feature cosine distance +0.1*normalized-image TV, tie by ascending
original member ID. Freeze top3 before any private load. Continue only these3
from5000 for20000 further steps, to total25000. Selection is over whole16-image
batches, excludes center, and is newly determined rather than copied from parent.
Exactly125000 new FIA updates and225000 original-CPA updates across9 separations.

Use authenticated raw image/Adam/scheduler/RNG checkpoints, immutable parent5000
prefixes, correct total-horizon state and child exports. A missing exported
candidate after saved25000 is repaired from saved state with zero new updates;
do not require a newly generated5000 prefix from a continuation child. Stop/score/
export must not mutate the raw optimization state. Authenticate all13 frozen
candidate sets: center5000/25000, eight member5000 and three member25000.

## Isolated integration, validation and source protection

Executor implements a new method-specific adapter/runner/gate and necessary
offline reporting adapter under new files only, e.g.
quantization_aware_cpa/qacpa/signed_int4_gdonly_screen3*.py and focused tests,
plus task-local scripts under this task's execution-results root. Reuse protected
implementations through their public APIs rather than altering constants or
monkeypatching FastICA metadata. Existing mutable metadata transitions must be
handled honestly in the new gate/schema.

Protect all existing official_cpa sources, upstream_naive.py, signed_baseline.py,
signed_combination_ica.py, deterministic_hybrid_setup.py, signed_bits_randomized.py,
the existing codec/runner/FIA/evaluation modules, old raw/public/private inputs,
checkpoint/LPIPS weights and all parent source/result/control/figure artifacts.
Snapshot source and immutable input hashes before and after. Existing files remain
byte-identical; use new uniquely owned paths and fail on incompatible collisions.
Only after parent finalization may the designated Executor update active task
ownership. Shared Planner/Reviewer/Executor peer configuration remains unchanged.

Before real work, run bounded meaningful synthetic integration checks covering:
1. Existing original CPA backend/config is called with no FastICA; native signed
   shape and exact abs target binding, input hash identity, separate caches.
2. Public permutation/sign-only alignment preserves native row magnitudes.
3. Actual CLI dispatch, fresh5000/total25000 scheduler, checkpoint resume/export
   (including already-completed child) and ranking use the right branch IDs.
4. New public gate authenticates nine sources/all13 candidates and rejects
   tampering or incompatible methods; final registration/status transitions pass.
5. Evaluator/renderer uses truth plus an actual same-method center column, includes
   rank1/2/3 and exact original member IDs; no skipped baseline/mislabeled truth.
Use small synthetic fixtures/mocks, existing relevant contract tests, compilation
and CLI help. No real-data pilot, extra arm/seed or tuning before the registered run.
Preserve and disclose the known upstream complex-to-real warning.

## Offline scientific comparison and plots

Freeze the configuration, all new public results/top3 and integrity gate before
the separate evaluator loads private x. Use canonical seed42/threads1, unchanged
MSE/PSNR/SSIM/VGG-LPIPS definitions and one pixel-MSE Hungarian assignment per
candidate reused across all metrics/figures. labels, true features and true
updates remain unconsumed; record deserialization versus actual consumption.

Evaluate all13 new candidates. Also authenticate and evaluate the parent's four
frozen final25000 candidates (center plus its three PUBLIC selected members) in
the same offline comparison, without rerunning their optimization. Require
archived control mean-metric reproduction within the unchanged2e-8 absolute
tolerance, and report assignment/per-image residuals. Use the same canonical
runtime that resolved earlier LPIPS reproduction differences. No tolerance
relaxation or selection changes in response to private results.

Predeclared primary contrast: new public rank1 final versus new center final.
Secondary: new center versus parent center; new public rank1 versus parent public
rank1. Also show all three frozen ranks for each method descriptively. These are
method comparisons on one fixed batch/seed, not a FastICA-only causal ablation,
statistical generalization or verified identity recovery. Image cell feasibility
is NOT_EVALUATED (no image-update replay); do not report it as false by assumption.

Provide baseline-inclusive final and5000 contact sheets from saved pixels and
stored assignments. The final main figure is Ground truth | GD-only center naive
dequantization baseline | new public rank1 | rank2 | rank3. Add a clearly labeled
side-by-side method-comparison figure showing parent center/rank1 versus new
center/rank1. Reuse the repaired renderer/slide helpers through isolated adapters.
Also export five3840x2160 slide PNGs (one all16 overview and four detail pages
covering01-04/05-08/09-12/13-16), index/manifest and ZIP under the new private root.
Keep exact candidate pixels/member IDs and the footer that matching by pixel
error does not verify image identity. Plotting never recomputes matching/metrics.

## Budget, persistence and completion

Exactly one new 4-bit condition, nine separations,125000 FIA updates and one
separate offline comparison. Real-work ceiling12600seconds (3.5hours), CUDA
allocated-memory cap12GiB; charge separation, FIA, evaluation and final reporting
and every same-scope recovery attempt cumulatively. Parent work, idle waiting and
synthetic implementation tests are excluded. Existing historical naive CPA took
43.17seconds for one CUDA separation; this is context, not a new timing promise.
Record actual per-stage/cumulative timing and memory. Do not silently increase
budget or change iterations to fit it. Stop cleanly and preserve last usable
states on an unrecoverable condition or budget exhaustion.

Use C:/miniconda/envs/DLG/python.exe for research. Prepare a persistent serial
wrapper with dependency/ownership checks, code/input registration, full stages,
budget accounting, final reporting and exact-resume evidence. Launch hidden after
the parent completion gate, verify one startup receipt and one non-model OS exit
callback to this SAME Executor, then end the turn. No model polling or new agents.
On exit finish ordinary authorized export/evaluation/report repairs from saved
states. Do not rerun successful optimization, add seeds/arms, or tune objectives.

Write report.md, execution_summary.json, final_audit.json and separate public
selection/private comparison/PNG manifests with exact paths. Report configured
and effective original objectives, no-FastICA evidence, input equality, nine
iteration counts,125000 FIA accounting, new top3, all control residuals and method
contrasts, source integrity and any limitation/failure. Deliver the final result
once to existing Planner and end; no follow-on experiment is authorized.


ACTIVE_CLAIM_STATUS: COMPLETED
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
CLAIMED_AT_UTC: 2026-09-18T10:39:50.577242+00:00
PARENT_FINAL_AUDIT_STATUS: COMPLETED_ARTIFACTS_VERIFIED
IMPLEMENTATION_TESTS: py_compile, CLI help, pytest 5 passed
ACTIVATION: USER_CONFIRMED_ORIGINAL_CPA_GD_ONLY_25000

FINAL_STATUS: COMPLETED
FINAL_RESULT_PATH: .ai/execution-results/CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918/final_result.json
PUBLIC_GATE_STATUS: PUBLIC_GATE_PASSED
EVALUATION_STATUS: COMPLETED
OPTIMIZER_STEPS_ADDED: 0
REPORTING_REPAIR: evaluation_repair_20260918; saved public gate reauthenticated with strict provenance
COMPLETED_AT_UTC: 2026-09-18T13:44:25.306083+00:00
