# INT4 row-wise signed quantization, FastICA then GD4000, randomized8 screen3 FIA

TASK_ID: CPA_SIGNED_INT4_ROWWISE_COMBINATION_SCREEN3_FIA_20260919
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: USER_AUTHORIZED_ROWWISE_QUANTIZATION_COMPARISON
ACTIVATION_POLICY: IMMEDIATE_AFTER_COMPLETED_OWNER_ARCHIVE_NO_DUPLICATE_RUN
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_INT4_ROWWISE_FASTICA_GDICA_EXPERIMENT_20260919
COMPARISON_CONTROL_TASK: CPA_SIGNED_BITS7654_RECOVERY_20260918
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_int4_rowwise_combination_randomized8_screen3_fia_B16_T10_20260919
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_int4_rowwise_combination_randomized8_screen3_fia_B16_T10_20260919
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_INT4_ROWWISE_COMBINATION_SCREEN3_FIA_20260919/report.md

## User authorization and ownership

The user requested: "fast ica  + gd ica 방식으로 row wise 한 방법으로실험 돌려볼래 4비트에서".
Originating user thread: 01a0a7ab-fa58-7d22-ac52-3247b1dd2255.
The follow-up "랜덤은 몇번할거야" asked the sample count. Planner answered eight
randomized dequantizations plus one center, FastICA -> GD4000 for all nine,
FIA5000 for all eight and public top3 to total25000. It did not change that count.

This explicitly authorizes ONE new four-bit row-wise quantization condition,
necessary isolated implementation, focused validation, real execution, saved
comparison and slide PNG reporting. No additional permission is needed. Interpret
row-wise as one quantization scale for each output-neuron row of each FC weight
matrix; bias vectors remain one group each. This definition was presented before
handoff. Preserve the deterministic signed codec family and all other conditions.

Use the existing designated Luna/xhigh CODEX_EXECUTOR only:
thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236/start1789518279030ms.
The actual settings must still be verified. Planner thread is
01a0a799-d62c-7ab0-9c2a-628acf6d2c95. Do not create a model session or rebind peers.
At preparation, the active GD-only task CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918
is COMPLETED and no research Python process is running. Recheck ownership once,
archive the completed active handoff without changing its evidence, adopt and
claim THIS plan, and proceed immediately through implementation and execution.
This is not a deferred reservation. Do not stop after writing a READY schedule.

Read the inherited screen/FIA contract:
.ai/execution-handoffs/CPA_SIGNED_INT8_RANDOMIZED8_SCREEN3_FIA_20260917_v1.md
SHA2567a9b09314f77458e1526cd318d99bef427633567f042cac975c5dfea341b5c4b,
and the tensor-scale FastICA/GD4000 plan:
.ai/execution-handoffs/CPA_SIGNED_BITS7654_SCREEN3_20260918_v1.md
SHA256f3f8c60e36f6fd1cece6d30acf6429a15ce0dc14a0ad5bac744f743f6d751612.
This document overrides only scale grouping, the associated versioned codec/
sampler, single-bit scope, output roots, control comparison and immediate launch.
It does not inherit original-CPA GD25000 settings from the most recent task.

## Same victim update, new row-wise public observation

Retain VGG16/ImageNet normalization, frozen Conv and all-FC local training,
B16/T10 single-client FedAvg, original data/order/training rules and signed raw
delta theta_after10_minus_theta_start. Do not retrain the victim or derive this
packet from an already quantized/dequantized observation.

A separate sender process consumes the SAME archived raw update:
quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/raw_updates.pt
SHA256ef9aefc66ab9fcfd0b023fdfc064dbadc27567bd3ef3508bc14ebda9af28ba4c.
Its sender_validation.json SHA256 is
6e78ec5ff2376b01c23a3606fa18b3d9e60d1209f55504211aaf811919d2e244.
Checkpoint SHA256 is
397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0.
Use existing local files only, no downloads. Only sender may deserialize true
updates; it must not load private x, labels or features. Its validation record is
private and cannot become an attack input. Hashing a private file is distinct
from deserializing/consuming its fields and must be reported accurately.

New versioned signed deterministic symmetric row-wise INT4 codec:
- Code range [-7,7], nearest-even rounding, CPU FP32 operations in this order:
  m_r=max_j(abs(G[r,j])); s_r=float32(m_r/7);
  q[r,j]=round(float32(G[r,j]/s_r)).clamp(-7,7).int8.
- For a zero row use q=0 and s=1. Reject nonfinite inputs or nonzero scale
  underflow; no arbitrary clipping, new rounding mode or silent precision change.
- Matrices keep their ORIGINAL axis0 rows: classifier.0.weight[4096,25088],
  classifier.3.weight[4096,4096], classifier.6.weight[1000,4096]. Scales are
  unambiguous row vectors broadcast as s[:,None]; never column scaling.
- Each bias vector is ONE row/group, shape[1,C] for encoding then restore original
  [C]. Do not allocate a separate magnitude/scale to every scalar bias element.
- All six FC tensors are encoded in the new public bundle, with exact shape,
  scale count/shape, grouping, zero convention, dtype/rounding and file hashes.
  Roundtrip the actual written packet and authenticate the decoded q/scales.

The changed observation exposes 4096+4096+1000+3=9195 FP32 scales, or36780 scale
bytes, versus24 for the tensor-scale codec. Report this metadata difference.
For minimum transport change, keep physical codes as int8 storage, declare logical
four-bit codes separately, and report exact code/scale/header/manifest/total bytes.
Do not describe int8 files as packed4 bandwidth. Do not reuse historical affine,
stochastic16-level or additive-dither-only experiments as this signed codec.

The public attack accepts only this new q/scales bundle and public model/rules.
The separation matrix is the first FC weight delta, shape[4096,25088], in its
original raw signed units; center is exactly q.float()*s[:,None]. No sign/lr/T/B
conversion, transpose, row normalization, integer-q-only ICA, or false claim of
identical inputs to the older tensor-scale experiment.

## Eight randomized dequantizations with exact row-scale semantics

Use exactly eight member seeds1234+1000003*m for m=0..7, a CPU torch.Generator,
FP64 uniform draws/interpolation followed by FP32 cast, row-major chunk_rows32,
same as the existing registered sampler. Generate independent matrices, not an
average. Same seed indices identify corresponding random streams; the sampled
values/cells necessarily change because the observation has changed.

For each nonzero row, compute the exact representable FP32 lower/upper code-cell
bounds, nearest-even tie ownership and finite absmax cap implied by its s_r,
using the existing signed_bits_randomized.cell_bounds_table semantics for bits4.
Vectorization/caching by identical FP32 scales is permitted only with demonstrated
bitwise equality to the scalar reference. Avoid thousands of Python scalar-bound
calls when an exact batched implementation suffices; do not approximate endpoints.

Interpolate within each coordinate's bounded cell. For EACH NONZERO ROW fix one
PUBLIC anchor: first column with abs(q)==7, signed according to that code, set to
the largest finite FP32 cap whose float32(cap/7) equals s_r. This is the row-wise
extension of the existing one-anchor-per-scale-group rule; it is not the true
argmax position. One global anchor is insufficient for row-wise scale consistency.
Rows with all q==0 are the canonical zero rows and must remain exactly zero/s=1;
reject inconsistent no-extremal-code nonzero rows. Do not add random values to
canonical zero rows. Keep the RNG draw order fixed even for anchored/zero entries.

Authenticate each sampled matrix against the SAME public packet: all fixed-scale
codes equal, each row's recomputed absmax/7 scale equals its observed FP32 scale,
and every value lies in its registered bounded cell. Record nonzero/zero row
counts, anchor mapping/hash, seed/generator/chunk, sampled matrix hash and all
code/scale residuals. No raw updates are read for these checks. Center remains
plain naive s*q; report its fixed-code and derived-scale checks honestly without
altering center values to imitate the sampler's anchors.

Do not call the old scalar sampler with a vector or silently reuse tensor-scale
metadata/gates. New method-specific validation must verify every scale group.
Report public quantization diagnostics (code histogram, zero-code proportion,
row-scale distribution) for new and archived packets. Do not use these diagnostics
to retune the registered method or introduce another run.

## Exact FastICA -> gradient ICA configuration

Use the UNCHANGED qacpa.signed_combination_ica.separate_observation core for
center plus eight members, CPU FP32/threads1/seed42, n_components16, each matrix
with its own whitening object shared between the two stages. Configuration:
- FastICA max200, tolerance1e-6, original identity start and symmetric update.
- GD refinement4000, plain SGD lr0.001, momentum0, weight_decay0.
- negentropy0.0935827857091549, decorrelation0.2513674104161625,
  sign0.05811961446881231, L1=17.94138146247377, eps1e-10, rank_floor1e-12.
Do not call the original CPA Adam25000 backend for this run. Do not adjust losses,
whitening, row-wise FastICA derivative broadcast, seed, early stopping or restart.
Report actual FastICA iterations/residual/converged flag for all9. Finite
max-iteration nonconvergence may proceed to GD as already specified by the core,
flagged honestly; nonfinite failure must not silently fall back to another method.
4000 completed GD steps are not by themselves proof of objective convergence.

Preserve native signed finite FP32 S_hat[16,25088]. Each member aligns to THIS
row-wise center using the unchanged public absolute-cosine Hungarian permutation
and sign rule (align_preserve_norm). Preserve row magnitudes, never norm-rescale,
average or use private feature/image/label alignment. Save native and aligned
sources and mapping/hashes. FIA target_z is EXACTLY abs(aligned_S_hat); center
target is abs(S_center). No source replacement or oracle component selection.

## FIA screen3, checkpoints and public selection

Preserve original Direct FIA and VGG16FeatureAdapter: CUDA FP32, seed42 common
actual initialized image parameter, Adam lr0.1, cosine T_max25000 from the start,
feature weight1 and normalized-image TV weight0.1. Reset construction/runtime per
branch as in the proven staged driver, record actual initialization hashes.

Center25000 retains its genuine5000 prefix. All8 members run5000. Rank member
batches by PUBLIC post5000 feature cosine distance+0.1*normalized TV, ties by
ascending member ID, excluding center. Freeze the new top3 before private loading.
Continue only those3 from5000 by20000 to total25000, not25000 additional.
Selection is over16-image batches, never individual image slots. Do not copy
member IDs from the previous experiment. Exactly125000 FIA updates in this run.

Use the proven staged driver/run_trajectory and completed-checkpoint export.
Save immutable raw image/Adam/scheduler/RNG/target/config/source checkpoint state,
not only clamped pixels. Child continuation does not require a new5000 prefix.
If successful optimization saved25000 but export fails, export from that saved
state with ZERO new steps. Authenticate13 new frozen sets: center5000/25000,
all8 member5000, and selected3 member25000. Never overwrite parent outputs or
restart successful trajectories to correct reporting defects.

## Isolated implementation and meaningful focused checks

Create new files only for the row-wise codec/sender, sampler, runner/public gate,
offline evaluator/plots and focused tests, for example
quantization_aware_cpa/qacpa/signed_int4_rowwise_*.py. Task-local orchestration
belongs under .ai/execution-results/CPA_SIGNED_INT4_ROWWISE_COMBINATION_SCREEN3_FIA_20260919.
Do not mutate existing signed scalar codecs/samplers, ICA/FIA core, upstream
wrappers, model, metrics, previous drivers or prior raw/public/private artifacts.
Hash all protected/imported source dependencies and old inputs before/after.
Registration must precede real sender/optimizer work. Separate immutable scientific
configuration from allowed lifecycle status so a legitimate final status change
does not cause the earlier registration-bytes failure. Never refresh expected
hashes to conceal a real code/input change after registration.

Before real work, verify small synthetic integration fixtures:
1. Matrix row-axis scaling with strongly different row magnitudes; one-row bias,
   zero/mixed rows, signs, nearest-even ties, finite-cap endpoints, underflow,
   actual wire roundtrip, malformed shape/range/scale and byte accounting.
2. Exact row-bound tables against existing scalar reference, anchor in every
   nonzero row, canonical zero rows, fixed q and re-encoded row scales equal;
   wrong-axis broadcasting/global-only anchor must be detected. Repeated seed is
   deterministic and different members are distinct where cells have variation.
3. Real parser->sender dispatch strips command selectors correctly; no private
   truth is loaded by sender/attack. Source core receives correct decoded units,
   FastICA+GD4000 config and exact signed/abs shape binding.
4. Fresh5000/total25000 scheduler, public top3, raw-state continuation and completed
   child export/gate across honest status transitions; hashes reject tampering.
5. Actual PNG renderer receives HWC uint8 exactly once, truth and center occupy
   distinct correctly named columns, original member IDs remain, all16 rows and
  3840x2160 slides are covered. Save metrics before plotting; a plot failure must
   not erase them or force recomputing valid metrics/optimization.
Use relevant existing upstream contract tests, compilation and CLI help. No real
pilot, new seed, extra arm, method tuning or scientific validation rerun is added.

## Frozen comparison, interpretation and deliverables

Primary scientific comparator is the SAME FastICA+GD4000 method with tensor-wide
scales, not the latest original-CPA GD-only run. Authenticate its frozen paths:
public: quantization_aware_cpa/results/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep
private comparison: quantization_aware_cpa/results_private/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json
The old observation manifest SHA256 is
e1519866632bce490c687420ce6f7f61afd9fb05cbe1fc25ff63c8fa51788fa9.
Its final public selection is[6,5,0]. Snapshot its final registration, sources,
gate, comparison and four final candidates before this run; no old optimization.

After freezing the new13 candidates/selection and PUBLIC_GATE_PASSED, run one
separate canonical-runtime offline evaluation of17 sets:13 new plus the four
archived tensor-scale finals (center and public ranks1/2/3). Private truth:
quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt
SHA256724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2.
Consume x only. Distinguish deserialization from field use; labels, true sources
and true updates are not consumed in evaluation/attack. Sender update use is
separately disclosed. No private feedback into attack, selection or tuning.

Use configure_runtime(42,threads=1), unchanged pixel MSE/PSNR/SSIM/VGG LPIPS and
one pixel-MSE Hungarian correspondence per candidate reused for all metrics and
figures. Require all archived mean metrics to reproduce within unchanged2e-8
absolute tolerance; record correspondences/per-image residuals. Do not loosen
tolerance, change matching, or launch another real run based on private outcomes.

Predeclared primary contrast: row-wise center minus tensor-scale center at25000,
which tests scale grouping with separator/FIA fixed. Secondary: row-wise public
rank1 minus tensor-scale public rank1, and row-wise public rank1 minus row-wise
center. All ranks remain descriptive, not oracle-selected winners. Explicitly
report the additional row-scale metadata/observation information and the group-wise
anchor policy; do not claim identical total communication cost. One batch/seed
cannot establish generalization. Actual-image cell feasibility is NOT_EVALUATED:
sampler q/scale consistency is not evidence that generated images reproduce the
actual local update. No cell optimization, FC replay, label search or generator.

Output comparison.json with mean/per-image metrics, stored assignments, all
contrasts/control residuals. Render saved matched pixels into5000 and25000 contact
sheets. Main final columns: Ground truth | row-wise naive center | new public
rank1 | rank2 | rank3, with actual member IDs. Add a method comparison showing
tensor-scale center/rank1 versus row-wise center/rank1. Reuse repaired plotting
helpers via isolated adapters with accurate row-wise/FastICA+GD captions, never
GD-only captions. Keep the matching-does-not-verify-identity footer.

Export five3840x2160 slide PNGs: int4_rowwise_25000_overview.png plus page01..04
covering01-04,05-08,09-12,13-16; index/manifest and slide_png_int4_rowwise_20260919.zip
under the new private evaluation root. Validate dimensions/columns/row coverage,
inspect overview and one detail. Never silently swap candidate or truth columns.

## Budget and persistent execution through completion

One fresh four-bit condition: one sender conversion,9 separations (FastICA up to
200 each, GD4000 each),125000 FIA updates, one gated17-set offline evaluation and
saved-artifact plots. Real-work cap12600seconds and CUDA allocated cap12GiB.
Charge sender, sampling/separation, FIA, evaluation/reporting and any recovery
cumulatively; implementation/synthetic tests excluded. The comparable previous
tensor-scale public run took10645.81seconds, so about3hours is an estimate rather
than a guarantee. Efficient exact row-bound computation matters to this budget.
Never extend iterations/budget, drop members, or change method to fit a ceiling.

Use C:/miniconda/envs/DLG/python.exe. Prepare the complete serial persistent wrapper
and meaningful preflight before launch. Ordinary same-scope implementation repairs
and precise resume from authenticated state are already authorized; retain failure
evidence and do not repeat completed optimizer stages. Stop on unrecoverable
data/integrity/method/budget failure and report actual partial state.

Launch hidden with verified PID/start, register exactly one model-free OS exit
callback to this existing Executor, verify one startup receipt and END the model
turn. Do not poll with model calls, create agents or leave the task merely queued
after preparation if launch gates pass. On completion finish authorized report/
export repairs from saved state, deliver one final result to the existing Planner
and end. No additional experiment follows automatically.

Write report.md, execution_summary.json, final_result.json/final_audit.json,
validation and source/input integrity evidence, code/scale/anchor diagnostics,
all9 ICA records, eight public scores/top3, actual FIA checkpoint iteration counts,
communication bytes, phases/data consumption, timing/memory and PNG/ZIP paths.
Preserve prior results and every failed attempt honestly. Success means the
registered work/artifacts completed, not that private-image recovery succeeded.
