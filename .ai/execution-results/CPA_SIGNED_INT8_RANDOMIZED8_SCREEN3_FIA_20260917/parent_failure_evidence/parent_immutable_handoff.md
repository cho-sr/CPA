# INT8 randomized dequantization: eight-member separation ensemble and Direct FIA

TASK_ID: CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: ONE_INT8_RANDOMIZED_DEQUANTIZATION_COMPARISON
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_INT8_RANDOMIZED_DEQUANTIZATION_20260917
REVIEW_REFERENCE: USER_REQUEST_AND_PLANNER_BOUNDED_EIGHT_MEMBER_DESIGN
PARENT_TASK_ID: CPA_SIGNED_ICA4000_FIA_20260917
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_int8_randomized8_fia_B16_T10_20260917
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_int8_randomized8_fia_B16_T10_20260917
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_INT8_RANDOMIZED8_FIA_20260917/report.md

## Authorization and concrete scope

The user said: "일단 int 8에서 radomized dequantization 방식 써보자 랜덤값을 몇개 뽑지". This authorizes trying randomized dequantization at INT8, and asks Planner to choose the sample count. Planner selected M=8 as a first bounded comparison, NOT as an established optimum. This is an explicit experiment-specific change from INT4 to INT8. Preserve all prior INT4 experiments and the overarching INT4 research objective; do not rewrite global FIXED_CONSTRAINTS or historical artifacts.

Implement the necessary isolated INT8 sender codec, sampler, blind source alignment, orchestration and focused synthetic validation, then perform exactly one registered comparison. Generate an INT8 observation from the SAME archived raw six-FC update using a separate sender process. Run one INT8 center-dequantization baseline plus eight randomized members through the unchanged FastICA -> GD4000 separator. Blindly align and average the eight signed sources, then run Direct FIA25000 exactly twice: once for the center baseline and once for the eight-member aggregate. Evaluate both frozen 16-image sets offline. No per-member FIA, extra sample counts, extra data/attack seeds, tuning, cell refinement, actual-image update replay, label inference, generator, new model/download, repeated client training or automatic continuation.

All sampling/alignment policies below are Planner choices within the new method, not user-specified constants or previously validated superiority claims. Ordinary isolated implementation fixes and focused validation needed to complete this scope are authorized. Preserve successful real stage artifacts; do not repeat completed optimizations after reporting/orchestration errors.

Existing designated Executor is Luna/xhigh thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, start1789518279030ms. Existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID24972, start1789518271364ms. Verify exact role/runtime/process identity and immutable READY handoff before claiming. No new AI session, replacement executor or peer rebinding.

## Fixed experiment context and source pins

Keep pretrained VGG16, frozen convolution, all six FC parameters trained, B16/T10, original batch order and images, original checkpoint, SGD lr.001/momentum0/decay0/foreach=False, eval mode/dropout0, seed42. Delta convention theta_after10_minus_theta_start. This is the same single-client local update, not a multi-client aggregate or a one-step gradient.

Starting checkpoint C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth SHA256397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0.

Archived sender inputs, read only by the separate sender phase:
- quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/raw_updates.pt SHA256ef9aefc66ab9fcfd0b023fdfc064dbadc27567bd3ef3508bc14ebda9af28ba4c.
- Same directory sender_validation.json SHA2566e78ec5ff2376b01c23a3606fa18b3d9e60d1209f55504211aaf811919d2e244. This records completed original training and exact ordinary/replay deltas.
- Same directory evaluation_truth.pt SHA256724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2, accessible only to final offline evaluation; sender conversion needs no images, labels or true features.
- Original INT4 public manifest quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation/manifest.json SHA2569b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69 stays unchanged and is not an INT8 artifact.

Preserved implementations:
- qacpa/signed_combination_ica.py SHA2561c7482e1ed77ca4641cd8822ebb6046c5150e794c3838aacc3e3d575f3d992d6.
- qacpa/upstream_naive.py SHA2568f29b885d86f4121eda5c55e17486a3a48bd8db08b2f83890fe7726730d1ee92.
- quantized_update_cpa/cell_utils.py SHA2566725bce108a8e795680342231696c70f1d2bc8b01a727022c605e40aa841d361.
- quantized_update_cpa/blind_alignment.py SHA2561ccf79890a579a0f242b8ca550ea6c2bb6a7f13a78d7f70be41d94d5f388ccf8.

Hash every protected/shared/official source, old artifacts and reused input required by the completed parent before and after. Register hashes of all new execution sources too. Do not loosen old task validators, monkeypatch protected codec constants, rewrite INT4 observations or call the old randomized script as though it supports INT8. That script's randomized_dequantize_tensor is hardcoded to four bits. It is design evidence only. Its default ensemble_size=8 is not scientific proof that eight is optimal.

## Phase 1: isolated sender conversion, no new local training

Authenticate the archived sender record and raw update file before deserializing. Load only the raw-update bundle, validate exact six keys/shapes, CPU FP32 finite values, checkpoint/delta metadata. Independently create a NEW signed symmetric 255-level INT8 packet: q in [-127,127], scalar per-parameter-tensor s=FP32(max_abs/127), all-zero convention q=0,s=1, nearest-even q=round(FP32(delta/s)).clamp(-127,127).int8. Reject unsupported nonzero scale underflow/nonfinite values explicitly. Code -128 is unused and rejected. Bit-width is eight; do not confuse existing INT4 tensors stored with torch.int8 with true INT8 quantization.

Write all six tensors under PUBLIC_OUTPUT_ROOT/observation with strict format/bit-width/rounding/shape/dtype/scale/hash/byte-count metadata and unambiguous .sgn8 or equivalent separate wire format. Each value uses one byte, with scales/header/manifest accounted separately. No raw FP32 delta, images, labels, true features, raw-error statistics or other private numerical data are exposed to the attack. Sender validation and private provenance can remain in PRIVATE_OUTPUT_ROOT/sender. Publish only public observation and non-sensitive pass/hash evidence needed by the attack.

Verify wire roundtrip and exact source codes/scales against the registered encoder, and record all six validation results. End the sender process before starting the public attack; do not carry its in-memory tensors into attack helpers. No client SGD/replay is necessary or authorized. Sender truth consumption must be reported honestly as true_updates_consumed=true, labels_consumed=false, images_consumed=false. Attack and final evaluator consumption have separate records: never claim globally that no true update was consumed.

## Phase 2: center baseline and eight registered randomized matrices

Attack input is ONLY the public INT8 observation and public starting model/settings. Separation uses classifier.0.weight, shape[4096,25088], source count16. Baseline matrix X0=q.float()*s exactly. Assert fixed-s code consistency for X0, but do not claim a newly inferred absmax scale without checking it.

Create exactly eight member matrices, m=0..7. Use independent CPU torch.Generator noise streams with seed 1234+1000003*m; separator/FIA seed remains42. These eight noise streams are components of one registered ensemble, not optional candidate-selection seeds. Process and discard one large matrix at a time; persist its seed, full sampled-matrix hash, sampler metadata and code/scale checks plus its small recovered features/logs. Avoid keeping eight large matrices in RAM or writing them all unnecessarily.

Sampler definition: heuristic uniform sampling within the public, bounded INT8 code cells, conditional on one public extremal anchor. This is not a claim about the true posterior or independent deterministic quantization errors.

1. Construct exact inclusive FP32 cell bounds L(q,s),U(q,s) for the actual fixed-s nearest-even divide/round/clamp encoder. Adapt the verified bit-search logic into a NEW INT8 module using 127 instead of7; do not change the protected INT4 module. At extreme codes use cap(s), the largest finite FP32 value m with FP32(m/127)==s, and signed symmetry. Check endpoint encodings and adjacent-representable exclusions. This also handles tie ownership and extreme-bin widths.
2. Select an anchor using public data only: the first flattened coordinate with abs(q)==127. Set its sample to sign(q)*cap(s), which lies in its bounded cell and pins the original per-tensor scale. Sample every other coordinate approximately uniformly over its FP32-bounded interval with the registered generator. Perform interpolation in FP64 and cast to FP32 in bounded chunks if needed to avoid avoidable overflow/precision loss; clamp only representational spill to the exact inclusive endpoints and log any such corrections. Freeze the chosen formula/chunk size before the real run. No private extrema/anchor coordinates are consulted.
3. Validate ALL sampled coordinates re-encode to the same q under observed s, obey bounds, and give the same FP32 absmax/127 scale. Persist counts, sampled-matrix hash, anchor index/value, and s. For a genuinely all-zero tensor, use zero with s=1; all-zero codes with incompatible metadata or missing extremal code for a nonzero observation fail. All eight matrices must be finite and distinct for a nontrivial input; do not resample failed members or change seeds.

The single-anchor rule deliberately induces a dependency and fixes one coordinate; disclose it. It avoids falsely claiming the full (q,s) packet matches merely because fixed-s codes match. Most interior coordinates reduce approximately to q*s+Uniform(-s/2,s/2). Do not sample beyond the extreme cells or use clipping that places substantial mass at a boundary without reporting it.

For each of X0 and the eight matrices, call unchanged signed_combination_ica.separate_observation with CombinationICAConfig(refinement_iterations=4000), CPU FP32, configure_runtime(42,threads=1), identical remaining defaults. Each matrix gets its own whitening computed ONCE and shared between its FastICA and GD stage. Do not reuse X0 whitening for perturbed matrices; do not average matrices before separation. Fixed-point FastICA identity initialization/max200/tol1e-6 and GD SGD lr.001/momentum0/decay0/no scheduler and existing four prior weights/formulas stay unchanged. Persist per-member FastICA convergence/residual/iterations, every GD log row, post-step final losses and S_hat[16,25088]/exact abs target. A finite max200 nonconvergence is reported, not silently retried/dropped; nonfinite/rank failures stop dependent stages. No learned mixing model, quantization loss or optimizer/iteration tuning is added to the separator.

There are exactly NINE separation calls: one center baseline plus eight sampled members. The old INT4 log is not expected to reproduce from the new INT8 input. Do not impose the old INT4 prefix equality gate on this experiment.

## Phase 3: blind alignment and averaging

Use the center-baseline signed S0 as a fixed, public-only reference. It fixes indexing and scale, but is NOT a ninth member in the average. For each member, maximize summed absolute cosine similarities between the 16 reference and member rows using Hungarian matching; use sign +1 for nonnegative dot and -1 otherwise. Reject zero/nonfinite row norms; do not discard low-similarity members or use private/source-truth scores to select a reference or member.

After permutation/sign correction, scale each aligned member row to the corresponding baseline row L2 norm. Explicitly, A_m[i]=norm(S0[i])*sign(dot(S0[i],Sm[pi_m(i)]))*Sm[pi_m(i)]/norm(Sm[pi_m(i)]), with deterministic +1 convention at exactly zero dot. Compute cosine/matching and norm ratios in CPU FP64 for stability, store permutation/sign/ratio and per-row matched cosine, then make aligned signed tensors FP32. Aggregate in fixed member order with a documented FP64 sum/8 then FP32 cast: Sensemble=(1/8)*sum_m A_m. Set zensemble=abs(Sensemble) EXACTLY after aggregation; no averaging of absolute sources first. Preserve baseline native S0 and z0=abs(S0).

No oracle_permutation_aligned_cosine_similarity or private tensor can be called by attack. The existing blind_alignment helper is useful reference for Hungarian/sign logic but does not itself implement the registered scale matching; use an isolated helper without editing that protected source. Validate nonzero finite aggregate rows before inversion. Poor alignment or cancellation is a reported limitation, not a trigger to change the method.

Report alignment distributions and relative dispersion about the mean across eight aligned members, aggregate row norms and source/target hashes. Optional read-only prefix summaries at M=2 and4 are allowed ONLY as diagnostics of the same eight saved members, no extra separation/FIA or target selection; M=8 remains the sole ensemble target regardless of diagnostics. Do not present the dispersion of fitted/aligned sources as a calibrated posterior uncertainty or independent confidence interval.

## Phase 4: exactly two Direct FIA runs

Use unchanged upstream_naive.run_upstream_fia / official Direct for z0 and zensemble. Both use the same starting VGG16, seed42, random-image initialization, CUDA FP32,25000 Adam iterations, initial lr.1, existing cosine schedule, feature cosine weight1, normalized TV weight.1, log_every5000. Explicitly reset the registered runtime/seed before EACH run. Record adapter runtime changes as in the completed4000 task. Capture initial image-state hashes to substantiate common initialization. Both targets are in the baseline source order, with no private correspondence in attack.

Save baseline/candidate.pt and ensemble8/candidate.pt as pixels[16,3,224,224] in[0,1], official attack checkpoints/logs, manifests, target bindings, final hashes and timing/resources. Preserve S_hat->abs target->FIA input->candidate provenance. No per-member FIA, gradient matching, label search, true-feature inversion, generator, cell loss or actual-image FC replay. Actual-image cell feasibility is NOT_EVALUATED, never false by scope.

## Phase 5: one offline comparison after public gate

Only after both candidates, all sample code/scale checks, stage provenance and integrity are frozen and PUBLIC_GATE_PASSED may a separate evaluator load the archived evaluation_truth.pt. Invoke configure_runtime(42,threads=1) before metrics. Consume x only; no labels/true z/true updates for computation. If deserialization materializes extra bundle fields, disclose that separately from which fields were consumed. Do not expose private output paths/metrics to ongoing attack stages; attack is already complete.

Evaluate exactly the two new frozen 16-image sets (center INT8 and randomized8 INT8) using the unchanged pixel-MSE Hungarian matching once per candidate, reused for MSE/PSNR/SSIM/VGG LPIPS and an aligned truth/center/randomized8 PNG. Reuse existing cached weights, normalization, metric formulas and evaluation helpers. No downloads or INT4-control retraining/re-evaluation required. Record pairwise means, all per-image values and improved/worsened/tied counts. No private selection of checkpoint/member/reference/hyperparameters.

Scientific contrast is ensemble8 minus INT8-center, keeping all other attack stages fixed. Lower MSE/LPIPS and higher PSNR/SSIM are favorable. Four favorable mean directions support a consistent descriptive improvement for THIS fixed observation; mixed directions are mixed/inconclusive. Do not invent a recovery-success cutoff, statistical significance, optimal M claim, population generalization or proof of why the method helps/fails. New INT8 results do not establish INT4 success. Include total compute cost and note that this first screen compares additional separation computation against a single center run, not a compute-matched multi-start method.

## Isolated implementation, registration and focused validation

Add narrowly scoped files such as qacpa/signed_int8_observation.py, signed_int8_randomized.py, signed_int8_randomized_runner.py, signed_int8_randomized_evaluation.py and corresponding focused tests. Exact filenames may be adjusted consistently before registration. Add task-local launch/report files under .ai/execution-results/TASK_ID. Do not edit the historical INT4 core, replay, upstream adapters, official model/FIA, old runners/evaluators, public/private inputs, metrics definitions or other active handoffs. Use strict new schemas, no-overwrite output roots and public/private path guards.

Before real computation freeze one registration with exact method, sample count/seeds, boundary/anchor/rounding/alignment/normalization rules, codec, shapes, all source/input hashes, execution order, budget and comparison rules. Public attack registration must not contain private numerical values. Final sampler details such as chunk size are fixed in this pre-run registration and not tuned using real reconstruction results.

Focused synthetic tests must meaningfully cover: signed255-level encode/decode and byte accounting, zero/finite/shape validation, endpoint/tie/adjacent-FP32 cells, extreme/anchor and full(q,s) sample consistency, seed reproducibility, preservation of old INT4 contracts, rank/finiteness behavior, permutation/sign/scale recovery on known synthetic sources, deterministic average and exact abs target, fake writer/loader->FIA binding for BOTH arms, identical initialization policy, member count/hash mismatch and output collision failures, sender/attack/evaluator separation, and public gate before private loading. Mock expensive model/separator/FIA calls for integration fixtures; use small real core fixtures where needed. Run relevant focused tests, py_compile and CLI help; no extra full-size pilot or research tuning. If a local code defect blocks an exact prescribed path, repair only the isolated implementation and necessary validation before continuing; log the issue.

## Budget, execution and completion

Fresh real-execution budget5400 seconds (90 minutes), covering one sender conversion, nine separations, two FIA runs, final evaluation and necessary end-of-run reporting computation. Development/focused synthetic tests and idle time excluded. CUDA allocated cap12GiB=12884901888 bytes. Stream matrices/members sequentially; preserve reasonable host memory. Previous anchors: one GD4000 separation about45 seconds, one Direct FIA about1994 seconds. Two FIA plus nine separations suggest roughly70-80 minutes, not a measured promise. No silent budget extension/device/precision/step changes.

Prepare the COMPLETE sequential wrapper before launch. Use C:/miniconda/envs/DLG/python.exe. Preserve hashes and actual stage exits. On a stage failure stop dependent stages, retain valid artifacts, report partial/needs-review accurately; no new scientific run or optimizer restart. Reporting-only repairs may finish from saved outputs within scope without recomputing successful attacks.

Launch persistent and hidden, verify wrapper PID/start once, register one non-model OS process-exit callback to the SAME initialized Executor, record startup receipt and END the Executor turn. No model polling/watcher/new agent. Use BOM-free UTF8 task-local callback transport and preserve global peer bindings. On exit complete only authorized remaining validation/evaluation/reporting, send ONE final result to verified existing Planner, then end. Planner ends immediately after one queue receipt; it does not wait for claim or experiment completion.

Final report and execution_summary.json must include focused validation, full scope/config, sources/hashes and integrity, exact sender true-update consumption distinguished from attack/evaluation, observation byte accounting, member seeds/code/scale checks, all nine FastICA/GD traces and convergence records, alignment diagnostics, both FIA targets/initialization/completion, private gate, metrics and aligned PNG, timings/resources/exits and limitations. Mark scientific quality evaluated only after actual offline metrics. No follow-on M16/INT4/extra experiment is automatically authorized.
