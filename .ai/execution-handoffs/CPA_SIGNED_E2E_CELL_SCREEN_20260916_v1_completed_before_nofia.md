# Actual-image signed-cell experiment: label initialization and penalty strength

TASK_ID: CPA_SIGNED_E2E_CELL_SCREEN_20260916
PLAN_VERSION: 1
STATUS: COMPLETED
TASK_KIND: IMPLEMENT_VALIDATE_AND_RUN_BOUNDED_E2E_SCREEN
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_ALLOW_EXPERIMENTS_TO_RESOLVE_AMBIGUITIES_20260916
REVIEW_REFERENCE: RELAY_TURN_56_AND_CODEX_CORRECTIONS
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_E2E_CELL_SCREEN_20260916/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_E2E_CELL_SCREEN_20260916/execution_summary.json
PUBLIC_OUTPUT_ROOT: quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916
COMPLETED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab

## Final execution receipt

RESULT_STATUS: COMPLETED
WRAPPER_PID: 12380
WRAPPER_EXIT_STATUS: 0
TESTS_EXIT_STATUS: 0
RUNNER_EXIT_STATUS: 0
EVALUATOR_EXIT_STATUS: 0
REGISTERED_ARM_COUNT: 5
ARM_STEPS: 100 each
TOTAL_WRAPPER_WALL_SECONDS: 641.903975400026
PUBLIC_OPTIMIZATION_WALL_SECONDS: 600.6513554999838
PRIVATE_EVALUATION_WALL_SECONDS: 24.376909699989483
FOCUSED_TEST_RESULT: 85 passed, 240 warnings
NUMERICAL_PATH_RESULT: terminal_deltas_torch_equal=true; feature_grad_enabled_equals_no_grad=true; all six tensors torch_equal=true
INTEGRITY_RESULT: protected_sources_unchanged=true; inputs_unchanged=true
EXACT_FEASIBILITY_RESULT: false for x0 and every final arm under both H0/H1 endpoint profiles
PRIVATE_LABEL_BOUNDARY: labels_consumed=false; true updates not used
PLANNER_NOTICE_STATUS: SENT_ONCE
PLANNER_NOTICE_THREAD: 01a0a799-d62c-7ab0-9c2a-628acf6d2c95
PLANNER_NOTICE_MESSAGE_ID: 01a0a91f-0350-7493-843c-254cbf410ee9

## Authorization and objective

Latest direct user instruction: "애매한 지점은 실험을 통해 방향을 결정해도 돼". It follows the user's insistence on DeltaTheta(x,y_hat) in C(q,s), the concrete image-update experiment proposal, Claude's independent review, and Codex's corrections. This authorizes the bounded public checks, implementation/validation, and comparative experiments below to resolve label-initialization and penalty-strength uncertainties. No repeat approval is needed for this concrete scope or ordinary implementation repairs. No unlimited tuning, new observation, seed, model, codec, private-data use in attack, or extension beyond these registered arms is authorized.

Both reviewers supported the actual-image update path. The specific finite comparison settings below are Planner selections under the latest experimental authorization, not a claim that Claude agreed to every numeric value. Read .ai/relay-runs/signed-end-to-end-turn56-codex-assessment.md: do not reuse Turn56's incorrect converse label-sign inference, fixed-label feasibility claim, exact-feasibility relaxation, AP5 rank/cell intersection claim, or automatic stop when initial cell loss is zero.

Executor is the existing user-designated Luna/xhigh CLI. Before claim, verify the receiving session's user-assigned execution role, current model/effort, current handoff ID/version/hash, workspace and no duplicate job. It was observed open as thread01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, creation1789518279030ms, but had no saved initial conversation at preparation time. The user was asked to initialize that existing window; reverify after initialization and do not assume its thread ID survived. Do not start another expert, switch models, infer a role from PID/window order, or modify Planner/Claude bindings. Planner return target: current thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID24972; reverify before the one final result notice.

## Fixed scientific inputs

Use the EXISTING signed packet at quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation and existing naive artifacts at quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive. Preserve all historical files. Signed CPU FP32 symmetric INT4 nearest-even, q[-7,7], observed positive scalar s per parameter tensor, exact existing endpoints/caps/zero convention. This is not the older affine/radius packet still described in historical STATE entries.

Public checkpoint C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth; VGG16 Conv fixed, all six FC parameters trained; B16/T10, same batch order repeated, SGD lr=.001, momentum=weight_decay=0, dropout inactive, mean cross entropy; raw theta_after10_minus_theta_start. Seed42 and existing normalization/runtime settings. No actual labels, original images, oracle embeddings or true updates enter attack/proposals/selection/calibration/stopping. Hashing a private artifact for provenance is not permission to deserialize it in attack.

Common starting pixels x0 = finalized naive/candidate.pt pixels (all16 in stored order). Fixed FIA target = naive/sources.pt target_z, validate target_z==abs(S_hat), shape/dtype/provenance. This is a NEW continuation from saved final pixels with fresh optimizer, not an exact resume of old Adam or an AP5 continuation. Do not rerun CPA, AP5, observation collection, historical naive/reference FIA, or latent-G optimization.

## Stage 1: public registration and label hypotheses

Read/validate public manifest/packet hashes and checkpoint; decoder uses existing signed loader. Register all input/source hashes, actual runtime and the exact plan BEFORE real optimization or private evaluation. Report tensorwise histogram over all15 codes, positive/negative/zero counts, s, and nonzero-code fraction. Resolve classifier.6.bias through manifest key, not a guessed filename.

Compute x0 features Z0 using exactly one ImageNet normalization, frozen/eval FP32 VGG16 features->avgpool->flatten, and the public starting classifier logits A[i,c]. Create two deterministic hard-label hypotheses, no private data:

- H0: y0[i]=argmax_c A[i,c]; first class index wins exact ties. Repeated classes are allowed.
- H1: positive-bias-coverage hypothesis. Let P be ascending class IDs with observed classifier.6.bias q_c>0, and k=|P|. Positive raw bias update is one-way evidence of presence under this sender rule; neither q<=0 nor magnitude determines absence/multiplicity. Do NOT claim H1 recovers the complete true labels.
  If0<=k<=16, build a16x16 reward matrix: columns0..k-1 have A[i,P[col]]; every remaining free-slot column has max_c A[i,c]. Solve one deterministic maximum-reward Hungarian assignment with the installed scipy linear_sum_assignment, recording version and full returned assignment. A row assigned a mandatory column gets that class; a row assigned a free column gets y0[i]. This enforces one occurrence of each positive-code class while allowing additional occurrences and arbitrary other classes. It does not exclude negative/zero-code classes. No RNG tie-breaking or score perturbation. Check deterministic repeats on a synthetic tie fixture.
  If k>16, report MODEL_OR_OBSERVATION_LABEL_HINT_CONFLICT and do not silently prune P. Stop scientific optimization pending review, preserving diagnostics and tests; do not invent another label rule.

Freeze each label vector throughout its arm. No online swaps or additional label search in this first screen. If H1==H0 exactly, report the duplicate hypothesis and run that unique vector once (two B arms instead of four); do not invent a replacement. Fixed wrong labels can make the restricted image problem infeasible. Results cannot establish class accuracy or impossibility of the unrestricted method.

## Stage 2: new actual-update cell loss and image runner

Implement separate signed_e2e_cells.py and signed_e2e_runner.py (or equally clear new names documented before launch). Reuse fc_sgd_updates and fc_recomputed_gradients without modifying the original replay, CPA, FIA, codec, AP5 or baseline files. The caller owns the image->Z graph and explicitly propagates the returned numerical dL/dZ through it. FC replay must have differentiable trainable FC parameters cloned functionally from the public start; Conv weights stay frozen. Every outer call starts the local trajectory from the same theta0, not the previous replay endpoint.

For each of the SIX observed FC tensors j, compute exact signed L_j,U_j from existing CPU FP32 cell_utils.get_cells semantics. Build the small15-code tables on CPU, copy the exact FP32 values to the computation device, and expand in chunks. q/s are immutable. Do not use affine radius loss, midpoint attraction, a scale anchor, or change endpoints to improve a score.

    r_j = relu(L_j - Delta_j(x,y)) + relu(Delta_j(x,y) - U_j)
    L_C = (1/6) * sum_j mean((r_j/s_j)^2)

Aggregate tensor means equally, explicitly recording this weights small tensors more per element. Every coordinate remains included. Chunk size65,536 elements. Use a custom autograd Function with chunkwise residual recomputation in backward (or an equivalently bounded analytic VJP implementation verified against reference); do not retain all chunk graphs. FP32 input/bounds, FP64 scalar accumulation allowed and declared consistently. No silent loss-scale floor or dtype/backend fallback. Zero q with s=1 is an interval, not an equality-to-zero constraint. Nonfinite values cause a recorded failure.

Optimization variable is pixel-space x in[0,1]. Define u=(x-mean)/std once; FIA target loss is mean(1-cos(f(u),target_z)) and TV uses the ORIGINAL Direct.total_variation definition applied to u (not pixel x). Weights lambda_F=1, lambda_TV=.1. Pixel-space optimization is declared as a new continuation; do not claim .001 pixels equals the historical .1 normalized-space LR.

At x0,y0 compute finite L_C0 using actual replay. Define ONE shared scalar lambda_base=1/max(1,L_C0), for all H0/H1 arms. This normalizes the initial H0 cell-loss scale without calibrating against a possibly vanishing FIA gradient; max(1,...) affects only the chosen global weight, NOT cells or tensor s. Define lambda_low=lambda_base and lambda_high=10*lambda_base. These two weights are experimental choices, not validated optimal values. Record actual initial loss/gradient norms for each label hypothesis and resulting weights before launch; no reweighting after seeing trajectories or private metrics. If L_C0=0, weights are1 and10 and the experiment still runs because later FIA steps may leave the cells.

## Stage 3: validation and full-size resource check

Required focused CPU synthetic validation, no real private fixtures:
- ordinary hard-label SGD vs replay and reverse-recomputed terminal deltas under the fixed recurrence;
- signed inside/outside/tie/cap/zero cells, chunk versus nonchunk loss and analytic/autograd gradient;
- directional finite differences away from model/cell kinks for the composed image/feature->SGD->cell loss; integration proving lambda_C changes image gradients through replay rather than a detached matrix;
- public-only H0/H1 including repeated labels, ties, k0,k16,k>16, duplicates; reject use of q sign as an absence rule or label magnitude as count;
- immutable theta0, all-six tensor coverage, correct raw-delta sign, one normalization and normalized-space TV;
- provenance/private-loader guards and fixed-step/final-only output selection.

Then one full-size x0 resource/composition check, before the planned arms: compare canonical fc_sgd_updates(create_graph=False, checkpoint_forward=False) deltas with the adjoint path's terminal deltas under y0, each tensor max difference in units of s and exact-code differences. If diagnostic access to the latter requires a helper, put it in NEW code using the same recurrence; do not weaken production validation or rewrite existing replay. Check exact equality of x0 features with grad enabled versus no_grad under the registered settings. Any nonzero terminal-delta difference or feature difference is a NUMERICAL_PATH_MISMATCH: stop before arms, report exact discrepancy and preserve artifacts. Routine implementation mistakes may be repaired and focused tests rerun within this task; do not mask genuine numerical divergence by changing scientific settings.

Compute one real recomputed cell image gradient; record finite/nonzero or explained zero, wall time and CUDA allocated/reserved peaks. Zero loss/gradient at feasible x0 is not a failure or an automatic stop. A nonzero loss with zero image gradient is diagnostic evidence: report and allow registered fixed arms if finite, without inventing a new objective.

Limits for the REAL phase (public x0 checks, calibration, all arms, final replay and offline evaluation): one global45-minute wall-clock ceiling,12GiB PyTorch peak allocated CUDA memory. Check at safe boundaries and record total stage times. These are caps, not cost predictions. Stop and mark PARTIAL/NEEDS_REVIEW on actual OOM/deadline/cap/provenance/numerical-path failure; do not change batch,T,precision,training,weights,seed or resume/restart a completed real arm. Existing tests do not establish full-size memory success.

## Stage 4: preregistered finite experiment matrix

All unique arms start from EXACTLY the same saved x0 and fixed target, fresh Adam state, pixel-space lr=.001 decaying by cosine to .0001 over100 image steps, betas(.9,.999), eps1e-8, weight_decay0. Clamp pixels to[0,1] after each step. Use seed42 and deterministic/TF32 flags matching existing signed configure(). Each arm is run sequentially once, with separate state/output. A completed step means after image update/clamp; logs label pre-step losses separately from post-step checks. No private early stopping or best-iterate selection.

1. A_FIA: L_FIA+.1*TV,100 steps.
2. B_H0_low: same plus lambda_low*L_C(x,H0),100 steps.
3. B_H0_high: same plus lambda_high*L_C(x,H0),100 steps.
4. B_H1_low: same plus lambda_low*L_C(x,H1),100 steps, omitted only when H1==H0.
5. B_H1_high: same plus lambda_high*L_C(x,H1),100 steps, same duplicate rule.

Maximum500 image optimizer steps,400 of them with full update reverse gradients, plus declared checks. No additional arm, label swap pass, auto tuning, long continuation or seed is authorized. Comparison across weights fixes labels; comparison across labels fixes the same numeric lambda. Equal image iterations are NOT equal compute: report all time, replay counts and peaks.

Save each final image set and hashes before private evaluation. At x0 and each final set, canonical hard replay evaluates BOTH unique H0/H1 vectors, allowing image and label fit to be distinguished (maximum12 ordinary full-batch replay evaluations for these endpoint checks). This is public-only reporting, not an extra optimization arm or a selector. The optimized label vector remains explicitly identified.

Report per tensor and overall fixed-s code mismatch, cell violation count/fraction, max and mean violations in s units, L_C, candidate-scale diagnostic; encoder norms, pixel dynamic range/TV, image displacement from x0 and feature-target fit. Exact feasibility means ALL six unprojected replay tensors have zero bounded-cell violations and zero fixed-s code mismatches. A projected surrogate cannot certify it. Scale equality/full packet reproduction is a distinct diagnostic, not silently required or asserted.

## Stage 5: finalized offline evaluation and directional decisions

Only after all scheduled unique arms are finalized (or a task stops with explicit partial status), the standalone new evaluator may use existing private x/z for output evaluation. Do not use actual labels or true updates in attack. Keep existing Hungarian pixel-MSE matching, PSNR/SSIM/VGG LPIPS over all16 images, full per-image scores and original/x0/A/B figures. Use installed weights only. Avoid loading trueG since it is unnecessary here. Private labels bundled with x/z may materialize on torch.load; disclose labels_consumed=false and never index them.

Reuse appropriate existing metric helpers, but do not falsify AP5/naive arm metadata to pass a validator built for a different algorithm. The new comparison authenticates its real input checkpoint/manifest/candidate/target/label/plan/source/settings hashes and phase boundary before loading private tensors. The proposed label hypotheses, lambda values,100-step outputs and all scheduled arms are frozen before oracle access. No winner is chosen to trigger further training.

Prespecified interpretation:
- For each fixed label vector, compare low vs high penalty with all public fit profiles and all image metrics. For each fixed lambda, compare H0 vs H1. x0 and A distinguish improvement from simple continuation.
- Exact feasibility is binary under the existing cell/code definition. Also report partial progress. For a vector of violation metrics (code mismatch fraction, cell violation fraction, L_C, maximum s-unit violation), nonworse in all and strictly better in one is descriptive domination; conflicting changes are MIXED, not a forced winner.
- For PSNR/SSIM/LPIPS, nonworse in all and strictly better in at least one is consistent directional image improvement on THIS batch. Mixed metrics remain mixed. No invented absolute reconstruction-success cutoff, statistical significance, or generalization. Tiny changes are reported as measured, not automatically noise or meaningful success.
- Recommend the next research direction supported by these paired contrasts; if label/weight effects remain entangled or no condition improves, say INCONCLUSIVE. The task ends with the report, without an automatic next experiment. Fixed-label failure does not prove unrestricted infeasibility.

## Files, artifacts and completion

Allowed research edits: new quantization_aware_cpa/qacpa/signed_e2e_cells.py, signed_e2e_runner.py, signed_e2e_evaluation.py (names may be adjusted transparently); new focused test files; new docs for this experiment. Protected: official_cpa/**; all existing codec, replay, CPA/FIA, signed baseline/AP5/diagnostic source and old outputs. Hash protected files before/after. Do not alter FIXED_CONSTRAINTS or rewrite historical results. Use task-specific temp/cache paths and preserve old data.

New public output root: quantization_aware_cpa/results/signed_e2e_cell_screen_B16_T10_20260916/ . New private evaluation root: quantization_aware_cpa/results_private/signed_e2e_cell_screen_B16_T10_20260916/ . Refuse unexpected preexisting outputs; do not overwrite/resume an unrelated run. Store registration, public histograms, H0/H1 and assignment evidence, common-x0/target hashes, exact weight calibration, resource/numeric checks, all per-step logs, final candidates/replay profiles, invocation records and exit codes.

Task report directory: .ai/execution-results/CPA_SIGNED_E2E_CELL_SCREEN_20260916/ with report.md, results.json, tests.log, exact commands, integrity manifest, runtime/resource records and completion receipts. Report execution completion separately from scientific success/partial feasibility. COMPLETED only after the authorized implementation, tests, scheduled bounded experiment and final reporting all finish; otherwise explicit FAILED/NEEDS_REVIEW/PARTIAL with completed work and cause.

Follow WORK_MODE: no model-based polling while the experiment runs. If backgrounded, launch the execution wrapper hidden, ensure persistence, register one non-model process-exit callback to this VERIFIED existing Executor, and end the turn. On process exit finish only the authorized evaluation/report, send ONE final result/failure notice to the existing Planner after identity verification, and end. Do not create expert/monitor agents, send research work to Claude, or relaunch the experiment automatically. Planner performs at most one immediate delivery/status check and ends after queue receipt.
