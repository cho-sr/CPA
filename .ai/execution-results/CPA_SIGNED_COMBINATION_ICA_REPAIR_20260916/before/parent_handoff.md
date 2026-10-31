# Adopt ICA-GA feature separation while retaining VGG16 and the existing image-update cell method

TASK_ID: CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: IMPLEMENTATION_AND_FOCUSED_SYNTHETIC_VALIDATION
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_ADOPT_ICA_ONLY_KEEP_MODEL_20260916
REVIEW_REFERENCE: PLANNER_STATIC_REVIEW_OF_LOCAL_ICA_GA_AND_USER_DIRECTED_ADOPTION
PLANNER_SESSION_ROLE: CODEX_PLANNER
FULL_RESEARCH_RUN_STATUS: NOT_AUTHORIZED_BY_THIS_IMPLEMENTATION_HANDOFF
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916/report.md
EXECUTION_SUMMARY_PATH: .ai/execution-results/CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916/execution_summary.json

## Authorization and deliverable

The user explicitly instructed: "우리도 저기 방법 차용하자 Generator model 은 버리고 ICA 쓰는 방식만, 모델은 유지". Implement this adoption; do not ask for the same implementation permission again. The approved change is the feature separator: FastICA initialization followed by prior-based gradient-descent refinement, adapted from the local ICA-GA source. VGG16, the public observation, the existing direct image inversion, and the actual image-generated-update cell loss remain intact. This is not authorization to substitute the ICA-GA generator or target model.

Deliver the working isolated separator, public-data pipeline entry points connecting its output to existing FIA and actual-update cell refinement, focused synthetic CPU tests, documentation, and an implementation report. The user has not specified a new full reconstruction run in this request. Do not run real B16/T10 separation, FIA, replay, private evaluation, training, downloads, tuning, or any new research comparison in this task. Small synthetic numerical tests and mocked orchestration tests are authorized implementation validation. Explicit future CLI commands should be executable after a separate research-run instruction. Mark scientific quality NOT_EVALUATED, not successful.

Use the existing user-designated Luna/xhigh Executor, thread 01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, creation1789518279030ms. Planner verified this process and the matching state database runtime before preparing the handoff; reverify role/runtime and hash before claim. Return one final completion/failure notice to existing Planner thread01a0a799-d62c-7ab0-9c2a-628acf6d2c95, PID24972, creation1789518271364ms. Do not change model settings or peer bindings or create an AI session. This task-specific assignment follows the user's established Luna execution arrangement.

## Scientific invariants and integration scope

- Existing target: same VGG16 public starting checkpoint, frozen convolutional feature extractor, all six FC tensors trained, single client, B16, same ordered batch for T10 SGD steps, lr .001, momentum/weight_decay/dropout zero. No new observation collection.
- Current observation is signed deterministic symmetric INT4, codes [-7,7], scalar scale per parameter tensor, CPU FP32 codec, existing exact cell/tie/finite-edge semantics. Historical affine/radius entries in STATE are not this packet. Preserve the codec, scale semantics and all existing observations.
- Existing public packet: quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation; manifest SHA256 9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69.
- Public checkpoint: C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth; SHA256 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0.
- Attack input for the new separator is exactly classifier.0.weight decoded as q*s, shape [4096,25088], source count16. No sign/learning-rate conversion, rank/cell preprocessing, randomized dequantization, extra denoiser, trainable latent update, learnable mixing matrix, or joint A,S objective is added. Those would be additional method changes.
- New separated features must retain their native estimated order. Save signed S_hat and target_z=abs(S_hat). Do not standardize target_z as ICA-GA does for its generator; existing Direct FIA consumes the abs-source target. No generator, generator checkpoint, auxiliary-data training, or ICA-GA model imports.
- Preserve original Direct FIA exactly through upstream_naive.run_upstream_fia: seed42, 25000 iterations, Adam lr .1, cosine schedule, feature-cosine weight1 and normalized-image TV weight.1. Its existing implementation and model adapter are read-only. The current implementation task only exercises this via mocks.
- Preserve the actual image-update cell path for subsequent refinement: image -> frozen VGG16 feature -> canonical10-step FC SGD -> all six unprojected updates -> existing signed cell penalty -> image. Do not replace this with a surrogate low-rank matrix or decoder output constraint. Labels and targets remain fixed during each refinement run as in the existing method.
- Original method/results remain an independently identifiable control. Name the new method explicitly (for example INT4_combination_ica); do not relabel it original CPA or silently overwrite CPA_CONFIG. The comparison changes initialization, prior formulation/weights and separator optimizer as one adopted package; it does not isolate the effect of each component.

## Read-only source references

ICA-GA-main/ICA-GA-main/attacks/ica_ga.py
ICA-GA-main/ICA-GA-main/attacks/feature_separation/{base_ica,fast_ica,gradient_descent_ica,combination_ica}.py
ICA-GA-main/ICA-GA-main/LICENSE
quantization_aware_cpa/qacpa/{upstream_naive,signed_baseline,signed_baseline_observation,signed_cell_rank_runner,signed_e2e_runner,signed_e2e_cells,signed_e2e_evaluation}.py
quantized_update_cpa/{cell_utils,quantization}.py
official_cpa/src/{gradient_inversion,feature_inversion}.py

Keep all of these existing source files unchanged. Implement in new modules rather than importing ICA-GA's attack class, private-data interface, generator or evaluation code. Preserve the upstream MIT notice for adapted code, including the exact existing copyright notice, and document source paths/hashes and intentional corrections.

## Separator specification

Implement a pure tensor core accepting only observed matrix X, source count and explicit numerical configuration. It must have no true source/image/label arguments, evaluation callbacks, oracle ordering, or private file paths. Model/checkpoint access belongs only in the pipeline, not the core.

Whitening: preserve the referenced ICA-GA algebra and FP32 convention. For X of shape[m,d], center each row, mu=mean(X,dim1), C=(X-mu)(X-mu)^T/(d-1), select the largest absolute eigenvalues from torch.linalg.eig as the local code does, K=diag(1/(sqrt(abs(eigenvalue))+1e-10))*selected_eigenvectors.T, Xw=K*(X-mu). Compute this once on CPU and share the SAME K, mu, Xw between both stages; then transfer the small/source-space work to the requested device. Do not use an independently recomputed whitening basis for the second-stage initialization. Record the algorithm, dtype/device and numerical limitations. Validate input shape/source count/finiteness and fail clearly for unusable rank/nonfinite output rather than silently changing source count/backend/dtype. Do not substitute a new eig/SVD denoiser.

FastICA initialization: parallel/symmetric fixed-point method, logcosh derivative tanh with alpha1, identity initial W followed by symmetric decorrelation, max200 steps, tolerance1e-6 from max_i abs(abs(dot(Wnew_i,Wold_i))-1). Correct formula:

    U = W @ Xw
    g = tanh(U)
    a = mean(1-g*g, dim=1)
    Wnew = symmetric_decorrelation((g @ Xw.T)/d - a[:,None]*W)

The upstream local fast_ica.py:152 uses `a*W`, which broadcasts along columns and is incorrect for this row-wise derivative term. Correct it in the new adapter and verify with a nonconstant-a fixture. Independent reference: scikit-learn official sklearn/decomposition/_fastica.py, _ica_par uses g_wtx[:,np.newaxis]*W. Symmetric decorrelation uses eigh(W W^T) and the upstream finite-positive numerical floor; no truth-based convergence or restarting. Accept mathematically valid first-step convergence (upstream rejects it); report convergence, actual iterations and residual. Finite max-iteration nonconvergence may continue to gradient refinement using the last iterate, explicitly flagged, matching the intended two-stage flow. Nonfinite state must fail; no silent fallback to identity/random or different method.

Prior-based refinement: initialize W from that exact FastICA output; plain SGD, lr .001, momentum0, weight_decay0, no scheduler,1000 iterations (the ICAGA constructor's setting, not the standalone GD class's4000 default). Normalize each row as Wbar=W/(row_norm+1e-10). Let T=Wbar@Xw and S=T+Wbar@K@mu. Use:

    Lne = -mean_i(mean_k(log(cosh(T_ik)+eps))**2)
    Ldecor = sum((abs(Wbar@Wbar.T)-I)**2)
    Lsign = mean_i(min(norm(relu(-S_i),2), norm(relu(S_i),2)))
    Ll1 = mean(abs(S))
    L = 0.0935827857091549*Lne
        + 0.2513674104161625*Ldecor
        + 0.05811961446881231*Lsign
        + 17.94138146247377*Ll1

These are imported source defaults, not values established optimal for VGG16 or INT4. Negentropy-like contrast is ACTIVE (do not carry over official_cpa's unconditional zero overwrite). This code's contrast is not exact entropy or a guarantee of independence. Use a stable algebraically equivalent log(cosh(t)+eps) implementation to avoid avoidable cosh overflow; verify values/gradients on finite moderate arguments against the direct expression. No new TV on flattened features and no Gaussian-reference subtraction/new objective terms. Nonfinite losses/gradients fail with a diagnostic instead of silently dropping the term.

Use final W after the1000 updates and recompute final signed S. Output detached contiguous CPU FP32 S_hat and target_z=abs(S_hat), their hashes and honest statistics. Do not choose the best iteration using private quality or a new unregistered selector. Source sign/ordering evaluation stays outside attack. Record exact stage configuration, finite checks, timing and convergence in the result.

## New files and executable integration

Suggested new paths (small additional files with the same prefix are allowed if justified):

- quantization_aware_cpa/qacpa/signed_combination_ica.py: pure separator core.
- quantization_aware_cpa/qacpa/signed_combination_ica_runner.py: explicit public-only subcommands for separate, invert and refine; optional combined attack command may chain separate+invert only when explicitly invoked.
- quantization_aware_cpa/tests/test_signed_combination_ica.py and test_signed_combination_ica_pipeline.py.
- quantization_aware_cpa/docs/signed_combination_ica.md and an accompanying upstream license notice.

`separate` validates the current signed public packet, decodes q*s and writes new sources.pt plus registration/result/logs. `invert` authenticates that exact completed source artifact, public packet and checkpoint, then calls the unchanged Direct FIA adapter and saves candidate.pt with pixels in the existing shape/range/schema. Use a new output root with no-overwrite checks and distinct method identity. Register input/source/config/runtime hashes before the actual optimization. Report completed and not-run stages truthfully; writing a runner is not a completed reconstruction.

`refine` makes the adopted target usable with our existing actual-update quantization-aware method. Create a thin new orchestration boundary accepting authenticated new source/candidate artifacts; reuse signed_e2e_runner's lower-level feature, replay, run_arm, profiling and path-check functions, and signed_e2e_cells unchanged. Do not call historical run_screen through a fake naive identity or loosen its provenance/path guards. If small orchestration logic must be duplicated to avoid changing historical code, document it.

Future refine preset: same actual100-step image optimizer/LR schedule/box as signed_e2e_runner.run_arm; feature weight1, TV.1, fixed cell weight0.24202997768915402; one coverage-based label assignment computed from this new candidate's public logits and observed positive last-bias codes using existing labels_from_public_logits. Use that routine's coverage-based hypothesis, freeze it before optimization, and explain the rule in words in user-facing reports. Do NOT reuse the old slotwise labels blindly: the new separator has an arbitrary source permutation. The assignment rule is preserved, but its output may change and must be disclosed. No label search, swap, recalibration, alternative arms or new tuning. This is a preserved-method integration preset, not authorization to run it now.

Before future refinement, retain the existing canonical/adjoint terminal-delta and feature grad/no-grad equality checks. Profile the start and final candidate over ALL6 tensors with the original exact-code and bounded-cell tests; scale diagnostics remain separate. Keep the original protected sources and inputs hashed before/after. Write endpoint artifacts incrementally so an optional reporting error cannot discard already-computed evidence. Make any output validator handle its explicitly versioned schema rather than requiring unrelated historical extra fields or silently intersecting keys.

Document future paths, for example quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916. Do not create research outputs/candidates by running actual data in this task. Provide exact CLI commands for the staged new pipeline and explain how existing original-separator results remain the control. Existing evaluators may reject the new method identity; do not bypass those guards. If an evaluator adapter is needed for a future comparison, document its required strict provenance and keep private loading out of this implementation task; implementing an isolated compatible offline adapter is allowed but running it is not.

## Validation and integrity

Use C:/miniconda/envs/DLG/python.exe for focused CPU tests. No full-matrix observation tensors or VGG model forwards in tests. Use small synthetic source mixtures and dependency mocks, without importing the full local attack class.

Required meaningful checks:

1. Correct row-wise FastICA derivative broadcast against an explicit per-row formula on a nonconstant derivative-mean fixture; symmetric decorrelation and finite/convergence handling, including first-step convergence and max-iteration nonconvergence.
2. Shared whitening basis between stages, row-normalization and source shape; rejection of invalid/nonfinite/unusable input. Tests must not assume exact eigenvector signs under ties.
3. Stable logcosh loss values/gradients agree with the stated finite expression; active non-Gaussian contrast contributes to W gradients (no unconditional zero overwrite). Check the four terms and recorded preset coefficients independently.
4. Real two-stage execution on a small non-Gaussian synthetic mixture produces finite signed sources and exact abs targets with deterministic seeds. Do not use synthetic quality to claim real INT4 improvement or add a brittle universal recovery threshold.
5. Core/interface has no truth/label dependency. Run tests with truth readers blocked and confirm no generator import/instantiation. Native estimated order is not aligned to true sources in attack.
6. Mock pipeline proves the adopted target actually reaches unchanged Direct FIA and subsequent actual-update cell refinement; all6 tensors and the original cell function are used. Same checkpoint/config/seed and source/candidate hashes are bound. The correct new target replaces the old target, not merely a unused saved file.
7. Negative provenance tests: changed packet/source/candidate/checkpoint hash, unsupported method/config, incomplete separator output, invalid shape/range and output collision fail before dependent work. No weakening of historical validators.

Run relevant existing small tests only if the new integration depends on those interfaces. No broad unrelated suite or real GPU tests. Preserve all existing sources, observations, results and checkpoints; record hashes for reused sources and adapted upstream files before/after, plus the precise list of new files. Never modify historical experiment metadata to pass a new validator.

## Completion

Finish implementation, focused tests, docs and report without re-requesting adoption permission. If an ordinary implementation/test defect is found, repair it within this scope. If an integration requirement cannot be met without changing a protected scientific condition, implement independent approved portions and report the exact remaining limitation, not an invented success.

Report test command/results/warnings, changes relative to upstream (broadcast, shared whitening, stable logcosh, first-step convergence, truth-free interface), executable future commands, source integrity, method limitations, and clearly: research runs0, private tensors loaded0, generator models used0. No claimed image improvement. Follow WORK_MODE; no model polling or background monitoring agents. If a long validation wrapper is used, launch it persistently/hidden with one existing process-exit callback and end the turn. On completion send exactly one result notice to the verified Planner and end. No automatic full experiment follows.
