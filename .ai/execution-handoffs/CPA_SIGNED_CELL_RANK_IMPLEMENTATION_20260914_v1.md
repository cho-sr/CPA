# Signed INT4 rank/cell comparison implementation

TASK_ID: CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914
PLAN_VERSION: 1
STATUS: READY
TASK_KIND: IMPLEMENTATION_AND_SMALL_CPU_UNIT_TESTS
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: NONE
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_LUNA_IMPLEMENTATION_REQUEST_20260914
CONSENSUS_REFERENCE: RELAY_TURNS_43_TO_46_WITH_CODEX_FINAL_C5_CORRECTION
DELIVERY_STATUS: READY_FOR_ONE_TIME_QUEUE
PLANNER_SESSION_ROLE: CODEX_PLANNER
FULL_EXPERIMENT_AUTHORIZATION: NOT_GRANTED_BY_THIS_IMPLEMENTATION_REQUEST
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914/report.md
MODEL_DECISION: User explicitly selected the existing Luna CLI for this implementation task

## 1. Authorization, deliverable, and stopping point

[FACT] The user asked whether comparing methods was worthwhile. Planner proposed two arms: existing signed naive sq -> original CPA -> original FIA; and external rank-16/cell alternating projection -> the SAME original CPA/FIA. The user then explicitly instructed: "계획짜서 루나한테 만들라해". This authorizes Luna to implement the comparison pipeline and the small CPU unit/integration tests needed to validate the implementation. It is not a new request to run the real B16/T10 reconstruction experiment. The prior Luna task was read-only and is completed; this is a new ID and newly authorized implementation scope.

Recipient is the already-open Codex thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID18272, runtime gpt-5.6-luna/xhigh. Planner verified process identity, workspace, saved CLI rollout and matching current ThreadSettings evidence (logs_2.sqlite row5741830). Reverify before claim. Do not change models, global role files, peer bindings, or launch another executor. The user's task-specific Luna selection supersedes historical Astra/high role wording for this task only.

Deliver working NEW modules, focused small CPU tests, documentation, and exact future experiment commands. Stop after implementation/testing/reporting. Do not invoke real-data preprocessing, full-matrix diagnostics, CPA/FIA, SGD replay, collection, GPU tests, real-image evaluation, or downloads. Small synthetic CPU fixtures and mocked pipeline integration are authorized as part of implementation; they are not evidence of research quality. No worker, monitoring loop, automatic future experiment, or replacement CLI. Send one final completion/failure report to the existing Planner under the established result-notification policy, then end the turn.

## 2. Fixed scientific contract

- Current codec is signed deterministic symmetric INT4 nearest-even, q in [-7,7], CPU FP32 s=FP32(max_abs/7), scalar PER PARAMETER TENSOR. Zero tensor uses q=0,s=1. Use the existing codec and exact-code cell utility unchanged, including precision/ties/finite edge cap. Historical affine per-row/radius data in STATE/RELAY is not this observation.
- VGG16 Conv frozen, all six FC tensors trained, single client, batch00 B16, same batch/order for ten SGD steps, lr=.001, momentum=weight_decay=0, dropout0; existing preprocessing/seed42 unchanged. This implementation collects no new update.
- Attack tensor classifier.0.weight has q int8 container of shape [4096,25088], s scalar float32. Global round/client IDs are null with NOT_RECORDED in the existing manifest: report UNKNOWN, do not fill them.
- The attacker sees only existing public q,s plus public checkpoint/training metadata. True x,z,G,y are never attack, tuning, stopping, checkpoint selection, or ranking inputs. No label estimation, soft-label variables, reconstructed update replay, quantization-aware FIA loss, or image-update reranking.
- Original CPA source/whitening/loss/optimizer and existing FIA, codec, baseline code/results must remain unchanged. CPA still uses original eig/top16 path, W-only optimizer, disabled-by-existing-code NE, and inactive embedding TV when nv=.13. Do not fix NE, add a whitening loss, enable embedding TV, or substitute legacy eigh CPA/quant_aware_ica. New external matrix projection is explicitly a different preprocessing stage, not a CPA whitening replacement.
- No midpoint-attraction term, scale loss, scale anchor/rescaling, standard MP threshold, dither assumption, iid Uniform quantization-noise model, or new weighted selector. Scale remains a raw metric. Do not ask the user to decide those settled rules again.
- Baseline reference is existing INT4_naive, not FP32 Reference as attacker input. Preserve FP32 Reference as evaluation/context only; this task does not rerun it.

## 3. Existing evidence and inputs (read-only)

Repo root: C:/Users/조상제/Desktop/CPA
Python for tests/future research commands: C:/miniconda/envs/DLG/python.exe

Existing public observation:
quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation
manifest SHA256: 9b4891765e6fc7e4adb1e3e11c158b3d4a2c3430d3397e1f1466ddafe2e1cd69

Existing public naive arm:
quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive
candidate SHA256: fe683f68013896a407a38130ee54ef9dc799a9ca1abf7c4f08efeefea7d91751
sources SHA256: c033e0df9304a6e0dcb19491bb43c3ac80cabed52566409e0f76dbe74194d7eb
raw-sq input tensor SHA256: 815f28e872de6f1b1d565cb413f26e958fa66758cd6fbbaad85758f77552f69c

Checkpoint for future attack:
C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth
SHA256: 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0

Private observation, ONLY for the future separately invoked offline evaluator:
quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation
(sender_validation.json, evaluation_truth.pt, raw_updates.pt).
Existing evaluation report/JSON may be read now, but do not load real private tensors or rerun evaluation during implementation.

Read these implementations before editing:
- quantized_update_cpa/cell_utils.py and quantization.py; trial_metrics.py
- quantization_aware_cpa/qacpa/signed_baseline.py, signed_baseline_observation.py, signed_baseline_evaluation.py
- quantization_aware_cpa/qacpa/upstream_naive.py; full_update_evaluation.py
- official_cpa/src/gradient_inversion.py and feature_inversion.py (read-only)
- existing test_cell_utils.py, test_upstream_naive.py, test_signed_baseline.py, test_signed_baseline_provenance.py
- .ai/proposals/CPA_LATENT_UPDATE_DISPUTE_REVIEW_20260914.md and RELAY Turns45-46

Completed baseline facts: same CPA25k/FIA25k, seed42, abs(S), all16 final images; Reference/naive PSNR12.391145/11.517332, SSIM.346884/.284894, LPIPS.459648/.822925. New method efficacy remains UNKNOWN. CPA+FIA arm time was approximately 36 minutes previously; this is context, not a new timing guarantee. New preprocessing/evaluation cost and full-size memory remain UNKNOWN.

## 4. Mathematical specification of the new preprocessing

[PROPOSAL adopted for IMPLEMENTATION ONLY]

    V_0 = dequantize(q,s) = s*q
    min_{V in C_box(q,s), rank(M)<=r} 0.5 * ||V-M||_F^2
    M_t = best_rank_r(V_t)
    V_(t+1) = project_to_cells(M_t, L, H)

C_box uses existing get_cells(q,s,CODEC_METADATA) semantics. H is the upper bound here to avoid confusing it with unmixing U. The scientific runner fixes r=16 from B16; the core may accept other ranks for small unit fixtures. Operate on the UN-CENTERED raw-update matrix: the rank bound concerns DeltaW=A Z. Do not center for this projection, add a mean back, convert to -Delta/lr, or divide by T/B. Original CPA will independently center/whiten its final input.

Use the final cell-projected V as CPA input. Keep V and M conceptually and in diagnostics distinct: V is cell/code-feasible but need not have rank16; M has the target numerical rank but can violate cells. Scale equality is not imposed. No assertion that convergence, nontrivial movement, a feasible intersection, or true images are guaranteed.

Proposed first configuration: 5 complete alternating steps, rank16, final-step-only selection, no multistart or tuning. Five is a Planner engineering default, not a user-specified number or validated optimum. Expose --ap-iterations (nonnegative; 0 is an identity control) and record its exact value. Never select the best iteration using truth or a new score. Nonfinite results/code violations fail the run; do not silently repair via a different codec/scale/backend.

Rank backend: implement a memory-conscious uncentered row-Gram eig projector named explicitly `gram_eigh`, outside CPA:

    C = V V^T
    Q_r = eigenvectors of C for its r largest algebraic eigenvalues
    M = Q_r (Q_r^T V)

Use torch.linalg.eigh on a numerically symmetrized row Gram; no D-by-D feature covariance, full d-by-d projector, or autograd. In real arithmetic this is the Frobenius best-rank projection. Record finite-precision/Gram conditioning limitations; do not call it an exact FP32 SVD. Scientific projection uses float32, TF32 disabled, with explicit --projection-device cpu|cuda (future command below requests cuda; all current tests CPU). No silent backend/dtype fallback. Near-ties may make basis choice nonunique; output M, not eigenvector signs, defines the projection. The original CPA must still use its own torch.linalg.eig; external `eigh` is not canonical CPA.

Candidate V and exact code/cell checks remain CPU FP32. If rank work is on GPU, return M as CPU FP32 before the existing cell projection/check; the GPU computation is not allowed to redefine codec endpoints. Build the 15-code bounds table once with existing get_cells and, when needed, expand/project/check in row/flat chunks using the exact existing utility. Avoid storing several needless full [4096,25088] endpoint tensors. Each FP32 full matrix is 392 MiB by shape; actual workspace/peak VRAM is unmeasured. Synchronize only for honest timing; release preprocessing GPU arrays before original CPA/FIA.

## 5. Files and integration boundary

Create these NEW files (minor new helper files under the same task-specific prefix are allowed when justified):

- quantization_aware_cpa/qacpa/signed_cell_rank.py: projection core, chunked exact-code/cell validation, preprocessing diagnostics; no model/private-data dependencies.
- quantization_aware_cpa/qacpa/signed_cell_rank_runner.py: explicit `preprocess` and `attack` CLI subcommands. Attack accepts only public observation, finalized preprocessed artifact, checkpoint, and output. Import/copy the existing immutable CPA_CONFIG/FIA_CONFIG/SEED, reuse upstream_naive.run_upstream_cpa/fia and compatible telemetry/config helpers unchanged. Do not mutate those modules or global method implementations to change algorithm behavior.
- quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py: separately invoked offline comparison of existing naive and new cell_rank final candidates.
- quantization_aware_cpa/tests/test_signed_cell_rank.py and test_signed_cell_rank_pipeline.py.
- quantization_aware_cpa/docs/signed_cell_rank_comparison.md: method, limits, schemas, exact commands, implementation validation, research-not-run status.

The old `_run_arm` labels conversion as raw_Delta_or_raw_sq; do not reuse that misleading registration for preprocessed V. Prefer a thin NEW runner using the existing CPA/FIA APIs and constants with a truthful input conversion tag and distinct arm `INT4_cell_rank`. The old evaluator hardcodes naive/reference identities and directory roots; do not loosen it. Implement new strict provenance checks for naive versus cell_rank and reuse its lower-level evaluation/matching helpers without changing the existing validator.

Protected: official_cpa/**; existing upstream_naive.py/seed.py; signed_baseline*.py; quantization.py/cell_utils.py; existing FIA/replay/evaluation code; all preexisting results/observations/checkpoints. Record relevant source hashes before/after. Do not rewrite existing tests or files simply to share a few lines. If a protected-file edit is truly necessary, stop that part and explain a concrete alternative; do not silently expand the boundary.

## 6. Outputs and comparison contracts

Future NEW public root: quantization_aware_cpa/results/signed_cell_rank_B16_T10/
- preprocess_ap5/: registered config, public manifest and six packet hashes, q/s shape/type/scope, immutable reference to q/s observation, Y_q tensor hash, final V.pt with tensor/file hashes, diagnostics.jsonl, preprocessing_result.json.
- cell_rank_ap5/: registered arm, private_inputs_used=false, original CPA/FIA settings/source hashes, per-arm independent CPA cache, initialization hashes, sources.pt (S_hat and unchanged abs target_z), candidate.pt, result.json, raw_metrics.json, stage runtime and peak CUDA allocated/reserved.

Store raw preprocessing diagnostics for t=0 and every completed step: movement from V0 and previous V (relative L2/maxabs/changed-coordinate count), pair residual ||V_t-M_t|| and ||V_(t+1)-M_t|| with explicit definitions, fixed-s exact-code ratio/mismatch count, bounded-cell satisfaction, candidate scale and error to observed s, matrix norms, rank target/backend/dtype/device, nonfinite status and runtime. Record zeros in denominators explicitly; serialize strict JSON with null+reason, not NaN. Approximate projection residual need not be monotonically decreasing in finite precision; report violations instead of hiding them or tuning.

No gradient norm is applicable to this no-autograd alternating projection: log NOT_APPLICABLE rather than claim a zero gradient is a passed optimization test. After CPA, reuse raw CPA objective/L1/NV/source-diversity metrics. MI and actual negentropy without an implemented estimator remain null+reason; overwritten NE objective is not an estimated negentropy. Code/scale metrics describe the inferred update input, not a reconstructed-image local update.

Do not gate progress or declare a no-op using only principal angles. Claude Turn46's claim that equal top16 subspace implies identical CPA behavior is too broad: actual whitened values/projected mean may differ. The first implementation needs no separate principal-angle experiment. Log the actual input/source artifacts; final reconstruction comparison decides efficacy.

Future NEW private output: quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5/
Offline evaluator must check before loading private tensors: both final arms complete; candidate/source file hashes; exact public observation/checkpoint match; distinct expected arm identity; matching registered AND executed CPA/FIA settings and seeds; original source hashes; own CPA cache containment and actual input fingerprint; consistent source abs adapter; identical initial FIA tensor hash and actual numeric runtime flags. Check resolved path separation and reject overlap/aliases to input/result directories. Never overwrite an existing output or silently rerun/replace naive.

Evaluation uses ALL16 fixed final candidates. Match each arm once by pixel-MSE Hungarian and use those same image pairs for PSNR/SSIM/LPIPS. Embedding-oracle Hungarian cosine is a separate offline analysis of abs(S) versus true z; no reordering/scaling before FIA. Labels are never consumed. True-G relative error is evaluation-only and binds to the same sender observation. Report per-image metrics, means, new-minus-naive deltas, and an all16 original/naive/new comparison figure. No cherry-picking, composite selector, or success threshold invented after results. Mixed image-metric changes are reported as mixed. Candidate/file selection is final-only, not an oracle-best search.

## 7. Required current validation: small CPU only

Use C:/miniconda/envs/DLG/python.exe. Authorized tests use tiny synthetic matrices/packets (e.g. <=32x64), temporary files and mocked full pipelines. No real B16 packet decompression/preprocessing, full VGG forward/backward, private tensor loads, GPU test, or LPIPS download/run on real images. Existing completed JSON/log and source reads are allowed.

Meaningful tests:
1. Core rank projector on small matrices agrees with an independent float64 direct SVD reference within a stated numerical tolerance (compare reconstructed matrices/residual, not sign-ambiguous eigenvectors). Confirm uncentered handling with nonzero means.
2. AP0 returns original sq unchanged. Already feasible low-rank and zero fixtures demonstrate legitimate fixed points. A constructed off-rank fixture demonstrates the implemented projection can move data; do not assert every input must move.
3. At every saved AP iterate verify exact observed-code ownership and bounded cells using original utility, including +7/-7/0, nearest-even boundary neighbors, finite edge cap and zero codec convention. Scale mismatch is allowed and reported without a hidden rescale/anchor.
4. Distinguish M and V feasibility, test rank/shape/dtype/nonfinite errors, no mutation of original q,s/Yq, and deterministic repeated CPU output for a nondegenerate fixture. Do not promise cross-device bit reproducibility from this.
5. Integration spies verify the new runner uses unchanged original CPA/FIA with identical constants/seed/abs/order, unique cache and truthful input provenance; no training/replay/private input access. Mock full model/optimizer calls.
6. Offline negative tests reject wrong/mutated inputs, incomplete arms, mismatched settings/init/codec, same arm passed twice, private/public output overlap, changed hashes and private loading before finalization. Use tiny/mock files and metric functions; no new experimental image-quality measurements.
7. Appropriate existing focused regression checks for cell_utils and original-adapter/provenance contracts must pass without real-data/GPU execution. Inspect tests before selection. Existing original-CPA tests on very small synthetic CPU data are allowed; full research runs are not.

Suggested commands (Executor may narrow by documented test names to honor CPU/small scope):

    C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py quantized_update_cpa/tests/test_cell_utils.py -q
    C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_upstream_naive.py quantization_aware_cpa/tests/test_signed_baseline_provenance.py -q

No background research process, long model-monitoring turn, or polling. Tests may run synchronously; report actual command/count/runtime/failures, and don't count overlapping suites as unique tests. After passing relevant checks, finish the report; do not broaden to experiments.

## 8. Future commands to IMPLEMENT/DOCUMENT, NOT RUN in this task

Run from repo root after a later explicit research-execution request; entry-point naming below is the intended interface. If a reasonable interface adjustment is needed, record the actual exact replacement commands in the report.

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_runner preprocess --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5 --ap-iterations 5 --projection-device cuda

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_runner attack --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --preprocessed quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5 --checkpoint "C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth" --output quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5

    C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_evaluation --naive quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive --cell-rank quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --private-observation quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5

Existing naive is reused only after provenance/settings/init checks. Do not automatically recollect, rerun naive/Reference, add seeds/arms, or vary budgets if checks fail. No experimental launch script runs at import, pytest collection, or doc generation.

## 9. Completion report

Write .ai/execution-results/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914/report.md and results.json with FACT/INFERENCE/PROPOSAL/UNKNOWN, changed files, implemented equations/semantics, tests and scope, protected hashes, actual commands, limitations, and full experiment status NOT_RUN. Mark the handoff COMPLETED only when the authorized implementation/tests/docs are complete; otherwise FAILED/NEEDS_REVIEW with actual missing items. Completion means implementation complete, not reconstruction improvement. Preserve this plan while updating claim/status fields.

The following remain UNKNOWN until separately authorized full testing: real convergence/motion, scale drift, CPA/FIA effect, relative image quality, actual full-size runtime/memory, cross-device repeatability. No scientific performance guarantee follows from unit tests or peer consensus.
