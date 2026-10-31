# CPA_SIGNED_IMPROVEMENT_CODE_REVIEW_20260914

## Review record

- **Task / plan:** `CPA_SIGNED_IMPROVEMENT_CODE_REVIEW_20260914`, plan version `1`.
- **Executor identity:** existing thread `01a09e4d-51f3-7121-84e2-03a7a9ac3039`.
- **Runtime verified:** the current thread's latest `turn_context` records `gpt-5.6-luna` with `xhigh` reasoning effort. This is the user-approved model exception for this one read-only review.
- **Handoff verification before claim:** task ID and plan matched; handoff was `READY`, unclaimed; pre-claim handoff SHA256 was `b1ede028689e09be8404c9c3fb843b3483b1911991689049bec5423a5c1c011c`.
- **Claim:** the READY review was claimed by this thread. The handoff was changed only for claim/status bookkeeping; the plan, fixed constraints, research code, and existing results were left unchanged.
- **Scope:** static code and mathematics review of the four requested areas and the legacy ICA caveat. No tests, training, reconstruction, CPA/FIA run, new metrics, matrix diagnostics, or data changes were performed.

The signed baseline remains the reference. Its fixed conditions are Conv frozen, all six classifier tensors trained, one B16 batch, ten SGD steps at `0.001`, zero momentum and weight decay, deterministic signed nearest-even INT4, and the original CPA/whitening/loss/optimizer path. This review does not promote a proposal to the baseline.

## 1. External sign/ReLU FIA target adapter — **REVISE**

### Fact

The current signed arm sends the observation through the original upstream CPA and then sets the FIA target to `S.abs()` (`quantization_aware_cpa/qacpa/signed_baseline.py:241-275`, especially `260-275`). The registration record explicitly names `abs(S_hat)` as the source adapter and says there is no oracle correction (`signed_baseline.py:251-255`). The original upstream feature-inversion path also uses `rec_gi_reord.abs()` (`official_cpa/src/attack.py:108-120`). The Direct FIA objective compares the supplied target `Z` with the recovered model feature `z_hat` by cosine similarity and adds TV (`official_cpa/src/feature_inversion.py:15-40`, `133-198`). The adapter uses the checkpoint's VGG features and average pool, with the classifier omitted (`quantization_aware_cpa/qacpa/upstream_naive.py:97-118`, `267-308`).

The proposed rule

```text
sigma_i = argmin_{sigma in {-1,+1}} ||ReLU(-sigma*S_i)||_2
target_i = ReLU(sigma_i*S_i)
```

uses only the recovered row `S_i`; it does not require labels, images, a ground-truth permutation, or a changed CPA objective. The proposal correctly records that the existing `abs` operation is inherited upstream and is not, by itself, a code defect. The proposal also records reference/naive minority-sign L2 means of `1.0568745` and `23.4559631`; those are prior-result facts, not values recomputed in this review.

### Inference

The rule is a plausible public-only FIA-input ablation. It preserves row order and changes only the target passed to Direct. It is not a proof that a recovered signed CPA row is a true VGG feature row. It removes one sign half of every row, and the selected sign can differ under a tie. For an all-zero row both signs have zero penalty and the target remains zero. `signed_baseline.py` does not itself reject a zero-norm target before Direct (`261-275`), while the separate deterministic pilot does reject zero CPA targets (`run_deterministic_hybrid.py:65-69`). Thus a silent zero/tie policy could create a weak or undefined feature signal even though cosine uses an epsilon.

The large naive minority-sign value is evidence that `abs` can materially alter a row; it is not evidence that the majority sign is the correct physical sign. The proposal therefore cannot claim an improvement from the formula alone.

### Proposal

Keep the existing `abs` arm unchanged and, in a separately authorized future FIA-only arm, apply the sign/ReLU transform after CPA. Predeclare a deterministic tie rule and an explicit all-zero-row action (reject/flag the row or the candidate rather than silently treating it as a useful feature). Record the selected signs, positive/negative row norms, zero-norm and zero-fraction counts, target norms, and source-to-target diagnostics without using any ground-truth selector. Keep CPA outputs and source ordering immutable; do not use evaluation metrics to choose a sign.

### Unknown

Whether this target improves image recovery, preserves source diversity, or is more accurate than `abs` is unknown. It requires an authorized isolated FIA comparison; no such experiment was run here.

### Limitations

This review cannot establish the sign of the true latent feature for any row. The code evidence establishes the current adapter and objective, not the efficacy of a new adapter.

## 2. Cell-constrained FC1 weight/bias factorization — **REVISE**

### Fact

The collector freezes the Conv feature extractor, computes one fixed feature batch `z`, then trains the classifier for ten ordinary SGD steps (`quantization_aware_cpa/qacpa/signed_baseline.py:119-155`). All six classifier parameters are checked against an independent replay and must be `torch.equal` (`signed_baseline.py:145-170`); their shapes are recorded in the public wire context (`signed_baseline_observation.py:21-33`). The replay contract uses mean cross-entropy, a repeated full batch, fresh SGD, and all classifier parameters (`quantization_aware_cpa/qacpa/fc_replay.py:45-70`).

The existing UIA diagnostic makes the first-layer factor explicit: it accumulates the transposed hidden gradients in `B_cumulative`, forms `Y_factor = B_cumulative.matmul(Z)`, and compares it with the observed first-layer delta divided by the learning rate (`quantization_aware_cpa/qacpa/uia.py:183-217`, `221-255`). With a fixed `Z` of shape `[16,25088]`, an `A` that absorbs the negative learning-rate convention and the sum over ten steps has

```text
DeltaW1 = A Z,       Deltab1 = A 1_16,
 A: [4096,16].
```

The sign convention must be stated: the signed baseline stores `theta_after10 - theta_start` (`signed_baseline.py:148-155`), whereas the UIA diagnostic uses `(W0 - W_after)/lr` (`uia.py:237-240`). The same algebra applies after absorbing that sign into `A`.

The observed INT4 cells are fixed by the transmitted code and scale. `cell_utils.get_cells` constructs bounded FP32 endpoints for each fixed code/scale (`quantized_update_cpa/cell_utils.py:117-161`), `project_to_cells` is only a coordinatewise clamp (`164-170`), and code agreement and recomputed-scale agreement are separate diagnostics (`173-199`).

### Inference

The rank bound `rank(AZ) <= 16` is valid in exact arithmetic when the same fixed feature batch is reused at every step. It is not an exact-rank certificate for the stored FP32 delta: subtraction and accumulation roundoff can leave a residual. The UIA code's explicit `Y_delta_over_lr - Y_factor` residual is the right kind of diagnostic (`uia.py:237-255`). Bias coupling is useful as a consistency constraint, but it does not identify `A`; it adds a vector relation and uses a different observed tensor/scale.

The proposed joint problem

```text
minimize ||V - A Z||_F^2
subject to V in the observed weight cells and Z >= 0,
```

does not by itself recover the actual VGG feature matrix. Bilinear scaling and factorization ambiguities remain, and an `A=0, Z=0` initialization kills the bilinear gradients; whether the full constrained point is stationary also depends on how `V` is parameterized and on the cell boundaries. A projected `V` can be code-valid while `AZ` has rank at most 16, so `rank(V)` and `rank(AZ)` must be reported separately. Conversely, a low-rank `AZ` does not ensure that its coordinates stay in the fixed observed cells. Cell membership also does not imply that the candidate's own abs-max scale equals the observed scale; the codec intentionally reports those checks separately.

### Proposal

Treat this as a future, isolated external dequantization proposal only. Anchor or initialize the factorization away from the bilinear zero point, specify whether `Z` is a free nonnegative matrix or is constrained by an actual frozen VGG feature map, and define the weight-only and weight-plus-bias objectives as separate ablations. Keep each tensor's observed scale fixed; do not add midpoint attraction or optimize the scale. Report, separately, objective residual, fixed-scale code mismatches, bounded-cell violations, candidate scale consistency, `rank(AZ)`, and any rank diagnostic for the final cell-constrained `V`. Do not call a cell-feasible solution an exact rank-16 update.

### Unknown

The code does not establish existence of a useful nonnegative `Z` inside the observed cells, uniqueness of `A,Z`, or a useful bias constraint. The conditioning, attainable residual, and reconstruction effect are unknown and require an authorized experiment.

### Limitations

The rank statement is conditional on fixed features and exact arithmetic. It does not validate a learned factorization, and this review did not solve or numerically diagnose the proposed optimization.

## 3. Simple centered rank-16 SVD — **REJECT as an improvement claim**

### Fact

The original CPA centers the observed rows before whitening (`official_cpa/src/gradient_inversion.py:67-83`, `109-123`). Its whitening computes the row covariance and retains `self.n_comp` eigen-directions (`113-123`); the signed wrapper requests `n_components=16` and explicitly selects the original upstream whitening with no debiasing, MP truncation, stage 2, or least-squares mixing matrix (`quantization_aware_cpa/qacpa/upstream_naive.py:204-238`). The attack then optimizes a `16 x 16` normalized unmixing matrix and restores the mean/whitening terms (`gradient_inversion.py:139-180`, `218-226`).

### Inference

For a centered FC1 observation with at most 16 row dimensions, a centered rank-16 SVD reconstruction retains the complete available row subspace. Adding the mean back therefore reproduces the input up to floating-point/SVD ordering error; it gives CPA no new observation. A full-rank ICA followed by its inverse has the same no-op risk. It could matter only if it actually changes the retained subspace, the component count, or the observation before CPA, each of which would be a new method condition.

### Proposal

Do not implement or report simple centered rank-16 SVD as an improvement under the signed baseline. It may be retained as a documented no-op control. Any future SVD proposal must state a changed retained subspace or rank and receive a separate authorization; it must preserve the canonical CPA whitening/loss for the baseline comparison.

### Unknown

Only a numerically different preprocessing operation could have an effect, and its effect is unknown. No SVD or ICA run was performed.

### Limitations

The no-op conclusion is algebraic and conditional on retaining all 16 centered components; it does not claim that every lower-rank or non-centered transform is a no-op.

## 4. Future hard-label whole-batch update validation — **NEEDS_TEST**

### Fact

The signed public arm accepts only the public packet/checkpoint and records that no private inputs or labels are used (`quantization_aware_cpa/qacpa/signed_baseline.py:294-314`). The collector's private labels are confined to sender-side training (`signed_baseline.py:119-140`). The replay API supports hard class indices or probability targets, mean cross-entropy, a repeated full batch, and every classifier parameter (`quantization_aware_cpa/qacpa/fc_replay.py:45-70`); the full replay contract requires an explicitly supplied batch order and fixed model buffers/mode (`quantization_aware_cpa/qacpa/full_update_replay.py:1-13`, `28-42`).

The existing deterministic hybrid pilot is a different affine observation experiment. It estimates public labels from shared model logits (`quantization_aware_cpa/qacpa/run_deterministic_hybrid.py:122-128`), optimizes soft probabilities in the hybrid branch (`180-190`), and only after optimization replays both soft and hard labels for a fit report (`235-241`). That post-hoc hard fit is useful scaffolding, but it is not a predeclared signed hard-label validation rule and does not justify reusing its results for the signed baseline. The signed packet reader validates all six public FC tensors and their immutable packet metadata (`quantization_aware_cpa/qacpa/signed_baseline_observation.py:120-156`).

### Inference

Because ten SGD steps share the evolving classifier state, the candidate update is a coupled B16 object. Selecting labels or candidates independently per image can produce a label vector and batch trajectory that were never jointly replayed. A valid future hard-label check must freeze one public-only label-estimation rule before optimization, replay the entire B16 in its declared order for all ten steps, cover all six FC tensors, and compare the candidate against the observed signed `q,s` using the exact signed codec. Code/cell agreement and candidate-scale agreement must remain separate; a weighted score or image-level oracle selector can conceal failures in one tensor.

### Proposal

Before any future run, publish the label-estimation rule, tie handling, batch ordering, and candidate-selection rule. For each candidate **batch**, run one joint hard-label replay with frozen Conv, the fixed ten-step SGD schedule, and all six classifier tensors. Re-encode each candidate with the observed per-tensor signed nearest-even scale convention, then report per-tensor code mismatches/cell violations and scale consistency separately. Never fill labels from ground truth, never select an image independently after seeing its fit, and do not combine soft/hard results from the older affine pilot with signed results.

### Unknown

No public label rule has been validated for the signed observation, and the relationship between a post-optimization hard label vector and the observed update is unknown. Whether joint hard replay can improve any reconstruction metric is unknown.

### Limitations

This is a validation specification, not a result. The present task forbids running it, and no success threshold or label estimator should be inferred from this review.

## Legacy ICA boundary

`quantized_update_cpa/quant_aware_ica.py` is explicitly a separate quantization-aware ICA objective. It re-whitens with `torch.linalg.eigh`, normalizes a learned `W_hat`, and can learn the observation with quantization and negentropy losses (`quant_aware_ica.py:75-88`, `102-136`, `154-196`). Reusing it as canonical CPA would change the whitening path and objective, violating the signed baseline's fixed original CPA conditions. Verdict: **REJECT for canonical reuse**; it can be a separately named diagnostic/ablation only under a new authorization and with existing baseline outputs preserved.

## Final review disposition

| Area | Disposition | Reason |
|---|---|---|
| Sign/ReLU FIA target | **REVISE** | Public-only idea is implementable, but tie/zero handling and efficacy gates are missing; keep `abs` baseline. |
| FC1 weight/bias factorization and cells | **REVISE** | Algebraic rank bound is conditional; optimization is underdetermined and cell, rank, and scale checks must be separated. |
| Centered rank-16 SVD | **REJECT** | With all 16 centered components retained it is algebraically a no-op; use only as a control. |
| Future hard-label whole-batch validation | **NEEDS_TEST** | Replay scaffolding exists, but a signed public label rule and jointly validated B16 protocol are not established. |
| Legacy quantization-aware ICA as canonical CPA | **REJECT** | It changes whitening/objective; isolate it as a separately authorized ablation if ever used. |

No proposal was accepted as a change to the baseline. Existing research code, fixed conditions, and completed results were preserved. This report is the only review artifact written for the assigned scope.
