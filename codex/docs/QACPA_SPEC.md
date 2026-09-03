# Quantization-Aware Cocktail Party Attack — Implementation Spec (v3)

> Paste into Codex. English is deliberate: this is full of tensor shapes and
> identifiers, and ambiguity costs days.
>
> **v3 supersedes v1/v2.** It reflects the first Milestone-1 run, which found a
> Stage-1 whitening bug and an unsettled Milestone -1.
>
> **Read §0 and §1 before writing any code.**

---

## 0. Current state — what exists, what does not

Do not rebuild what is already verified. Do not assume anything else works.

| Component | State |
|---|---|
| Per-row quantizer, both rounding modes | **verified** (Gate 0 passed, bits 2/3/4/8) |
| Hook-based `A_true` / `S_true` extraction | **verified** |
| `Y == A_true @ S_true` vs autograd | **verified** |
| LP feasibility / `gamma` diagnostics | **works** |
| Stage 1 (debias + ICA) | **BUGGY — see §2** |
| Milestone -1 (CPA ICA port, fp32 repro) | **NOT DONE — see §1** |
| **Stage 2 solver** | **does not exist** |
| `bplus`, `naive`, `debias`, `A_oracle`, `gma` | **do not exist** |
| Stage 3 | does not exist (intentional) |

### The three existing rows in `results/runs.jsonl` are void

They were produced with the §2 bug. Every derived number (`rho`, `gamma`,
`source_cos`, `cond_A_hat`, infeasible-LP counts) is invalid. Tag them and never
mix them into paper numbers:
```
run_tag: "void_prewhitening_bug"
```
Add a required `run_tag` field to the schema. Paper runs use `run_tag: "v3"`.

### What the first run did establish (worth keeping in mind)

- Failure is **bimodal**, not graceful: 6/8 sources at cosine 0.995-0.999, 2/8
  at 0.001-0.017, in all three batches. The mean (0.75) hides this entirely.
- The two dead sources correspond to the two smallest covariance eigenvalues,
  and their recovered scales collapsed to ~0.002-0.03.
- Even the weakest eigenvalue sits ~25x above the quantization noise floor, so
  **the information is present**; the loss is algorithmic.
- 64-79% of sampled columns had an **empty** feasible set, i.e. `A_hat` is wrong
  enough to make the constraint system inconsistent.
- `cond(A_hat)` was 11-16 while `cond(A_true)` was 92-3131: the estimate cannot
  represent an ill-conditioned true mixing.

---

## 1. BLOCKING DECISION — settle Milestone -1 first

The first run used `sklearn.FastICA` (`fun=logcosh, max_iter=1000, tol=1e-4`).
That is **not** CPA's method. CPA optimizes a custom unmixing objective
(non-Gaussianity + independence penalty + TV prior), not off-the-shelf FastICA.
No upstream fp32 reproduction numbers exist yet either.

This must be settled **before** Milestone 2, because swapping the ICA later
invalidates `A_hat`, `S_hat`, every diagnostic, every Stage-2 result, and every
tuned hyperparameter.

**Required path:**

1. Port the upstream repo (`facebookresearch/cocktail_party_attack`) to
   py3.10+/torch>=2.7+cu128. Its `python=3.8` env cannot run on sm_120
   (RTX 5060 Ti, Blackwell). Verify
   `torch.cuda.get_device_capability() == (12, 0)`.
2. Read their ICA implementation. **Write down the exact objective and optimizer
   as implemented**, not as described in the paper.
3. Reproduce upstream CPA at **fp32, B=32**, unmodified. Compare `source_cos`
   and image metrics against the paper's reported values.
4. Replace `sklearn.FastICA` with CPA's ICA in our Stage 1, keeping our debiased
   whitening in front of it.

Do step 4 **after** the §2 fix, so "CPA ICA is better" and "whitening was broken"
do not get confounded.

If you believe keeping `sklearn.FastICA` is preferable, **stop and report the
argument** — do not decide unilaterally. A baseline weaker than published CPA
makes the entire comparison unfalsifiable.

---

## 2. Bug fixes to apply

### 2.1 Whitening (primary suspect for the two dead sources)

`stage1.warnings` contained:
```
"Ignoring n_components with whiten=False."
```
`whiten=False` means the ICA assumes its input is already white. If Stage 1 only
**projects** onto the top-`B` eigenvectors without rescaling, the input is not
white — the eigenvalue spread was 128:1 — and the lowest-variance directions are
lost. That is exactly the observed failure.

Correct Stage 1:
```
C = (Y_hat @ Y_hat.T) / d                      # (4096, 4096)
C = C - torch.diag(delta**2 / 12)              # DIAGONAL, only if stochastic
lam_deb, P = top_B_eigh(C)                     # lam_deb: (B,), P: (B, 4096)
sigma2  = (delta**2 / 12).mean()
mp_edge = sigma2 * (1 + sqrt(N / d))**2
keep    = lam_deb > c_mp * mp_edge             # c_mp default 1.5, config logged
lam_k   = lam_deb[keep]                        # (B_eff,)
P_k     = P[keep]                              # (B_eff, 4096)
Y_red   = (lam_k ** -0.5)[:, None] * (P_k @ Y_hat)   # (B_eff, 25088)
A_hat_red, S_hat = ICA(Y_red)
A_hat = P_k.T @ (lam_k[:, None] ** 0.5 * A_hat_red)  # lift back to (4096, B_eff)
```
The `lam ** -0.5` scaling is required, but absolute eigenvalue clamps are forbidden.
Directions below the Marchenko-Pastur noise-relative threshold are dropped
before whitening so whitening cannot amplify them into all retained coordinates.
Assert `cov(Y_red) ≈ I` in the `B_eff` subspace (off-diagonal < 1e-3) before
calling ICA.

### 2.2 Debias must be a diagonal, not a scalar

The logged `debias_variance` was a single scalar. Scales are **per-row**, so the
quantization noise covariance is `diag(delta_j^2 / 12)`, not
`mean(delta^2)/12 * I`. Row-to-row gradient magnitude varies a lot in this FC
layer, and the error falls hardest on the weak components — the ones that died.

### 2.3 The `bits16` gate was vacuous

It ran with `deterministic` rounding, where debias is disabled, so `debias` and
`naive` are identical by construction and `relative_l2 = 0.0` is automatic. It
tested nothing. **Re-run with `stochastic` only**, and assert
`relative_l2 < 1e-3` rather than `== 0`.

### 2.4 Verify the factorization gate is not self-comparing

`factorization_matches_autograd` reported `max_abs_err` of exactly `0.0`.
Comparing the hook product against autograd's `G` should leave float error
around `1e-7`, not exactly zero. Confirm the two sides are computed
independently; if not, fix the test.

### 2.5 Cache invalidation

Changing any Stage-1 parameter must invalidate cached `A_hat` / `S_hat`. Hash the
Stage-1 config into the cache key.

---

## 3. Exact shapes (unchanged — already verified)

Model: `torchvision.models.vgg16(weights=IMAGENET1K_V1)`, eval, frozen.
Attacked layer: `model.classifier[0]` = `Linear(25088, 4096)`.

```
Y = A @ S
Y : (N, d) = (4096, 25088)
A : (N, B) = (4096, B)      # backprop deltas, columns indexed by sample
S : (B, d) = (B, 25088)     # per-sample embeddings = ICA sources, S >= 0 exactly
N = 4096 mixtures,  d = 25088 ICA samples per source
```

Ground truth via hooks on `classifier[0]`: forward hook `input[0]` -> `S_true`;
full backward hook `grad_output[0]` -> `A_true.T`.

Large `B`: accumulate over microbatches — the aggregate gradient is a sum, so
this is exact. Rescale if the loss is a batch mean. Hooks must accumulate in the
same order.

**Batches must have all-distinct labels** for every gate run. Add
`data.require_distinct_labels: true`, assert it, and log the label list.
`cond(A_true) = 3131` in one of the void runs suggests this was not enforced;
duplicate labels correlate the columns of `A` and break ICA independently of
quantization.

---

## 4. Quantizer (verified — do not change)

Per-row bucketed uniform affine; one scale per row `j` of `Y`. A global scale is
degenerate at 4 bits and does not match how QSGD-style defenses bucket.

Invariant, holds for every entry:
```
(Y - Y_hat).abs() <= delta[:, None] / 2 + 1e-6
```

Both rounding modes must run and are **not** equivalent:
- `stochastic` — error unbiased, independent, uniform; the §2.2 debias is exact.
- `deterministic` — error correlates with the signal; debias becomes an
  approximation. **The Stage-2 hinge stays exactly valid in both cases.**

That asymmetry (our constraint is assumption-free; debiasing is not) is a
**result to report**, not a bug to hide.

---

## 5. Method

### Stage 1 — estimate A

As corrected in §2.1/§2.2, first debias the covariance diagonally, compute the
top requested `B` eigenvalues, determine `B_eff` with the MP threshold
`lam_deb > c_mp * mp_edge`, and whiten only the retained directions. Use CPA's
ICA (§1) in place of FastICA after this folded-in Milestone-1 fix.

Log `cond(A_hat)`, `cond(A_true)`, requested `B`, retained `B_eff`, `sigma2`,
`mp_edge`, `c_mp`, all pre-truncation `lam_deb` values, discarded component
indices, kept `lam/mp_edge` ratios, and the debias diagonal summary on every run.

### Stage 2 — consistency solve (BUILD THIS — it does not exist yet)

Column-separable and convex. For each column `k` of `S` independently:

```
minimize_{s in R^B}   sum_j [ max(0, |a_j . s - Y_hat[j,k]| - delta_j/2) ]^2
subject to            s >= 0
```

Batched projected-gradient implementation, chunked over `d`:
```python
# A_hat: (N, B)   Y_chunk: (N, C)   S_chunk: (B, C)  init = S_hat[:, chunk]
L   = 2.0 * torch.linalg.matrix_norm(A_hat, ord=2) ** 2      # Lipschitz const
eta = 1.0 / L
for it in range(max_iter):
    R = A_hat @ S_chunk - Y_chunk                            # (N, C)
    V = (R.abs() - delta[:, None] / 2).clamp_min(0) * R.sign()
    G = 2.0 * (A_hat.T @ V)                                  # (B, C)
    S_chunk = (S_chunk - eta * G).clamp_min(0)               # nonneg projection
```
- The `1/L` step size gives guaranteed convergence for this convex problem.
  Optional FISTA momentum; do not hand-tune `eta`.
- Chunk `C` so the `(4096, C)` residual fits comfortably; `C = 4096` is ~67MB.
  Whole stage should be minutes, not hours.
- Record per-column final hinge to compute the §7 metrics.
- **No prior `r(S)` yet.** Permitted only if Milestone 2 shows *high consistency
  with poor quality* (see §7 diagnosis table).

Feasibility note: the feasible set was **empty for 64-79% of columns** in the
void runs. The hinge is a min-violation formulation, so it still runs — but the
solution is then pulled toward a wrong `A_hat`. Hence `--refine-A` (below)
matters earlier than originally planned.

Optional flag `--refine-A`, **off by default**: an empty feasible set means
`A_hat` is wrong, so residual infeasibility is a usable objective. Alternate a
few `S`-steps with one small gradient step on `A_hat` against the same hinge.
Never jointly with Stage 3.

### Stage 3 — end-to-end (LOCKED — do not build)

Gated behind Milestone 3. `A_hat` frozen, warm-started from `S*`:
```
minimize_x  - sum_i CS(f(x_i), S_star[i])
            + lam_tv * sum_i TV(x_i)
            + lam_q  * sum_{j,k} hinge(|(A_hat @ f(x))[j,k] - Y_hat[j,k]|, delta_j/2)^2
```

---

## 6. Baselines — build in this order

Implement `A_oracle` **first**. It uses `A_true`, so it is independent of Stage-1
quality and needs only the Stage-2 solver. It is the method's upper bound: if it
fails, the other baselines are wasted work.

| order | id | description |
|---|---|---|
| 1 | `A_oracle` | Stage 2 given `A_true` — upper bound |
| 2 | `bplus` | decoupled relaxation (see below) |
| 3 | `A` | **ours**: corrected Stage 1 + Stage 2 |
| 4 | `debias` | Stage 1 with debias, no Stage 2 |
| 5 | `naive` | Stage 1 without debias, no Stage 2 |
| 6 | `gma` | upstream GMA, unmodified |

### `bplus` — and its expected degeneracy

Per-source Mahalanobis ball, no coupling:
```
sigma_i^2 = sum_j (delta_j^2 / 12) * w_ij^2      # w_i = row i of the unmixing matrix
bplus: minimize r(s)  s.t.  |s_i - S_hat[i,k]| <= c * sigma_i,  s >= 0
```

**Expect this to nearly collapse to `debias`.** Without a prior `r`, the ball is
centred at `S_hat`, so `S_hat` is already optimal and the only content left is
the nonnegativity clip — `bplus ≈ relu(S_hat)`.

That degeneracy is **not a bug to fix; it is the argument.** It shows the
decoupled relaxation carries no information on its own, and that coupling across
the batch is the sole source of gain. Report it explicitly.

**Do not give `bplus` coupling to make it "fairer."** If a prior is later added
to `A`, add the identical prior to `bplus`, with equal tuning budget and equal
iteration count.

---

## 7. Metrics

### Mandatory reporting changes from v2

1. **`source_cos` must be logged per-source, not only as a mean.** The mean hid a
   6/8-perfect + 2/8-dead structure in the first run. After MP truncation, log
   the full length-`B_eff` matched vector plus `n_sources_recovered` (count with
   `|cos| > 0.9`), `recovered_over_beff`, and the honest whole-batch
   `recovered_over_B`.
2. **`feasible_column_fraction` and `infeasible_column_fraction` are first-class
   metrics**, not byproducts of the `gamma` diagnostic. They are the most direct
   measure of `A_hat` quality.
3. **`gamma` may never be reported alone.** It is measured only over feasible
   columns, so it is optimistically biased. Always emit it next to
   `feasible_column_fraction`.
4. **Log per-source `||a_i||` and per-sample loss and label.** Test whether the
   dropped/unmatched sources are the small-gradient, low-loss, well-classified
   samples. If so, that is a reportable finding about *which* samples leak.
5. **Log MP truncation state.** Include `B_eff`, `sigma2`, `mp_edge`, `c_mp`,
   all requested `lam_deb/mp_edge` ratios, and discarded component indices.

### Constraint satisfaction
- `consistency_rate` — fraction of `(j,k)` inside their cell
- `residual_violation` — final mean hinge
- `feasible_column_fraction`, `infeasible_column_fraction`

### Recovery quality (Hungarian matching on `|cosine|`)
- `source_cos_per_source` (length `B_eff`), `source_cos` (mean over `B_eff`),
  `n_sources_recovered`, `recovered_over_beff`, `recovered_over_B`
- `scale_err` — mean `| ||s_hat_i|| / ||s_true_i|| - 1 |` (only `A` can report this)
- image metrics — Stage 3 only

Judge success in **embedding space** (`source_cos_per_source`), not image space.
ImageNet 224x224 feature inversion is the weakest link even upstream; image
metrics cannot separate "Stage 2 failed" from "FIA failed."

### Diagnostics
`rho`, `gamma` (+ feasible fraction), `mean_axis_extent`, `cond_A_hat`,
`cond_A_true`, all requested `B` eigenvalues, `B_eff`, `mp_edge`, debias diagonal
summary, `||a_i||` per source, per-source loss and label.
Compute LP diagnostics on 512 sampled columns; for `B > 64` use 32 random
directions instead of `2B` axis directions.

### Diagnosis table — use this instead of blind tuning

| consistency | quality | diagnosis | what to change |
|---|---|---|---|
| high | high | success | nothing |
| high | low | wrong point inside the cell | add `r(S)` / TV |
| low | low | `A_hat` wrong, or empty feasible set | Stage 1, `--refine-A` |
| low | high | **implementation bug** | re-run Gate 0 |

---

## 8. Milestones — STOP and report at each

Debug config: `B=8, bits=4, stochastic, distinct labels, seed=0`.
First **reportable** config: `B=32`.

### Milestone -1 — settle the ICA and reproduce fp32 (BLOCKING)
Per §1: port env, verify sm_120, document CPA's real ICA objective, reproduce
upstream fp32 at `B=32`, then swap CPA's ICA into our Stage 1.

**STOP. Report reproduction numbers, the ICA objective as implemented, and
`get_device_capability()`.**

### Milestone 0 — correctness (re-run)
1. Quantizer invariant, both modes, bits {2,3,4,8}. *(already passing)*
2. Oracle test: hinge at `(A_true, S_true)` is exactly 0. *(already passing)*
3. `Y == A_true @ S_true` vs autograd — re-verify per §2.4.
4. `bits16` debias-vs-naive — re-run with **stochastic** per §2.3.
5. **New:** `cov(Y_red) ≈ I` after whitening (off-diagonal < 1e-3).
6. **New:** batch labels are all distinct.

**STOP. Report all six.**

### Milestone 1 — re-run diagnostics after the §2 fixes
3 batches with mandatory MP truncation folded in. Report requested `B`,
retained `B_eff`, all 8 `lam_deb/mp_edge` ratios, `source_cos_per_source`
(length `B_eff`), `n_sources_recovered`, `recovered_over_beff`,
`recovered_over_B`, `rho`, `gamma`, `feasible_column_fraction`, `cond_A_hat`,
`cond_A_true`, `||a_i||`, loss, and label per source.

Decision:
- 8/8 recovered **and** feasible fraction high -> go straight to Milestone 2.
- 8/8 recovered but feasible fraction still low -> run `A_oracle` feasibility
  to isolate residual `A_hat` error.
- Still weak after MP truncation -> Stage 1 remains the blocker. Do not advance
  to Stage 2.

**STOP. Report.**

### Milestone 1b — `B_eff` truncation (MANDATORY, folded into Milestone 1)
Drop components below the MP threshold before whitening, then re-measure the
feasible fraction and report both `recovered_over_beff` and `recovered_over_B`.
If the dropped/unmatched samples are low-loss or small-gradient samples, report
that as the mechanism and keep `c_mp` as a Milestone-4 ablation axis.

**STOP. Report.**

### Milestone 2 — does the method work at all?
Build the Stage-2 solver, then run `A_oracle` vs `bplus` on 5-8 batches. The
effect size here is "recovered vs not," so 8 batches suffice; needing more means
it is not a result.

**If `A_oracle` does not clearly beat `bplus` on `source_cos_per_source` /
`n_sources_recovered`, stop. Do not build Stage 3.**

Then run `A` and report the gap `A_oracle - A` (the `A_hat` estimation loss).

**STOP. Report the 6-row table** (`gma`, `naive`, `debias`, `bplus`, `A`,
`A_oracle`) with per-source values, not just means.

### Milestone 3 — is Stage 3 needed?
Feed `S*` from Stage 2 into upstream **unmodified** feature inversion (no
consistency term). If that captures the whole gain, Stage 3 is unnecessary and
the paper is simpler. Build Stage 3 only if measurable headroom remains.

**STOP. Report.**

### Milestone 4 — grid (only after -1..3 pass)
- `bits`: {2, 3, 4, 6, 8, fp32}
- `B`: **{32, 64, 128, 256, 512, 1024}** — CPA's published VGG-16 range;
  off-grid batch sizes are not comparable to prior work
- `rounding`: {stochastic, deterministic}
- seeds: 5
- ablations: debias on/off; scalar-vs-diagonal debias; `--refine-A` on/off;
  `N` row-subsample {100%, 50%, 25%}; duplicate-label batches (sweeps `cond(A)`);
  `c_mp` MP truncation threshold

Stage-2 metrics over the full grid; Stage 3 only on `bits in {3,4} x B in {32,128}`.

---

## 9. Hard prohibitions

Violating any of these invalidates the experiment.

1. **Never use a straight-through estimator on `Q`.** STE collapses the
   consistency term into `||g(x) - Y_hat||^2`, which *is* naive dequantization.
   The hinge is the method. There is nothing to differentiate through `Q`.
2. **Do not treat the hinge dead zone as a bug.** Zero gradient inside the cell
   is correct — the observation carries no information there. The hinge is a
   barrier, not an attractor, so never remove the CS/prior term in Stage 3.
3. **Do not give `bplus` coupling.** Its degeneracy is the argument (§6).
4. **Do not modify CPA's ICA objective, FIA, metrics, or matching code.**
5. **Do not add priors, TV, or extra regularizers before Milestone 2.**
6. **Do not report `source_cos` as a mean only.** Per-source vector is mandatory.
7. **Do not report `gamma` without `feasible_column_fraction`.**
8. **Do not tune only `A`.** Equal budget and equal iteration count for all
   baselines.
9. **Do not use a global quantization scale**, and do not use a scalar debias.
10. **Do not report `sign_accuracy` as a contribution** — `S >= 0` already fixes
    sign on VGG16. Report `scale_err`.
11. **Do not pass the full (4096, 25088) matrix to sklearn.** Reduce to `B` dims
    first; full-SVD whitening in float64 will exceed 16GB system RAM.
12. **Do not skip Milestone -1 or Milestone 0.** A wrong `delta`, an unwhitened
    input, or a non-CPA ICA all produce plausible bad numbers, not a crash.

---

## 10. Engineering constraints (16GB VRAM / 16GB RAM)

| item | size | note |
|---|---|---|
| `Y` fp32 (4096x25088) | 411 MB | one copy, on GPU |
| `C` (4096x4096) | 67 MB | fine |
| VGG16 full gradient | ~550 MB | fine |
| Stage-2 chunk residual | ~67 MB | chunk over `d` |

- Config-driven single entry point:
  `python -m qacpa.run --config configs/spike.yaml`
- Fully seeded and deterministic given a seed; log the seed.
- **Required schema fields:** `run_tag`, `git_sha` (currently `"unavailable"` —
  fix this), `schema_version`.
- **Cache as `int8` q + `delta`**, never fp32 `Y_hat` (~822MB/config, ~25GB over
  the grid; int8 + scales is ~103MB). Recompute `Y_true` from the seed.
  Cache keys must hash the Stage-1 config (§2.5).
- One row per run appended to `results/runs.jsonl` with full config, every metric
  in §7, every diagnostic, wall time. Figures are generated later from this
  file — **do not build plotting now.**
- Log every metric **per victim batch**, never only the mean.

---

## 11. First deliverable

Milestone -1 only:
1. Env ported; `torch.cuda.get_device_capability()` output.
2. CPA's ICA objective and optimizer, as actually implemented upstream.
3. Upstream fp32 CPA reproduced at `B=32`, compared against the paper.

Nothing else. Report and wait.
