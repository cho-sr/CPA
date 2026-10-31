# Latent-G extension design (proposal only)

This is a source-grounded design for a future, separately authorized implementation. It is not a completed optimizer run and it does not alter the signed CPA implementation or the repaired diagnostic.

## Facts from the current sources

`official_cpa/src/gradient_inversion.py:CocktailPartyAttack` creates a trainable square `W_hat`, initializes it to identity, row-normalizes it, and computes `S_hat = W_hat_norm @ X_w`. It then restores the centered mean with `W_hat_norm @ W_w @ X_mu` (lines 88-112, 144-166). Whitening is the current upstream covariance `X_zc @ X_zc.T / (d-1)`, `torch.linalg.eig`, top absolute eigenvalues and inverse square roots (lines 111-125). A full-rank square `W_hat_norm` followed by the corresponding inverse mixing is therefore algebraically a change of basis followed by its inverse; it does not create a low-rank source prior.

The active baseline configuration is CPA 25,000 iterations, Adam, lr 0.001, no scheduler, `decor=5.3`, `T=7.7`, `ne=1`, `nv=.13`, `l1=5`, `tv=.1`, seed 42. FIA remains 25,000 iterations, Adam lr .1, cosine scheduler, TV .1 and feature-inversion weight 1. In `step`, negentropy is calculated when `ne>0` but immediately overwritten by `loss_ne = torch.tensor(0.0)`, so its gradient is structurally zero. It must remain inactive in any baseline-preserving reproduction. Decorrelation is `mean(exp(T * abs(W_hat_norm @ W_hat_norm.T)) - 1)` and its direct gradient is only through normalized `W_hat`; it does not differentiate through reconstructed sources or through a fixed whitening matrix. With `nv>.0`, the TV branch is skipped (`tv` is reserved for a generator), while the active sign/sparsity prior is `minimum(||relu(-S_hat)||, ||relu(S_hat)||)` multiplied by `.13`. The active L1 term is `abs(S_hat).mean()` multiplied by 5; TV is inactive in this path because `nv` is nonzero. These statements follow the actual code, not an assumed ICA objective.

The original signed observation uses per-tensor FP32 scalar `s=max_abs/7`, signed integer `q` in `[-7,7]`, nearest-even rounding and the existing zero/cell rules. For a cell, a value at `q*s` with `q=+/-7` is an observed endpoint. A zero tensor has a zero scale and must stay in its registered zero cell. A finite `tanh(u)` is strictly inside `(-1,1)`, so `atanh(+/-1)` is not finite; a tanh parameterization cannot claim to attain an endpoint exactly. A bounded parameterization must either represent endpoint cells with an explicit discrete/active-bound state or use a safe interior approximation and report the approximation. It must not silently change `s` or add a scale loss.

## Proposed API and objective

Create a separate module with an API such as `optimize_latent_g(observation, model, *, whitening="fixed", config, output)`. It accepts the already validated public `(q,s)` packet and produces a candidate `G` plus an audit record. It must preserve the baseline artifact format and leave `signed_baseline.py`, CPA/FIA, whitening and old results untouched. The default proposal is a latent variable `G` in the same shape as the original flattened update and a deterministic decoder

```
G_centered = G - mean(G, dim=-1, keepdim=True)
S(G) = W_norm @ K @ G_centered + X_mu
```

where `K` is the exact registered `W_w` from the upstream whitening pass and `X_mu` is the exact observed mean. If an implementation stores the uncentered variable, it must use the equivalent mean-restoration equation from the source; no second centering or extra whitening is allowed. `W_norm` is a fixed snapshot of the baseline normalized unmixing matrix for the first validation. A recomputed `K(G)` is a separate ablation only.

The smallest defensible objective is a reconstruction/data term that is explicitly chosen in a future scientific decision, plus the existing active priors evaluated on decoded `S(G)`:

```
L = L_data(S(G), q, s) + 5 * mean(abs(S(G)))
    + .13 * mean(min(||relu(-S(G))||_2, ||relu(S(G))||_2))
```

The baseline decorrelation term may be included as a separately reported ablation on `W_norm`, but it is not a direct G-gradient and must not be described as source independence. The source currently has no active data likelihood that identifies a unique latent G from q,s alone. Therefore `L_data`, any sign prior, and any independent source prior are proposals requiring an explicit scientific choice; no selector, oracle z, private labels, true-G ranking, midpoint attraction or scale enforcement is allowed. The overwritten negentropy term remains labeled inactive unless the user authorizes a new experiment changing that source behavior.

## Gradients, degeneracy and whitening choices

With fixed `K`, `dS/dG = W_norm K (I - 11^T/d)`. If `W_norm K` is rank deficient, its nullspace contains latent directions with zero observation effect; optimization cannot identify those components. Even when full rank, sign symmetries and source permutations remain. `W` decorrelation constrains row geometry of `W_norm`, not the independence or noncollapse of `S(G)` after `K`; a collapsed or globally sign-flipped source can still satisfy that proxy.

The fixed-whitening default has a stable derivative, one cached `K`, and memory proportional to the 16x16 whitening matrix plus `G`; it is the only baseline-compatible first test. Recomputing covariance/eigendecomposition from every iterate introduces eigenvector sign/order ambiguity, repeated `O(B^2 d)` work, sensitivity at repeated or small eigenvalues, and additional graph memory. A future recomputed-whitening ablation must register its eigensolver, absolute-eigenvalue ordering, epsilon, gradients and memory separately and cannot overwrite canonical CPA results or be used as a silent fallback.

Cell constraints can be represented by an explicit projection after each optimizer step: round to the registered nearest-even cell, clip q to `[-7,7]`, and preserve zero cells. A continuous tanh surrogate can be used only for interior cells with a documented straight-through or active-bound choice; endpoint `q=+/-7` and zero-scale cells require explicit handling. The observed scale remains a metric-only value.

## Prospective bounded validation

First validate the decoder on synthetic matrices with known rank, nullspace, endpoint cells, all-zero cells and global sign reversal. Check finite-difference gradients against autograd for fixed K, and verify that mean restoration exactly matches the source equation. Then run a CPU mock with 16 sources and a tiny feature model to test provenance, cell projection, permutation/sign invariance, and deterministic serialization. A future real pilot should use one seed, one arm and a small iteration budget chosen in a new handoff; memory/runtime are UNKNOWN until the chosen `L_data` and G parameterization are fixed. Compare fixed versus recomputed K only as an explicitly registered ablation, with no oracle selection or private tensors. Full CPA/FIA and GPU costs are outside this design task.

**FACT:** the current code has active decorrelation, L1 and sign/sparsity terms, while negentropy and this path's TV contribution are structurally zero; original whitening restores the mean as shown above.

**INFERENCE:** fixed-K latent G has a transparent gradient and exposes a nullspace that a W-only decorrelation proxy cannot resolve.

**PROPOSAL:** implement the isolated decoder and synthetic validations first, then choose a data term and objective through a new scientific authorization.

**UNKNOWN:** q,s alone do not identify the true latent G, and no causal benefit, recovery guarantee, or runtime estimate can be claimed before the objective and parameterization are selected and measured.
