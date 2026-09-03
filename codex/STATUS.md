# STATUS

Active milestone: **2 partial** (A_oracle only; bplus/A/Stage 3 not started)

| Milestone | State | Note |
|---|---|---|
| -1 | DEFERRED | CPA ICA replacement remains pending; do not start before this MP truncation report. |
| 0 | PASSING | Static gates and batch gates run before Milestone 1 rows. |
| 1 | REPORT READY | `v3_mptrunc` completed on 3 debug batches. |
| 1b | FOLDED | MP `B_eff` selection is now part of Milestone 1 before whitening. |
| 2 | PARTIAL | Stage-2 hinge solver implemented and `A_oracle` run for 5 batches. `bplus` not built. |
| 3 | NOT STARTED | |
| 4 | NOT STARTED | `c_mp` is an ablation axis after earlier milestones pass. |

## Current Task

- Stage 1 now determines `B_eff` using `lam_deb > c_mp * mp_edge` before whitening.
- Report both `recovered_over_beff` and `recovered_over_B`.
- Log full pre-truncation eigenvalue ratios, discarded PCA components, per-source `||a_i||`, loss, and labels.
- Diagnostic `algorithm=deflation` without truncation was saved as `run_tag=diag_deflation`.
- Mandatory Milestone 1 rerun was saved as `run_tag=v3_mptrunc`.

## Latest Results

- `v3_mptrunc`: `B_eff=[7, 6, 7]`, recovered over `B_eff=[6/7, 6/6, 7/7]`, recovered over `B=[6/8, 6/8, 7/8]`.
- `diag_deflation`: no truncation, recovered over `B=[6/8, 4/8, 7/8]`, `fastica_n_iter=[11, 12, 8]`.
- CUDA check: `NVIDIA GeForce RTX 5060 Ti`, capability `(12, 0)`.
- `v3_cmp_sweep`: `c_mp=1.0` keeps all 8 and gives `0/8` in all three batches. `c_mp>=3` gives `6/6, 6/6, 7/7` except `c_mp=12` gives `6/6, 6/6, 6/6`.
- `v3_feas_decomp`: A_true full feasibility is `1.0` for all three batches; true-kept vs A_hat gaps are `[0.508, 0.400, 0.562]`.
- `v3_A_oracle`: recovered over B for 5 batches is `[7/8, 6/8, 7/8, 6/8, 5/8]`. Hinge-only Stage 2 does not restore the low-gradient sources from pinv initialization.

## Void Runs

- Initial rows produced before the whitening/debias fixes are tagged `run_tag=void_prewhitening_bug` and must not be mixed into paper numbers.
