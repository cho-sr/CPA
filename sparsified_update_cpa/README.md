# Naive Sparse CPA

This folder contains Experiment 1 for client-level sparsified observations:

```text
one-sample client delta_w_k
  -> full-model Global Top-k S_r(delta_w_k)
  -> uniform aggregate delta_obs = mean_k S_r(delta_w_k)
  -> attack FC observation Y_obs = [delta_obs]_attack_fc ~= A Z
  -> CPA -> Z_hat -> FIA -> X_hat
```

Sparse entries dropped by Global Top-k are stored as literal zero values in the
dense update tensors passed to CPA.  Per-client masks are not passed to CPA and
are summarized only as provenance statistics.

The `AZ` relation is used only for the attack FC layer matrix consumed by the
author CPA implementation, not for the full model update.

Default sweep:

```powershell
python sparsified_update_cpa/01_naive_sparse_cpa.py --keep_ratios 1.0 0.80 0.20 0.10 0.05 0.01
```

`keep=1.0` is the one-sample-per-client FP32 control for this client-level
pipeline. It is distinct from the older full-batch FP32 update control.

Run CPA/FIA for the same sweep:

```powershell
python sparsified_update_cpa/01_naive_sparse_cpa.py --run_attack
```

Randomized desparsification from a fixed client-level sparse observation:

```powershell
python sparsified_update_cpa/02_randomized_desparse_ensemble_cpa.py --keep_ratios 0.10 --ensemble_size 8 --run_attack --run_aggregate_fia
```

This keeps each observed `S_r(delta_w_k)` fixed, samples only dropped
attack-layer coordinates from `U(-tau_k, tau_k)`, runs CPA per member, then
blind medoid-aligns and averages the recovered sources before the final Direct
FIA.

Fast contract checks:

```powershell
python sparsified_update_cpa/sanity_checks.py
```

Outputs are written under `sparsified_update_cpa/outputs_v1/`.  Attack result
directories include `oracle_evaluation.json`, `image_metrics_by_sample.csv`,
`metric_summary.csv`, and `whitening_pre_stats.json` when the artifact contains
attack-layer rank/spectrum diagnostics.
