# Client-Wise Bin-Constrained QICA

This folder contains the corrected int4 FedAvg QICA path:

1. collect final-round client deltas,
2. quantize each client delta to signed symmetric int4 immediately,
3. load client-wise `q_k`, `s_k`, and FedAvg weights,
4. optimize client-wise bin-constrained QICA,
5. run FIA from recovered `final_embeddings.npy` with q-bin consistency.

The method does not re-quantize an aggregate FP32 update. Q-on ablations train
`X_client_hat` directly and project it back into finite max-based bins after
each optimizer step. Q-off ablations keep `X_client_hat = q*s` fixed and train
only `W_hat`.

`end_to_end_qaware_fia.py` runs the requested QICA-conditioned path:
`final_embeddings.npy -> FIA image -> q consistency`. It optimizes pixels so
the model feature `z(x)` matches the recovered QICA embedding, then adds an
attack-layer int4 bin loss on `A_k @ z_k(x)`. The mixing matrix is initialized
from the recovered embedding by least squares and is frozen by default, so the
q-consistency gradient flows through `z(x)` back to pixels. Use
`--train_mixing` only when you also want to adapt `A_k`.

Useful checks:

- default diagnostics write `diagnostics.json` with client/sample order,
  `A_k` column shape/order, sample-axis permutation/sign/scale inverse checks,
  and an `L_q -> pixel` gradient probe.
- `--embedding_alignment oracle_context` aligns QICA rows to `context['z']` by
  Hungarian abs-cosine plus least-squares sign/scale. This is an oracle
  diagnostic or upper-bound setting, not a blind attack claim.
- `--compare_fia_only` runs a same-init FIA-only baseline (`q_weight=0`) and
  writes `comparison_fia_only_vs_qaware.json` against the frozen-A q-aware run.
- Final image metrics are written to `fia_metrics.json` and
  `image_metrics_by_sample.csv`: MSE, PSNR, SSIM, and LPIPS. Use `--skip_ssim`
  or `--skip_lpips` when the optional torchmetrics pieces are unavailable.

Smoke:

```powershell
python "int 4 fedavg/qaware_client_qica/sanity_checks.py"
python "int 4 fedavg/qaware_client_qica/run_qaware_ica.py" --run_dir "int 4 fedavg/imagenet/vgg16/clients3_rounds20_spc8_seed42" --qica_ablation qon_neoff --qica_n_iter 2 --qica_n_log 1 --qica_columns_per_step 64 --qica_trials 2 --n_components 24 --device cpu --fresh_start
python "int 4 fedavg/qaware_client_qica/end_to_end_qaware_fia.py" --run_dir "int 4 fedavg/imagenet/vgg16/clients3_rounds20_spc8_seed42" --embedding_path "int 4 fedavg/imagenet/vgg16/clients3_rounds20_spc8_seed42/qaware_client_qica/qon_neoff/final_embeddings.npy" --client_ids 0 --loss_scope attack_layer --n_iter 2 --n_log 1 --device cpu --fresh_start
python "int 4 fedavg/qaware_client_qica/end_to_end_qaware_fia.py" --run_dir "int 4 fedavg/imagenet/vgg16/clients3_rounds20_spc8_seed42" --embedding_path "int 4 fedavg/imagenet/vgg16/clients3_rounds20_spc8_seed42/qaware_client_qica/qon_neoff/final_embeddings.npy" --client_ids 0 --loss_scope attack_layer --embedding_alignment oracle_context --compare_fia_only --n_iter 2 --n_log 1 --device cpu --fresh_start
python "int 4 fedavg/qaware_client_qica/eval_qaware_fia_metrics.py" --output_dir "int 4 fedavg/imagenet/vgg16/clients3_rounds20_spc8_seed42/qaware_client_qica/qica_embedding_qaware_fia_attack_layer_oracle_context_n24_it1000" --device cuda
```

Collector:

```powershell
python "int 4 fedavg/qaware_client_qica/collect_final_round_client_qs.py" --num_clients 3 --rounds 20 --samples_per_client 8 --seed 42 --store_scope all
```
