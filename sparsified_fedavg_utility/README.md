# Sparsified FedAvg Utility

This folder measures validation loss and accuracy when FedAvg applies Global
Top-k sparsification to each client update before server aggregation.

Update path:

```text
delta_k = theta_k_local - theta_global
y_k = S_r(delta_k)
theta_global <- theta_global + sum_k alpha_k y_k
```

`keep=1.0` is the dense FedAvg baseline. `keep=0.8` and `keep=0.5`
reuse the same client split, optimizer settings, and initial VGG16 ImageNet
model, but sparsify the transmitted client update.

ImageNet/VGG16 command:

```powershell
python sparsified_fedavg_utility\run_sparse_fedavg_utility.py `
  --ds imagenet `
  --model vgg16 `
  --num_clients 3 `
  --rounds 20 `
  --samples_per_client 8 `
  --local_epochs 10 `
  --local_batch_size 8 `
  --keep_ratios 100 80 50 `
  --eval_split test `
  --eval_every 1
```

For a faster first pass, evaluate only the final round on a fixed subset:

```powershell
python sparsified_fedavg_utility\run_sparse_fedavg_utility.py `
  --ds imagenet `
  --model vgg16 `
  --num_clients 3 `
  --rounds 20 `
  --samples_per_client 8 `
  --local_epochs 10 `
  --local_batch_size 8 `
  --keep_ratios 100 80 50 `
  --eval_split test `
  --eval_max_samples 5000 `
  --eval_every 0
```

Outputs are written under:

```text
sparsified_fedavg_utility/outputs/<ds>/<model>/<run_tag>/
```

Each keep-ratio folder contains `metrics.csv`, `metrics.json`,
`summary.json`, and `manifest.json`. The parent folder contains
`comparison_summary.csv` and `comparison_summary.json`.
