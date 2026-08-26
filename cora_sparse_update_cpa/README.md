# CoRA Sparse Update CPA

This folder keeps CoRA experiments separate from the earlier sparse CPA runs.

CoRA here means:

1. Calibrated candidate generation
   - The sparse observation is fixed.
   - Observed attack-FC coordinates are copied exactly.
   - Dropped attack-FC coordinates are sampled with a calibrated prior.
   - Default: `retained_rescaled`, which beta-fits the retained attack-layer
     magnitude distribution after rescaling it into `[0, 1]`, maps samples into
     `[0, tau_k]`, and uses the retained sign balance.
   - `tau_k` comes from the client's global Top-k threshold.
   - `--desparse_sampler uniform_tau` is also available for the simpler
     `U(-tau_k, tau_k)` ablation.

2. Independent restarts
   - Each candidate update is passed to the existing CPA runner.
   - Each member also runs Direct FIA, so every member produces reconstructed images.

3. Consistency ranking
   - For every reconstructed image, the runner recomputes a local-SGD update.
   - It applies the same global Top-k sparsification.
   - It compares the recompressed attack-layer sparse update to the observed sparse client updates.
   - Hungarian matching is used between reconstructed images and observed sparse clients.
   - The selected member is the one with the lowest mean matched relative L2.

The ranking is blind by default: no true labels, true embeddings, or original
images are used.  The default label policy is `model_pred`; `zero` is available
for compatibility checks.

## Smoke Command

```powershell
python cora_sparse_update_cpa\run_cora_sparse_cpa.py `
  --ds tiny_imagenet `
  --model fc2 `
  --n_samples 4 `
  --attack_trials 1 `
  --local_epochs 1 `
  --keep_ratios 0.5 `
  --ensemble_size 2 `
  --attack_n_iter 2 `
  --n_iter_fi 0 `
  --run_attack `
  --skip_consistency_ranking `
  --fresh_start
```

For VGG16/ImageNet CoRA ranking, keep `--n_iter_fi > 0`; ranking recomputes
local updates from member reconstructions and can be much slower than the
previous medoid-mean ensemble.

## Full-Shape Example

```powershell
python cora_sparse_update_cpa\run_cora_sparse_cpa.py `
  --ds imagenet `
  --model vgg16 `
  --n_samples 64 `
  --attack_trials 3 `
  --local_epochs 10 `
  --keep_ratios 0.3 `
  --ensemble_size 8 `
  --noise_seed 1234 `
  --desparse_sampler retained_rescaled `
  --run_attack `
  --attack_n_iter 25000 `
  --n_iter_fi 25000 `
  --n_sample_fi 8 `
  --fia_micro_batch_size 1
```

Useful speed-limit switches for ranking experiments:

```powershell
--consistency_max_recon_per_batch 2 --consistency_max_clients 8
```
