# CoRA Single-Client Compressed Update CPA

This folder implements the paper-style setting:

```text
Delta theta_k = LocalTrain(theta, X_k, y_k) - theta
Y_k = S_r(Delta theta_k)
```

The attacker observes one client's compressed update `Y_k`, knows the model,
loss, optimizer settings, and retention ratio, but does not know the values
removed by compression.  CoRA keeps this sparse observation fixed, generates
randomized attack-FC completions, runs the existing CPA + Direct FIA runner for
each member, then ranks members by recompressing each recovered batch and
comparing it to the same observed compressed update.

Default local training is one logical local step:

```text
--local_epochs 1
--local_batch_size n_samples
```

Increase `--local_epochs` or reduce `--local_batch_size` for a few local steps.

## Tiny smoke

```powershell
python cora_single_client_update_cpa\run_cora_single_client_cpa.py `
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

## VGG16/ImageNet example

```powershell
python cora_single_client_update_cpa\run_cora_single_client_cpa.py `
  --ds imagenet `
  --model vgg16 `
  --n_samples 24 `
  --attack_trials 1 `
  --local_epochs 1 `
  --keep_ratios 0.8 `
  --ensemble_size 8 `
  --noise_seed 1234 `
  --desparse_sampler retained_rescaled `
  --run_attack `
  --attack_n_iter 25000 `
  --n_iter_fi 25000 `
  --n_sample_fi 24 `
  --fia_micro_batch_size 4
```

CMD one-line:

```bat
python cora_single_client_update_cpa\run_cora_single_client_cpa.py --ds imagenet --model vgg16 --n_samples 24 --attack_trials 1 --local_epochs 1 --keep_ratios 0.8 --ensemble_size 8 --noise_seed 1234 --desparse_sampler retained_rescaled --run_attack --attack_n_iter 25000 --n_iter_fi 25000 --n_sample_fi 24 --fia_micro_batch_size 4
```
