# CoRA Round-Update CPA

This folder keeps the original per-client sparse CoRA runner untouched and adds
an aggregate-round-observation variant.

Threat model:

- The attacker observes one sparse FedAvg round update:
  `Y_round = sum_k alpha_k S_r(delta_theta_k)`.
- Client-level masks, retained values, and thresholds are not available to the
  attack.
- Randomized desparsification is applied only to the aggregate attack-FC layer.
- Consistency ranking compares reconstructed candidates against the observed
  aggregate sparse round update, not against per-client sparse observations.

Why ranking changes:

- Per-client ranking would require `{Omega_k, Y_k, tau_k}` for each client.
  Those are not observable under this aggregate-only setting.
- Aggregate ranking is still valid, but it must recompute a sparse aggregate
  update from the reconstructed images and compare it to `Y_round`.
- Full aggregate ranking requires reconstructed images for the full attacked
  round batch. If only `--n_sample_fi 8` is reconstructed out of 64, the runner
  skips exact aggregate ranking unless `--allow_partial_aggregate_ranking` is
  explicitly set.

CMD example:

```bat
python cora_round_update_cpa\run_cora_round_update_cpa.py --ds imagenet --model vgg16 --num_clients 3 --samples_per_client 8 --rounds 20 --local_epochs 10 --local_batch_size 8 --keep_ratios 0.8 0.5 0.3 0.1 --ensemble_size 8 --noise_seed 1234 --desparse_sampler retained_rescaled --run_attack --attack_n_iter 25000 --n_iter_fi 25000 --n_sample_fi 24 --fia_micro_batch_size 4
```

For a cheaper first pass, add `--skip_consistency_ranking` or keep
`--n_sample_fi` small. Exact aggregate ranking needs `--n_sample_fi` equal to
`num_clients * samples_per_client`.
