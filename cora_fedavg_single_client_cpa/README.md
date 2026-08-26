# CoRA FedAvg-Context Single-Client CPA

This runner attacks one compressed client update sampled from a sparse FedAvg
round.  The attacker observes

```text
Y_k^t = S_r(theta_k^{t,local} - theta^t)
```

not the round aggregate.  FedAvg is used to produce the realistic round model
`theta^t`; the selected client's compressed update is then passed to the same
CoRA candidate generation, CPA, Direct FIA, and consistency ranking path used by
the single-client runner.

Default observation policy:

- each keep ratio builds its own sparse FedAvg trajectory;
- each round aggregates `sum_k alpha_k S_r(delta theta_k^t)`;
- at `--observe_round` the runner saves the model checkpoint before local
  training and observes only `--target_client_id`;
- dropped attack-FC coordinates are desparsified per member;
- ranking recomputes the selected client update from reconstructed images using
  the saved `theta^t` checkpoint.

Example:

```bat
python cora_fedavg_single_client_cpa\run_cora_fedavg_single_client_cpa.py --ds imagenet --model vgg16 --num_clients 3 --rounds 20 --observe_round 20 --samples_per_client 24 --target_client_id 0 --local_epochs 1 --keep_ratios 0.8 --ensemble_size 8 --noise_seed 1234 --desparse_sampler retained_rescaled --run_attack --attack_n_iter 25000 --attack_n_log 5000 --n_iter_fi 25000 --n_log_fi 5000 --n_sample_fi 24 --fia_micro_batch_size 24 --reuse_existing
```
