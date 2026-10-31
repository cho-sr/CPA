# Quantized FedAvg Update Observations

This folder collects real FL update observations where each client update is
quantized during FedAvg before server aggregation.  It is not a post-hoc
quantization of a stored FP32 observation.

Default behavior:

- same client data selection as the FedAvg collectors
- per-client, per-parameter symmetric quantization
- server aggregates dequantized client updates
- final CPA-compatible aggregate pickle is saved
- `--quant_bits 4` or `--quant_bits 8` runs fixed-bit experiments
- omitting `--quant_bits` keeps the older mixed 4/8bit random bit-selection mode
- `--observation_scope single_client` saves one selected client's final-round update instead
- final-round client q/s for the attack layer is saved under `client_qs/`

Fixed 4bit single-client observation:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\collect_mixed_bit_fedavg_aggregate_updates.py ^
  --num_clients 3 ^
  --rounds 20 ^
  --samples_per_client 8 ^
  --local_epochs 10 ^
  --seed 42 ^
  --quant_bits 4 ^
  --observation_scope single_client ^
  --observed_client_id 0 ^
  --fresh_start
```

Fixed 8bit single-client observation:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\collect_mixed_bit_fedavg_aggregate_updates.py ^
  --num_clients 3 ^
  --rounds 20 ^
  --samples_per_client 8 ^
  --local_epochs 10 ^
  --seed 42 ^
  --quant_bits 8 ^
  --observation_scope single_client ^
  --observed_client_id 0 ^
  --fresh_start
```

Run sparsification-style randomized dequantization on fixed 4bit:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\run_randomized_dequant_cpa_fia.py ^
  --update_file mixed_bit_fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42_int4_obsclient0\round_020_client000_int4_dequant_update_cpa.pickle ^
  --ensemble_size 8 ^
  --noise_seed 1234 ^
  --perturb_scope attack_layer ^
  --attack_n_batch 1 ^
  --attack_n_iter 25000 ^
  --n_iter_fi 25000 ^
  --n_sample_fi 8 ^
  --run_aggregate_fia ^
  --fresh_start
```

Run CoRA-style member ranking on fixed 4bit:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\run_randomized_dequant_cpa_fia.py ^
  --update_file mixed_bit_fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42_int4_obsclient0\round_020_client000_int4_dequant_update_cpa.pickle ^
  --ensemble_size 8 ^
  --noise_seed 1234 ^
  --perturb_scope attack_layer ^
  --attack_n_batch 1 ^
  --attack_n_iter 25000 ^
  --n_iter_fi 25000 ^
  --n_sample_fi 8 ^
  --run_member_fia ^
  --run_consistency_ranking
```

Run sparsification-style randomized dequantization on fixed 8bit:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\run_randomized_dequant_cpa_fia.py ^
  --update_file mixed_bit_fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42_int8_obsclient0\round_020_client000_int8_dequant_update_cpa.pickle ^
  --ensemble_size 8 ^
  --noise_seed 1234 ^
  --perturb_scope attack_layer ^
  --attack_n_batch 1 ^
  --attack_n_iter 25000 ^
  --n_iter_fi 25000 ^
  --n_sample_fi 8 ^
  --run_aggregate_fia ^
  --fresh_start
```

Run CoRA-style FIA screening on fixed 8bit:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\run_randomized_dequant_cpa_fia.py ^
  --update_file mixed_bit_fedavg\imagenet\vgg16\clients3_rounds20_spc8_seed42_int8_obsclient0\round_020_client000_int8_dequant_update_cpa.pickle ^
  --ensemble_size 8 ^
  --noise_seed 1234 ^
  --perturb_scope attack_layer ^
  --attack_n_batch 1 ^
  --attack_n_iter 25000 ^
  --attack_n_log 5000 ^
  --n_iter_fi 25000 ^
  --n_log_fi 5000 ^
  --n_sample_fi 8 ^
  --fia_micro_batch_size 4 ^
  --run_member_fia ^
  --run_consistency_ranking ^
  --fia_screen_iter 5000 ^
  --fia_screen_top_k 5 ^
  --fresh_start
```

For a quick smoke run:

```bat
C:\miniconda\envs\DLG\python.exe mixed_bit_fedavg\run_randomized_dequant_cpa_fia.py ^
  --update_file mixed_bit_fedavg\outputs_smoke\fixed4_single_client_obs\round_001_client000_int4_dequant_update_cpa.pickle ^
  --ensemble_size 2 ^
  --attack_n_batch 1 ^
  --attack_n_iter 1 ^
  --n_iter_fi 1 ^
  --n_sample_fi 1 ^
  --run_aggregate_fia ^
  --fresh_start
```
