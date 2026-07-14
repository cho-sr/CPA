# Quantized FedAvg Update CPA

This folder keeps the quantization-aware CPA extension separate from the
original `cocktail_party_attack` implementation.

The first three experiment stages are:

1. `01_fp32_update_cpa.py`
   - Collect FedAvg local model updates.
   - Save the original FP32 update signal `delta_w = w_local - w_global`.
   - Optionally run the original CPA/FIA pipeline on that update signal.

2. `02_naive_4bit_update_cpa.py`
   - Collect the same FedAvg local updates.
   - Apply symmetric signed 4-bit fake quantization and dequantization.
   - Save `delta_w_q = D(Q(delta_w))`.
   - Optionally run the original CPA/FIA pipeline on that quantized update signal.

3. `03_randomized_dequant_ensemble_cpa.py`
   - Collect the same naive 4-bit dequantized update signal.
   - Generate K randomized dequantization members with
     `delta_w_q + Uniform(-delta/2, delta/2)` per tensor.
   - Optionally run the original CPA/FIA pipeline once per ensemble member.
   - This is the first noise-aware ICA/CPA probe; sign/permutation alignment and
     source averaging are intended as the next analysis layer.

Both scripts save data in the same pickle schema expected by
`cocktail_party_attack/src/attack.py`: `x`, `y`, `z`, and `grad`.
Here, `grad` intentionally contains FedAvg update tensors, matching the
original attack code's `--fl_alg=fedavg` path.

Example:

```bash
python CPA/quantized_update_cpa/01_fp32_update_cpa.py \
  --ds tiny_imagenet --model vgg16 \
  --global_checkpoint fedavg_fp32/outputs_vgg16_parallel/fp32/global_round_020.pt \
  --n_samples 256 --n_rounds 10 --local_epochs 10 --local_batch_size 256

python CPA/quantized_update_cpa/02_naive_4bit_update_cpa.py \
  --ds tiny_imagenet --model vgg16 \
  --global_checkpoint fedavg_fp32/outputs_vgg16_parallel/fp32/global_round_020.pt \
  --n_samples 256 --n_rounds 10 --local_epochs 10 --local_batch_size 256 \
  --rounding nearest

python CPA/quantized_update_cpa/03_randomized_dequant_ensemble_cpa.py \
  --ds tiny_imagenet --model vgg16 \
  --global_checkpoint fedavg_fp32/outputs_vgg16_parallel/fp32/global_round_020.pt \
  --n_samples 256 --n_rounds 10 --local_epochs 10 --local_batch_size 256 \
  --rounding nearest --ensemble_size 8
```

Add `--run_attack` to immediately launch the original CPA attack using the
newly saved pickle file.

The wrappers force the CPA dataset module to read from `/root/CPA/datasets`
so they use the shared dataset copy at the repository root, not
`cocktail_party_attack/datasets`.
