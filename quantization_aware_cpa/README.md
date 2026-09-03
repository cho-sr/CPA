# Quantization-Aware CPA

Fresh implementation for the quantization consistency milestones in the pasted
spec. The default run is Stage 1 diagnostics; Stage 2 currently includes only
the `A_oracle` hinge solver path.

## Layout

- `qacpa/quantizer.py`: per-row affine quantizer plus cell checks.
- `qacpa/victim.py`: VGG16/ImageNet victim batch collection and caching.
- `qacpa/stage1.py`: covariance debiasing, PCA whitening, and FastICA.
- `qacpa/stage2.py`: hinge-only projected-gradient consistency solver.
- `qacpa/diagnostics.py`: `rho`, `gamma`, and condition-number diagnostics.
- `qacpa/cmp_sweep.py`: MP threshold sweep diagnostics.
- `qacpa/feas_decomp.py`: LP-only feasibility decomposition.
- `qacpa/run_stage2_oracle.py`: Stage 2 `A_oracle` baseline runner.
- `tests/`: Milestone 0 gates as tests.

## Run

From this folder:

```powershell
python -m qacpa.run --config configs/spike.yaml
python -m qacpa.cmp_sweep --config configs/spike.yaml
python -m qacpa.feas_decomp --config configs/spike.yaml
python -m qacpa.run_stage2_oracle --config configs/spike.yaml --n-batches 5
```

Expected first config:

```text
bits=4, B=8, rounding=stochastic, seed=0, n_batches=3
```

Rows are appended to `results/runs.jsonl`. Victim tensors are cached under
`cache/`.

## Milestone 0

```powershell
pytest
```

The VGG16/ImageNet transpose test is marked `integration` and runs only when
`QACPA_IMAGENET_ROOT` is set.

## Note On Stochastic Quantization

The pasted stochastic-rounding pseudocode can emit dequantized values whose
nearest cell does not contain the original value, which violates Gate 0. This
implementation uses a subtractive-dither observation for `rounding=stochastic`:
the returned `Y_hat` is always within `delta / 2` of `Y`, and the additive
observation error is independent uniform noise with variance `delta^2 / 12`.
That is the combination required by both the exact hinge constraint and the
Stage 1 debiasing term.
