# Quantization-Aware CPA

The current full Conv+FC, single-client 4-bit update implementation is described
in [Full-update FIA](docs/FULL_UPDATE_FIA.md), following the user decisions in
`.ai/STATE.md` at the repository root. It directly optimizes images without CPA
source separation. Its resource gate and short pilot are separate from the
historical experiments below.

The earlier milestone is the direct upstream FedAvg naive CPA baseline; see
[`STATUS.md`](../codex/STATUS.md) and the authoritative
[`QACPA_SPEC.md`](../codex/docs/QACPA_SPEC.md). This baseline includes CPA.
Historical raw `naive` keys also include the ported CPA path; the source-separation
ablation is explicitly called **no-CPA**.

## No-CPA control invalidation (2026-09-09)

The user rejected both no-CPA control conditions because variable control failed.
Their results are **invalid for scientific use**. The no-CPA contrasts, combined
IUT claim, normalized LPIPS retention, calibrated success thresholds/counts and
dependent figures are withdrawn. Completed execution markers do not establish
validity. Original data and locked protocols remain provenance only.

Read the [invalidation record](../codex/docs/NO_CPA_CONTROL_INVALIDATION_20260909.md)
and [`results/result_validity.json`](results/result_validity.json) before using
historical results. Dependent report generators reject this invalid input.
Do not rerun the archived no-CPA queue or restore its old report as current evidence.

The current graph compares only Cocktail Party Attack + feature inversion attack
on the same80 client updates: FP32 (`m6_uia_fp32_B16_fia`) versus the 4-bit
observation model (`m7_uia_4bit_B16`). These source CSVs are independent of no-CPA.
[Current PNG](results/client_update_fp32_vs_4bit/client_update_fp32_vs_4bit.png) ?
[Current ZIP](results/client_update_fp32_vs_4bit.zip).

## Historical completed successor: UIA fp32 n=16

After M7 exits with a complete report, the authorized successor is
`m6_uia_fp32_B16_fia`: five independent round-0 client updates, n=16, T=10,
uncompressed fp32, FC-only SGD 0.001, full 16 CPA sources and all 80 FIA images.
FIA keeps the existing 25,000 iterations, TV=0.1, Adam lr=0.1, cosine schedule,
and seed=42 settings.
The actual parameter-difference matrix is attacked; ideal factors remain
diagnostic only. Protocol differences from the published Table5 are recorded
before execution in
[`UIA_PUBLISHED_PROTOCOL.md`](../codex/docs/UIA_PUBLISHED_PROTOCOL.md).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/queue_m6_uia_after_m7.ps1
```

This hidden successor waits on the existing M7 process handle, checks its
completion markers and registered hash, then launches only the fp32 UIA runner.
It does not poll M7 metrics. A failed predecessor prevents the successor from
starting. Repeated submission detects an active queue or runner. Status and
logs are under `results/m6_uia_fp32_B16_fia/`; `queue_status.json` distinguishes
waiting, running, complete and failed. No Stage2, `--refine-A`, 4-bit or GMA
is included, even if the resulting LPIPS reaches the reference band.

## Layout

- `qacpa/quantizer.py`: per-row affine quantizer plus cell checks.
- `qacpa/victim.py`: VGG16/ImageNet victim batch collection and caching.
- `qacpa/stage1.py`: covariance debiasing, MP-truncated PCA whitening,
  CPA-style ICA/unmixing, and observed least-squares `A_hat` scaling.
- `qacpa/stage2.py`: hinge plus L1-prior projected-gradient consistency solver.
- `qacpa/bplus.py`: decoupled interval baseline with the same L1 prior budget.
- `qacpa/diagnostics.py`: `rho`, `gamma`, and condition-number diagnostics.
- `qacpa/cmp_sweep.py`: MP threshold sweep diagnostics.
- `qacpa/feas_decomp.py`: LP-only feasibility decomposition.
- `qacpa/axis_extent.py`: coordinate-wise LP extent diagnostics.
- `qacpa/run_stage2_oracle.py`: Stage 2 `A_oracle` baseline runner.
- `qacpa/run_stage2_A.py`: Stage 2 runner using the estimated `A_hat`.
- `qacpa/run_bplus.py`: `bplus` baseline runner.
- `tests/`: Milestone 0 gates as tests.

## Historical diagnostic commands (inactive)

These commands document the existing implementation. They are not the active
M7 run plan; do not execute the Stage-2 commands for M7.

```powershell
python -m qacpa.run --config configs/spike.yaml
python -m qacpa.cmp_sweep --config configs/spike.yaml
python -m qacpa.feas_decomp --config configs/spike.yaml
python -m qacpa.axis_extent --config configs/spike.yaml
```

Expected first config:

```text
bits=4, B=8, rounding=stochastic, seed=0, n_batches=3
```

Rows are appended to `results/runs.jsonl`. Victim tensors are cached under
`cache/`.

Recent summary artifacts:

- `results/v5_4bit_table_lsrescale_cpaica.csv`
- `results/v5_subthreshold_cosines_lsrescale_cpaica.csv`
- `results/v5_subthreshold_histogram_lsrescale_cpaica.png`

## Milestone 0

```powershell
python -m pytest
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
