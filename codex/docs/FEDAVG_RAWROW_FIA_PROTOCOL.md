# M9 FedAvg raw update rows + absolute value + FIA

User authorization: **"ㅇㅋ ㄱㄱㄱ"**, following the proposed raw-row FIA
experiment. Run tag: **`m9_fedavg_rawrow_dither4_B16`**.

Protocol written on 2026-09-10 before any M9 FIA execution. M8 and historical
results were already known; this fixes the new experiment prospectively and
does not claim retrospective preregistration of earlier work.

Preparation status at registration: **PREPARING**. The runner must freeze
all five batches' selected row IDs, inputs, configuration and code hashes
before the first FIA call. Runtime status belongs in STATUS and execution
artifacts; this protocol is not rewritten in response to observed metrics.

## Meaning and scope

This is an explicit **raw update rows + abs + FIA** baseline without source
separation. Dequantization or the existing dither observation alone produces
an update matrix, not images. The row-selection and absolute-value adapter
below is therefore part of the method and must appear in its description.

The comparison holds the observed updates, feature map, downstream FIA budget
and evaluation rule equal to M8. A raw-row adapter is one specified alternative
to CPA source estimation; it is not a unique or exhaustive CPA ablation and
does not by itself prove CPA's independent contribution under every adapter.
Only the FIA budget is matched: the control omits M8's CPA optimization cost.

The former `m7_no_cpa_control` remains invalid. This run does not reuse that
control's tensors, images, metrics, success thresholds or normalization floor.
[The invalidation record](NO_CPA_CONTROL_INVALIDATION_20260909.md) is unchanged.

## Fixed inputs and row adapter

- Same VGG16 IMAGENET1K_V1 weights and frozen convolutional feature map as M8.
- FedAvg n=16, T=10; five stored client updates and all80 original images.
- Input for batch b is the exact saved `Y_hat` in
  `quantization_aware_cpa/results/m7_uia_4bit_B16/batch{b:02d}/quantized.pt`
  already used by `m8_fedavg_naive_cpa_dither4_B16`.
- Observations originate from actual `(W_initial-W_final)/lr` updates.
  Do not substitute `A_true @ S_true`, recollect updates or quantize again.
- The inherited observation is the per-row 4-bit-interval dither model
  `Y_hat = Y + Uniform(-delta/2, delta/2)`, `delta=2*row_max_abs(Y)/15`.
  It is not an integer 4-bit stochastic-rounding codec.

For every batch b=0,1,2,3,4, select rows using exactly:

```python
generator = torch.Generator(device="cpu").manual_seed(0 + b)
row_indices = torch.randperm(4096, generator=generator)[:16]
raw_sources = Y_hat.detach().cpu().index_select(0, row_indices).float()
target_z = raw_sources.abs().contiguous()
```

`Y_hat` must have shape `(4096,25088)`. Preserve the selected order and all
16 rows, including zero or weak rows. Never resample, remove zero rows, rank
rows by norm, or choose rows by ground-truth similarity or reconstruction
quality. Record the raw signed rows, row IDs, seed, zero flags, row norms and
target hashes. Freeze all80 selected row IDs before the first FIA execution.

There is no row L2 normalization, centering, whitening, learned unmixing, CPA,
LS-A, covariance debiasing, MP threshold/truncation/diagnostic, Stage 2,
refinement, new client training, or source replacement. Elementwise `abs` is
the same final feature-target adapter used for M8's signed CPA sources; it
does not separate the mixed raw observation rows into individual sources.

## Fixed FIA and evaluation boundary

Call the existing `qacpa.upstream_naive.run_upstream_fia` adapter, which
executes the original `official_cpa/src/feature_inversion.py:Direct` class.
Do not call `run_upstream_cpa` or instantiate `CocktailPartyAttack`.

Each batch independently inverts its 16 targets with the M8 configuration:
25000 iterations, TV=0.1, FI=1, Adam lr=0.1, cosine learning-rate scheduler,
logging every5000 iterations, and seed42 reset after feature-model creation.
Use the same five batches of16, preprocessing, VGG16 feature adapter and
runtime settings as M8. Only reconstruction pixels are optimized.

The FIA API receives `target_z` and configuration, not true images, labels or
true features. After a batch's FIA has completed, evaluation may load that
batch's original images and true features. Use **the exact M8 correspondence
rule**: full16 Hungarian matching on the absolute cosine between true
features and the **signed raw sources**, giving one permutation only.

This distinction is intentional: the FIA objective sees `abs(raw_sources)`,
whereas the M8-compatible evaluation assignment uses `raw_sources` with
absolute cosine. Record both semantics rather than silently choosing a new
assignment based on the absolute targets or image-level reconstruction
scores. Any reported target-absolute cosine must be named separately.
Apply no ground-truth scale, sign correction, source substitution or oracle
fill. Retain raw-row-order and evaluation-aligned reconstructions, and score
all16 originals per batch, including outputs from zero or weak targets.

## Outputs and interpretation

Persist input/config/code fingerprints and the pre-FIA row-selection manifest,
signed raw sources, absolute FIA targets, row statistics, original FIA logs,
runtime, and raw-order plus aligned image tensors. Report per-image and
per-batch LPIPS/PSNR/SSIM and signed-source absolute-cosine statistics.

The final report compares M9 with the completed M8 on the same80 originals.
Provide a PNG showing all80 original/reconstruction pairs in global-index
order and a fixed NumPy seed42 random12 grid, explicitly labelled
"not cherry-picked". Include weak and zero-row cases without selection.

Do not calibrate an LPIPS success threshold, define a reconstruction floor,
reuse the withdrawn 68.3% normalization, or revive the former no-CPA/IUT
contribution claim. The primary outputs are the directly measured image
metrics and clearly labelled examples under this declared raw-row adapter.

Runner: `quantization_aware_cpa/qacpa/run_m9_rawrow_fia.py`.
Output directory: `quantization_aware_cpa/results/m9_fedavg_rawrow_dither4_B16`.
The existing M8 protocol remains separately documented in
[FEDAVG_NAIVE_CPA_PROTOCOL.md](FEDAVG_NAIVE_CPA_PROTOCOL.md).
