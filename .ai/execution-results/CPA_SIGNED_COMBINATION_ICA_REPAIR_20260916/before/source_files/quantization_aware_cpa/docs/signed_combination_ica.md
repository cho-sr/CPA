# Signed combination ICA implementation

This implementation adds an isolated public separator at
`qacpa/signed_combination_ica.py` and its future pipeline adapter at
`qacpa/signed_combination_ica_runner.py`. It adopts only the registered ICA-GA
FastICA initialization and active prior-based gradient descent. The original
separator, signed observation, VGG16 checkpoint, Direct FIA adapter, and
six-tensor actual-update cell runner remain the controls and are imported at
their existing interfaces.

The separator accepts only a public FP32 observed matrix `X[m,d]` and a source
count. It row-centers on CPU, computes the covariance with denominator `d-1`,
selects the largest absolute eigenvalues from `torch.linalg.eig`, and creates
one whitening basis. That same `mu`, `K`, and `Xw` object is used by both
stages. FastICA uses parallel symmetric updates with tanh/logcosh alpha 1,
`a[:,None] * W` row broadcasting, identity initialization, symmetric
decorrelation, 200 iterations, and tolerance `1e-6`. A valid first update can
converge; a 200-step nonconvergence is recorded and refinement continues.

Refinement uses plain SGD for 1000 steps with learning rate `0.001`, momentum
`0`, weight decay `0`, no scheduler, and the four registered prior terms. The
stable logcosh expression is algebraically equivalent to
`log(cosh(T) + eps)`. The output keeps the native estimated row order: signed
`S_hat` and the exact elementwise `target_z = abs(S_hat)`, both contiguous CPU
FP32 tensors.

The future CLI is deliberately split into three stages. These commands are
documented for a later explicitly authorized real run and are not run by this
implementation task:

```powershell
Set-Location C:\Users\조상제\Desktop\CPA
$env:PYTHONPATH = 'C:\Users\조상제\Desktop\CPA\quantization_aware_cpa;C:\Users\조상제\Desktop\CPA'
$python = 'C:\miniconda\envs\DLG\python.exe'

& $python -m qacpa.signed_combination_ica_runner separate `
  --output quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916

& $python -m qacpa.signed_combination_ica_runner invert `
  --source-root quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916 `
  --output quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916/direct_fia

& $python -m qacpa.signed_combination_ica_runner refine `
  --source-root quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916 `
  --candidate-root quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916/direct_fia `
  --output quantization_aware_cpa/results/signed_combination_ica_B16_T10_20260916/actual_update_refinement
```

`invert` passes this new `target_z` to the unchanged Direct FIA implementation
with seed `42`, 25,000 steps, Adam learning rate `0.1`, cosine scheduling,
feature weight `1`, and TV weight `0.1`. `refine` authenticates the new source
and candidate, recomputes the coverage-based H1 assignment from the candidate's
public logits and positive last-layer bias-code evidence, freezes that
assignment, runs the unchanged public replay preflight, and passes the new
candidate and target into the unchanged actual-image path. The existing
ten-step FC update and all six FC tensors stay in that path. Historical
slotwise labels are never attached to the new source permutation.

The implementation validation used only small synthetic CPU tensors and mock
bridges. It performed no B16/T10 separation, Direct FIA, image replay, private
evaluation, generator training, downloads, new seed, tuning, or long
continuation. Research runs: `0`. Private tensors loaded: `0`. Generator models
used: `0`. Scientific quality: `NOT_EVALUATED`.

The adapted reference files are covered by `qacpa/ICA_GA_MIT_NOTICE.txt`, which
retains the upstream notice `Copyright (c) 2025 anonymous` and its MIT terms.
