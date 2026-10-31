# Signed INT4 external rank/cell comparison

This document describes the implementation behind the future `INT4_cell_rank`
arm. It is an external preprocessing stage around the existing signed INT4
observation. It does not replace or modify canonical CPA whitening, its loss,
its optimizer, or Direct FIA.

## Contract

The public input is the existing signed per-parameter tensor observation. For
the attack tensor, `q` is CPU `int8` with codes `[-7, 7]` and `s` is the
observed CPU FP32 scalar. The decoder is exactly the existing utility:

```text
V0 = dequantize(q, s) = q * s
```

The external alternating map operates on the **uncentered** matrix:

```text
Mt     = best_rank_r(Vt)
Vt+1   = project_to_cells(Mt, L(q,s), U(q,s))
```

The rank projector is `gram_eigh`:

```text
C  = V V.T
Qr = eigenvectors of C for its r largest algebraic eigenvalues
M  = Qr (Qr.T V)
```

It uses `torch.linalg.eigh` only on the row Gram. The scientific runner fixes
`r=16` for B16 and defaults to five complete alternating steps. The core
accepts smaller ranks for tiny CPU fixtures. `--ap-iterations 0` is an identity
control. No centering, mean restoration, `-Delta/lr` conversion, `T/B`
division, midpoint attraction, scale loss, rescaling, MP threshold, or dither
model is applied.

`M` and `V` are separate artifacts. `M` has the target numerical rank but may
violate cells. The final `V` is the only matrix passed to CPA and is required to
match the observed fixed-scale codes and bounded cells. Its own recomputed
abs-max scale is recorded as a metric; it is never used as an anchor or a
constraint.

## Output schemas

`preprocess` writes a new output directory containing:

- `registered_preprocess.json`: public manifest and six packet hashes, q/s
  shape and dtype, `Y_q` hash, rank/backend/device, exact iteration count, and
  `private_inputs_used: false`.
- `Y_q.pt`: decoded public input (`q*s`) for provenance.
- `V.pt`: final cell-projected CPU FP32 matrix and its tensor/file hashes.
- `M.pt`: final rank projection when at least one alternating step ran.
- `diagnostics.jsonl`: one strict JSON record for the initial matrix and each
  completed step.
- `preprocessing_result.json`: completion status, final hashes, fixed-scale
  code/cell metrics, scale metric, and `full_experiment_status: NOT_RUN`.

The preprocessing artifact records `preprocess_status=COMPLETED`,
`attack_status=NOT_RUN`, and `evaluation_status=NOT_RUN`. An attack artifact
records `attack_status=COMPLETED` with `evaluation_status=NOT_RUN` after both
original adapters finish; failures retain individual `cpa_status`/`fia_status`
and `status=FAILED` in `result.json`.

The attack entry point requires an independently declared preprocessing contract:
AP5 is the default comparison (`--expected-ap-iterations 5`) and AP0 is an
explicit identity control (`--expected-ap-iterations 0`). Registration and
result metadata must agree on rank 16, uncentered `gram_eigh`, FP32/device,
final-only selection, and metric-only scale semantics before original CPA is
called. The finalized-V path and file/tensor hashes are retained in the arm
input provenance.

Each diagnostic records movement from `V0` and the previous `V`, the two pair
residuals (`||Vt-Mt||` and `||Vt+1-Mt||`), exact fixed-scale code mismatches,
bounded-cell violations, candidate scale consistency, matrix norms, rank
backend/dtype/device, finite status, runtime, and a
`gradient_norm: {status: NOT_APPLICABLE, ...}` entry. Zero denominators are
represented by `null` plus a reason. No NaN is emitted.

`attack` writes a distinct `cell_rank_ap5` arm containing a truthful
`registered_arm.json`, independent original CPA/FIA caches, `sources.pt` with
unchanged `S_hat` and `abs(S_hat)` target in estimated order, `candidate.pt`,
the original CPA/FIA result files, raw metrics, and hashes. It uses the same
`CPA_CONFIG`, `FIA_CONFIG`, seed 42, 25,000 CPA iterations, 25,000 Direct FIA
iterations, and final-only selection as the signed naive arm. It accepts only
the public observation, a finalized `V` artifact, and the public checkpoint;
it has no label, replay, private truth, or oracle path.

The CPA execution fingerprint uses the existing adapter's canonical
`upstream_naive._tensor_hash(V)` (NumPy `float32` dtype encoding), exactly as
the original CPA manifest does. Preprocessing `V` tensor/file hashes remain
separate provenance hashes and are not substituted into that fingerprint.

## Label-free offline comparison

`signed_cell_rank_evaluation.py` is a separate command. Before opening private
truth it authenticates both completed arms, candidate/source hashes, public
manifest and checkpoint, distinct arm identities, registered and executed
CPA/FIA settings, original source hashes, cache containment, actual input
fingerprints, source adapter, FIA initialization hash, runtime flags, and path
separation. Existing `INT4_naive` is reused only after these checks; it is never
recollected or rerun.

Initialization evidence is strict: both arms must provide a valid lowercase
64-hex tensor SHA256, `[16, 3, 224, 224]` shape, seed 42, and every fixed
runtime field. The baseline runtime keeps `deterministic_warn_only=true`;
missing fields, invalid hashes, wrong types, or an invalid fixed value fail
before equality comparisons.

After those checks, the evaluator reads the private sender artifact only for
the final offline `x`/`z` evaluation. It never indexes labels. Each arm is
evaluated over all 16 final candidates with one pixel-MSE Hungarian matching;
the same pairs supply PSNR, SSIM, and LPIPS. Embedding Hungarian cosine is a
separate offline analysis of `abs(S_hat)` against truth `z` and cannot reorder
or rescale FIA inputs. The report gives per-image values, means, and
`cell_rank - naive` deltas plus all-16 figures. No candidate or image is
selected by an oracle or a post-hoc composite score.

After the public/final-arm checks, the offline evaluator may load the private
`raw_updates.pt` only to validate the recorded
`theta_after10_minus_theta_start` true update for `classifier.0.weight`. It
reports, per finalized `sq` or `V` input, absolute Frobenius error, true-G
reference norm, and relative error. A zero reference norm records JSON `null`
with `UNDEFINED_ZERO_REFERENCE_NORM` while retaining the absolute error. These
metrics are evaluation-only: they do not reorder candidates, tune, stop, or
change reconstruction.

Numeric comparison JSON and report text are atomically committed before either
figure is rendered. Completion is recorded only after both required files are
present and nonempty. The full view uses a `(16, 3)` grid and the compact view
uses `(4, 12)` so each of the sixteen images is an original/naive/cell-rank
triplet. If plotting fails, the numeric artifacts remain readable with
`numeric_results_status=COMPLETED`, `plot_status=FAILED`, and overall
`evaluation_status=FAILED`; the command returns a failure status.

## Future research commands (implemented, not run here)

Run from the repository root after a separate explicit research-execution
request:

```powershell
C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_runner preprocess --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5 --ap-iterations 5 --projection-device cuda

C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_runner attack --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --preprocessed quantization_aware_cpa/results/signed_cell_rank_B16_T10/preprocess_ap5 --checkpoint "C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth" --output quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5

C:/miniconda/envs/DLG/python.exe -m quantization_aware_cpa.qacpa.signed_cell_rank_evaluation --naive quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive --cell-rank quantization_aware_cpa/results/signed_cell_rank_B16_T10/cell_rank_ap5 --public-observation quantization_aware_cpa/results/signed_int4_baseline_B16_T10/observation --private-observation quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation --output quantization_aware_cpa/results_private/signed_cell_rank_B16_T10/evaluation_ap5
```

These commands are intentionally absent from this implementation's test run.
They would perform real B16 preprocessing/CPA/FIA or private offline
evaluation and therefore remain outside the current scope.

## Implementation validation

The authorized validation is small synthetic CPU data only. The focused tests
compare `gram_eigh` with a float64 direct-SVD reconstruction, cover AP0,
rank-feasible and off-rank movement, endpoint/tie/zero codec cells, rank/shape/
dtype/nonfinite errors, input immutability, deterministic repeated output, and
separate `M`/`V` feasibility. Mock integration tests spy on original CPA/FIA
calls, constants, seed, abs/order, truthful provenance, and cache isolation.
Offline negative tests verify that invalid or incomplete arms are rejected
before private tensors are loaded. They do not run VGG, GPU code, LPIPS, or
real observations.

## Limits

The alternating projection is an engineering implementation, not a convergence
claim. Cell feasibility and rank of `M` do not imply rank of `V`, and fixed-code
agreement does not imply candidate-scale equality. Finite-precision Gram
conditioning, near-eigenvalue ties, full-size memory/time, cross-device
reproducibility, real movement/convergence, CPA/FIA effect, and image quality
remain unknown until a separately authorized full run.
