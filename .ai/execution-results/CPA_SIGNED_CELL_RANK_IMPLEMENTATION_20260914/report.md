# CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914 — implementation report

**Status:** COMPLETED  
**Plan version:** 1  
**Runtime:** `gpt-5.6-luna / xhigh`  
**Executor thread:** `01a09e4d-51f3-7121-84e2-03a7a9ac3039`  
**Pre-claim handoff SHA256:** `122962474ac3ad8cd6558c21e32277a6bbb2e92c6b40e502f5cfda4ca8e41393`  
**Full experiment:** `FULL_EXPERIMENT_NOT_RUN`

The task/version, recipient thread, model/effort, and pre-edit handoff hash were
verified before claiming the READY job. The handoff was then marked RUNNING with
this thread ID. Only the authorized new implementation, documentation, and small
CPU/mock checks were performed. No other AI/worker, monitor, real observation,
private tensor, download, GPU run, VGG training/reconstruction, collection, or
full CPA/FIA run was started.

## FACT

The new external preprocessing implements the adopted uncentered map in
[`signed_cell_rank.py`](../../../quantization_aware_cpa/qacpa/signed_cell_rank.py):

```text
V0       = dequantize(q, s) = s*q
Mt       = best_rank_r(Vt)
Vt+1     = project_to_observed_cells(Mt, L(q,s), U(q,s))
```

`gram_eigh` forms only the row Gram `V @ V.T`, symmetrizes it, takes the largest
algebraic eigenvectors with `torch.linalg.eigh`, and reconstructs `M` without
autograd or a feature-dimension projector (lines 90–112). The scientific runner
fixes rank 16 and defaults to five complete steps, while exposing
`--ap-iterations`; zero is an identity control (runner lines 38–41, 131–146).
The rank work has an explicit CPU/CUDA device and rejects unavailable CUDA or
enabled TF32 rather than silently falling back (core lines 324–355).

Cell projection uses the existing immutable `quantized_update_cpa.cell_utils`
codec metadata, inclusive FP32 endpoints, and row chunks (core lines 115–154).
The returned final `V` is checked for exact fixed observed `q,s` codes and bounded
cells. `M` is retained separately and can violate cells. Candidate scale and its
error to observed `s` are recorded as diagnostics only; no rescaling or scale
constraint is applied (core lines 157–197).

Diagnostics record the initial matrix and each completed step, movement from
`V0` and the previous `V`, both pair residuals, code/cell metrics, norms, rank,
backend, dtype/device, finite status, runtime, and
`gradient_norm: NOT_APPLICABLE` (core lines 200–307). JSON serialization rejects
NaN/Infinity.

[`signed_cell_rank_runner.py`](../../../quantization_aware_cpa/qacpa/signed_cell_rank_runner.py)
has explicit `preprocess` and `attack` commands. Preprocessing consumes only the
existing public `q,s` packet and writes `Y_q.pt`, final `V.pt`, optional `M.pt`,
diagnostics, registration, and result metadata (lines 131–190). The attack path
passes only final cell-projected `V` to the unchanged original CPA/FIA adapters,
uses the copied immutable CPA/FIA configurations and seed 42, preserves 25,000
CPA and 25,000 Direct FIA iterations, and feeds `abs(S_hat)` in estimated order
(lines 251–335). It records a truthful input conversion, independent cache
directories, source hashes, private/label flags, and `FULL_EXPERIMENT_NOT_RUN`.

[`signed_cell_rank_evaluation.py`](../../../quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py)
is a separate offline evaluator. It authenticates both completed arms and all
public/private hashes, arm identities, registered/executed settings, source
hashes, cache containment, actual input fingerprints, FIA initialization, runtime
flags, and path separation before loading private tensors (lines 79–205). It
accepts the protected legacy naive telemetry location as well as the new arm's
independent FIA cache location, without weakening the identity/hash checks.
After provenance succeeds, it evaluates all 16 candidates label-free using one
pixel-MSE Hungarian matching per arm for PSNR/SSIM/LPIPS; embedding Hungarian
cosine is a separate offline diagnostic and cannot reorder or rescale FIA input
(lines 208–347).

The method, schemas, limits, and exact future commands are documented in
[`signed_cell_rank_comparison.md`](../../../quantization_aware_cpa/docs/signed_cell_rank_comparison.md).
The future commands are documented only and were not executed.

## INFERENCE

The implementation keeps the numerical rank target on `M` and cell feasibility on
the final `V`; therefore the final CPA input is code-feasible without claiming
that it remains rank 16 or that the two constraint sets intersect. Row-Gram
projection is the prescribed memory-conscious Frobenius projection in real
arithmetic, but finite FP32 conditioning and near-eigenvalue ties can affect its
finite-precision result. Unit tests establish implementation behavior on tiny
fixtures, not research efficacy or convergence.

## PROPOSAL

For a later, separately authorized research run, use the three commands in the
documentation with `--ap-iterations 5 --projection-device cuda`, then run the
separate evaluator only after both final arms pass its provenance gates. Preserve
the original CPA/FIA configurations, `abs(S_hat)` adapter, all-16 final-only
comparison, and the fixed constraints in the handoff.

## UNKNOWN / LIMITATIONS

- `FULL_EXPERIMENT_NOT_RUN`: no real B16/T10 preprocessing, rank-16 GPU projection,
  CPA/FIA execution, image evaluation, or private tensor load occurred here.
- Real-size convergence/movement, scale drift, full-size memory/VRAM and runtime,
  cross-device reproducibility, CPA/FIA effect, and image-quality deltas remain
  unknown.
- The row Gram is FP32 `torch.linalg.eigh`, not an exact FP32 SVD; conditioning and
  eigenvalue ties are reported limitations.
- The current evaluator's LPIPS path is exercised only by the documented future
  command; no download or real-image metric was performed in this task.

## Changed files

Only these task-scoped new implementation/test/documentation files were added,
plus this report and `results.json`:

- `quantization_aware_cpa/qacpa/signed_cell_rank.py`
- `quantization_aware_cpa/qacpa/signed_cell_rank_runner.py`
- `quantization_aware_cpa/qacpa/signed_cell_rank_evaluation.py`
- `quantization_aware_cpa/tests/test_signed_cell_rank.py`
- `quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py`
- `quantization_aware_cpa/docs/signed_cell_rank_comparison.md`
- `.ai/execution-results/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914/report.md`
- `.ai/execution-results/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914/results.json`

Protected research sources, codec files, existing observations/results, and
checkpoints were not edited.

## Validation performed

All commands used `C:/miniconda/envs/DLG/python.exe`, tiny synthetic or mocked
inputs, and task-local pytest basetemp directories. The explicit `PYTHONPATH`
contains the repository root and `quantization_aware_cpa` package root.

1. `C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_signed_cell_rank.py quantization_aware_cpa/tests/test_signed_cell_rank_pipeline.py -q --basetemp C:/Users/조상제/Desktop/CPA/.tmp_pytest_signed_cell_rank_2`  
   **PASS — 9 passed in 1.44 s** (one pre-existing pytest cache permission warning).

2. `C:/miniconda/envs/DLG/python.exe -m pytest quantized_update_cpa/tests/test_cell_utils.py -q --basetemp C:/Users/조상제/Desktop/CPA/.tmp_pytest_cell_utils`  
   **PASS — 79 passed in 1.44 s** (one pre-existing pytest cache permission warning).

3. `C:/miniconda/envs/DLG/python.exe -m pytest quantization_aware_cpa/tests/test_upstream_naive.py quantization_aware_cpa/tests/test_signed_baseline_provenance.py -q --basetemp C:/Users/조상제/Desktop/CPA/.tmp_pytest_existing_contracts`  
   **PASS — 30 passed in 5.49 s** (one expected complex-cast warning in the existing
   synthetic upstream contract and one pytest cache permission warning).

The focused tests cover direct-SVD agreement, nonzero-mean uncentered handling,
AP0 identity/no mutation, zero and finite-endpoint codec cells, legitimate
off-rank movement, deterministic CPU output, separate `M`/`V` feasibility,
invalid rank/dtype/nonfinite rejection, original CPA/FIA call/config/seed/abs
adapter spies, truthful provenance, output isolation, and rejection before
private `torch.load`. No test loaded private truth or executed GPU/model code.

## Protected source hash evidence

The pre-existing naive arm's source manifest matched the official/upstream hashes
below before implementation. The same values were recomputed after implementation;
the protected files are unchanged. Additional protected codec/metric hashes were
also captured after implementation for the audit record.

```text
official_cpa/src/gradient_inversion.py  c598e1ee7b4079f741060d10c8dfd14711e393a97384c63cf32be77a458b1873
official_cpa/src/feature_inversion.py   dc73880427e3eaf1f873d295f17f1d48d301882f634ee5df36ddeea8e56c4dec
official_cpa/src/vgg16.py               8114eca8cde553dace5f777aedd21b70851c570d473e597b005cb6be4b248d36
qacpa/upstream_naive.py                 8f29b885d86f4121eda5c55e17486a3a48bd8db08b2f83890fe7726730d1ee92
qacpa/seed.py                           769366fcc2cf3e144d8caa86f73a55dcb0b6be615a45dab8b6e117399150c558
qacpa/signed_baseline.py                a2a6b3e3c7909fb15ab44d49839e6bb0759e44832e223d16f0f2c24e0e45e467
qacpa/signed_baseline_observation.py   2c18135e86e80c8f666033d5ea9d0c3495869ee60809fed04e295f28fa92cda2
quantized_update_cpa/quantization.py    ae0f812c7de3de07cce3796c701475d6fc3e0a822f7d6dd8ab951d34201e3ceb
quantized_update_cpa/cell_utils.py      6725bce108a8e795680342231696c70f1d2bc8b01a727022c605e40aa841d361
quantized_update_cpa/trial_metrics.py   37368d41d5b5caac142949493ab7f501e7d5aecf9ed5e317c7112cc4c4ca1e50
```

The handoff plan and fixed constraints were preserved; only claim/status/result
bookkeeping is updated to record completion.

## One-time Planner result handoff

The bound visible Planner record (`.ai/relay-runs/visible/peers.json`, CODEX
`01a09ac2-0e8d-7f90-94a8-c45be99f4343`, PID 12728) was checked and matched a
running `codex.exe` process. The explicitly authorized one-time `codex queue`
attempt was made after writing this report. It returned exit code 1 with
`no rollout found for thread id ... (code -32603)`. This is recorded in
`planner_final_delivery.json`; no retry, replacement session, or additional
model/monitoring turn was started. The scientific implementation remains
COMPLETED and the result handoff is marked `FINAL_RESULT_HANDOFF_UNCONFIRMED`.
