# CPA_SIGNED_COMBINATION_ICA_IMPLEMENTATION_20260916

Status: **COMPLETED** for the authorized implementation and focused validation scope.

Scientific quality: **NOT_EVALUATED**. Research runs: **0**. Private tensors loaded: **0**. Generator models used: **0**. No B16/T10 separation, real Direct FIA, real image replay, private evaluation, download, extra seed, tuning, or continuation was run. The future research output root was not created.

## Implemented

- Added the isolated pure tensor core in `quantization_aware_cpa/qacpa/signed_combination_ica.py`.
- Added the public staged CLI and authenticated adapters in `quantization_aware_cpa/qacpa/signed_combination_ica_runner.py`.
- Added the adapted upstream MIT notice, focused CPU tests, and future-run documentation.
- Preserved the original separator and all existing VGG16, observation, Direct FIA, FC replay, signed-cell, and historical result files.

The core uses the registered FP32 CPU row-centered covariance with denominator `d-1`, `torch.linalg.eig`, top absolute eigenvalues, and one whitening object shared by FastICA and refinement. FastICA uses the corrected `a[:,None] * W` row broadcast, parallel symmetric updates, tanh/logcosh alpha 1, identity initialization, max 200 iterations, and tolerance `1e-6`. Valid first-step convergence is accepted; finite nonconvergence returns the last iterate with `converged=false` and refinement continues. Nonfinite states fail.

The second stage starts from that exact FastICA matrix and uses plain SGD for 1000 steps with learning rate `0.001`, momentum `0`, weight decay `0`, no scheduler, the four registered weights, and a stable algebraically equivalent `log(cosh(T)+eps)`. It emits signed `S_hat` in native estimated order and exact `target_z = abs(S_hat)` as contiguous CPU FP32 artifacts, with hashes and descriptive statistics.

The staged integration authenticates the public packet, checkpoint, source artifact, candidate artifact, hashes, schema, and paths. `invert` passes the new target to unchanged `qacpa.upstream_naive.run_upstream_fia` with the registered seed/config. `refine` recomputes coverage-based H1 labels from the current candidate's public logits and positive last-bias codes, freezes them, runs the unchanged replay preflight, and passes the new candidate and target to `signed_e2e_runner.run_arm`. The existing ten-step FC replay, all six FC tensors, 100 image steps, TV weight, and cell weight remain at their registered interfaces. A real runner deadline clock is used; no no-op resource adapter is used.

## Validation

Focused command:

```text
C:/miniconda/envs/DLG/python.exe -m pytest -q tests/test_signed_combination_ica.py tests/test_signed_combination_ica_pipeline.py tests/test_signed_e2e_cells.py
```

Result: **20 passed**. The tests cover the corrected row broadcast, first-step convergence, max-iteration reporting, shared whitening, rank and finite-input rejection, stable logcosh values/gradients, active negentropy gradient, deterministic small non-Gaussian two-stage execution, truth-free core arguments, exact `abs(S_hat)`, current-candidate H1 derivation, target forwarding to both unchanged bridges, six-table forwarding, candidate schema/hash checks, and exact hash-set binding.

Related unchanged-adapter contract command:

```text
C:/miniconda/envs/DLG/python.exe -m pytest -q tests/test_upstream_naive.py
```

Result: **5 passed, 1 warning**. The warning is the pre-existing complex-to-real cast warning in the unchanged official upstream `gradient_inversion.py` test path. `py_compile` for the two new modules and their tests passed, and the new CLI `--help` path passed. Logs are retained beside this report.

## Integrity and future use

`source_integrity.json` records before/after hashes for 13 protected sources, five shared replay sources, the public packet and checkpoint, hashes for the five read-only ICA-GA references, and hashes for all six new files. Protected sources, shared sources, and public inputs were unchanged. The new notice text exactly matches the upstream MIT license, including `Copyright (c) 2025 anonymous`.

The exact future `separate`, `invert`, and `refine` commands, output roots, source-order rule, and no-automatic-run boundary are documented in `quantization_aware_cpa/docs/signed_combination_ica.md`. The new ICA source has an arbitrary permutation, so historical slotwise labels must not be reused; the public coverage rule is regenerated from the new candidate when refinement is later authorized. No image improvement is claimed.
