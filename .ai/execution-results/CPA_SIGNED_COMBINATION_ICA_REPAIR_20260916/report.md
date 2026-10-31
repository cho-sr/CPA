# CPA_SIGNED_COMBINATION_ICA_REPAIR_20260916

Status: **COMPLETED** for the authorized implementation repair and focused synthetic integration validation.

Scientific quality: **NOT_EVALUATED**. Research runs: **0**. Private tensors loaded: **0**. Generator models used: **0**. No real B16/T10 separation, Direct FIA, image replay, private evaluation, download, tuning, extra seed, arm, or continuation was run. The future research output root was not created.

## Repair delivered

The executable public pipeline now derives one canonical feature width of `25088` from the existing signed observation/VGG attack interface. The source writer, loader, registration and result schemas all require `[16,25088]`; `sources.pt` contains only FP32 `S_hat` and exact `target_z = abs(S_hat)` in native estimated order. The old erroneous `[16,4096*25088]` contract is gone.

The staged runner now binds the registered separator and Direct FIA presets, sets and records the existing seed-42 deterministic FP32 runtime before each stage, and records Direct FIA pre/post runtime states separately. Each stage records public input, consumed artifact, protected source and shared source hashes before and after processing; a change writes an unsuccessful integrity record and blocks a completed result. Authorized manifest and checkpoint hashes are checked at the public boundary.

Refinement now recomputes the coverage-based H1 hypothesis from the new public candidate, freezes it before optimization, and calls the unchanged all-six-FC actual-image path through the explicit lower-level arm `B_H1_combination_ica`. The top-level method remains `INT4_combination_ica`. A successful public replay preflight is written before optimization; `start_profile.json` is written before the optimizer; the final candidate is authenticated before `final_profile.json` is written. Canonical endpoint profiles retain exact code mismatch and bounded-cell violation counts plus separate fixed-scale and candidate-scale diagnostics for all six FC tensors. Exact feasibility is reported independently from software completion.

## Validation

Focused command:

```text
C:/miniconda/envs/DLG/python.exe -m pytest -q tests/test_signed_combination_ica.py tests/test_signed_combination_ica_pipeline.py tests/test_signed_e2e_cells.py
```

Result: **31 passed**. This includes the prior core and adapter contracts plus real synthetic writer/loader and separate-to-invert orchestration, `[16,25088]` source validation, public candidate target forwarding, endpoint profiles for all six FC tensors, feasible and infeasible mocked outcomes, ordered preflight/profile/refiner calls, downstream evidence preservation, and negative method/config/schema/hash/collision/integrity checks.

Unchanged adapter contract:

```text
C:/miniconda/envs/DLG/python.exe -m pytest -q tests/test_upstream_naive.py
```

Result: **5 passed, 1 warning**. The warning is the pre-existing complex-to-real cast warning in unchanged `official_cpa/src/gradient_inversion.py`. `py_compile` for the new runner/core and focused tests passed, and the CLI help path passed. Logs are beside this report.

## Preservation and limits

Exact pre-repair code/test bytes and the parent completed handoff/report are under `before/`. The documentation pre-repair bytes are also archived with the parent recorded hash. The numerical ICA core remained byte-identical to its repair-start hash. Existing `signed_e2e_runner.run_arm`, Direct FIA, VGG16, observation, shared replay/cell code and historical outputs were not changed.

`source_integrity.json` records the authorized public manifest/checkpoint, 13 protected sources, five shared sources, five adapted reference files, repair file hashes, and archive verification. The tests use only temporary synthetic fixtures and mocked expensive boundaries. They establish software contracts; they do not establish production exact feasibility, reconstruction quality or scientific improvement.

Future staged CLI commands are documented in `quantization_aware_cpa/docs/signed_combination_ica.md` and in `results.json`. A separately authorized research run is still required, and no run follows automatically.
