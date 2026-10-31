# Archive scope and interpretation

This archive was prepared from the CPA workspace on 2026-10-31. It preserves the
existing Git history and overlays selected working-tree source, configurations,
manuscripts, result summaries, and historical execution evidence. The original
workspace, its branch and index, and its raw research assets are preserved.

## Included

- Scientific Python code, meaningful tests, run scripts, configuration files,
  protocol documents, licenses, and requirements.
- `test_cora.tex`, its saved PDF, related manuscript versions, `spconf.sty`,
  `compile_cora.ps1`, and paper assets/builders. The `test*.tex` names denote
  manuscript versions, not disposable test fixtures.
- Small JSON/CSV/Markdown result records, public candidate selection/ranking
  evidence, evaluation summaries, and selected all-image comparison figures.
- Scalar records under `results_private/`; raw private image, label, update,
  and evaluation-feature tensors and model/optimizer states are excluded.
- Selected historical `.ai/execution-handoffs` and `.ai/execution-results`
  records, including source snapshots required by historical recovery runners.
  These are archived evidence, not instructions to launch or resume a job.
- Previously saved environment snapshots. They record the environments at their
  stated dates and are not newly validated dependency lockfiles.

`UPLOAD_MANIFEST.json` inventories the selected files and their SHA-256 hashes.
Existing committed files inherited from the source branch remain in Git history;
the manifest is not an inventory of the whole repository or all local assets.

[Known source issues](KNOWN_ISSUES.md) identifies one unfinished script and five
historical JSON records with a trailing literal backslash-n. Their original bytes
are preserved; the archive does not claim that every historical file is executable
or parseable. File identities are recorded in `KNOWN_ISSUES.json`.

## Retained locally

The archive excludes datasets (including the `datasets/` junction into the separate
DLG workspace), model weights, `.pt`/`.pth` checkpoints, `.pickle`/`.pkl` dumps,
`.npy`/`.npz` arrays, binary quantization packets, large transfer ZIPs, raw training
or optimization logs, Python/pytest caches, compiler dependencies, and duplicate
preview images. Existing previously committed artifacts are not purged from history.

Important local-only asset families include:

| Local path | Reason to retain |
| --- | --- |
| `quantization_aware_cpa/cache/` | Historical victim data and observations; not all files are disposable caches |
| `quantization_aware_cpa/results/` and `results_private/` tensors | Exact observations, reconstruction endpoints, and evaluation inputs |
| `exp/`, `official_cpa/exp/` | Original collected gradients and model/update artifacts |
| `sparsified_update_cpa/outputs_v1/` and `cora_*/outputs_v1/` binaries | Historical sparsification and CoRA observations/results |
| `bit04/`, `bit08/` tensors | Candidate-specific sources and final reconstruction endpoints |
| `handoff_m7_*` and transfer ZIPs | Different transfer packages have different coverage |

Some historical task records reference a separate
`Desktop/fedavg_cora_transfer_20261006` workspace. Assets outside the CPA folder
were not copied by this archive operation. Publishing this repository alone does
not establish that every external experiment is fully backed up.

## Scientific status

- The no-CPA controls were invalidated for failed variable control. Dependent
  normalization, thresholds, claims, and figures are wholly or partly withdrawn;
  consult the [validity registry](../quantization_aware_cpa/results/result_validity.json)
  and its [decision record](../codex/docs/NO_CPA_CONTROL_INVALIDATION_20260909.md).
- The R100 Same80 task's archived `stage_execution.json` records
  `STOPPED_BY_USER` during the 8-bit, batch00 candidate screen. It is incomplete.
- Different datasets, batches, local steps, samplers, search budgets, quantizers,
  and checkpoints identify different protocols. Historical results should not be
  silently combined into a matched comparison.
- Saved manuscripts and historical figures retain their original context and may
  contain superseded claims. Their inclusion is preservation, not validation.

## Verification performed for this upload

Verification is limited to file selection, copied-file hashes, static parsing,
manuscript dependency availability, credential-pattern checks, Git staging, and
remote commit identity. No training, CPA, FIA, benchmark, metric recomputation,
or scientific test suite was run for the archival upload.
