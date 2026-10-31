# Same80 FedAvg CoRA server bundle — full FIA for every member

TASK_ID: CPA_SAME80_FEDAVG_R1357_ALL_CANDIDATES_FIA25K_20261007
PLAN_VERSION: 1
STATUS: READY
ASSIGNED_ROLE: CODEX_EXECUTOR
CLAIMED_BY: UNCLAIMED
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_REQUEST_2026-10-07 — create a server ZIP with CoRA code/data, ImageNet same80, FedAvg rounds 1/3/5/7; remove the FIA 5k/top-3 pruning and take every candidate to FIA 25k.
REQUIRED_MODEL: gpt-6-luna
REQUIRED_REASONING_EFFORT: xhigh
DELIVERY_STATUS: NOT_DELIVERED_EXISTING_EXECUTOR_EXITED (executor_bridge --status: bound CLI PID 3140 is not running)

## Objective

Produce a new, self-contained server ZIP based on the Same80 QACPA/CoRA code that currently implements `screen_steps=5000` and `screen_top_k=3`. The new packaged code must run every registered candidate through the full 25,000 FIA iterations before the existing public-score ranking/selection. Include only FedAvg model checkpoints for rounds 1, 3, 5, and 7, plus the 80 original ImageNet validation JPEGs and their manifest/protocol/loader.

Use the existing Same80 QACPA implementation as the primary CoRA code because it is the implementation matching the user's 5k/top-3 description. The separate `imagenet-fedavg/run_cora_single_client.py` experiment has different observation semantics and already records full 25k runs; do not substitute it as the primary method.

## Inputs

- FedAvg transfer package root: `C:\Users\조상제\Desktop\fedavg_cora_transfer_20261006\fedavg_cora_transfer_20261006`
- FedAvg checkpoints:
  - `imagenet-fedavg\runs\imagenet_pretrained_full_reset_fc_E5_R20\rounds\round_001.pt`
  - `imagenet-fedavg\runs\imagenet_pretrained_full_reset_fc_E5_R20\rounds\round_003.pt`
  - `imagenet-fedavg\runs\imagenet_pretrained_full_reset_fc_E5_R20\rounds\round_005.pt`
  - `imagenet-fedavg\runs\imagenet_pretrained_full_reset_fc_E5_R20\rounds\round_007.pt`
- Same80 data source: `quantization_aware_cpa\exports\imagenet_same80_data_20260911\imagenet_same80.zip` (verified to contain 80 JPEGs; retain its manifest, protocol, README, and loader).
- Same80 QACPA source/config: `quantization_aware_cpa\qacpa\run_same80_bits_r100.py`, `quantization_aware_cpa\qacpa\same80_bits_r100_core.py`, related imports, and `quantization_aware_cpa\configs\same80_bits87654_R100_S5k_K3_F25k.json`.
- Existing code-data archives from 2026-09-27 are historical inputs only. Preserve them byte-for-byte.

## Required implementation and package behavior

1. Work in a new staging/package directory. Do not overwrite existing source, configs, results, or ZIPs.
2. Create a new versioned config (do not edit the historical `S5k_K3_F25k` config) that preserves the existing bits, 100 candidate IDs, five 16-image Same80 batches, seeds, preprocessing, optimizer/loss settings, and public ranking, except for the user-requested FIA workflow change and the selected FedAvg rounds.
3. Remove the 5,000-step FIA screening/continuation branch and all top-3 pruning from the packaged runner. Every candidate must execute FIA iterations 1–25,000. Only after all candidates finish may the existing public feature-cosine-plus-TV score be applied; retain the existing final public selection rule and keep private metrics out of candidate selection.
4. Provide a runnable config/command for each FedAvg checkpoint round 1, 3, 5, and 7. Load the corresponding packaged round checkpoint as that round's starting model. Do not silently fall back to the old round-0 VGG16 checkpoint.
5. Ensure observations/packets used for a round are generated from, or cryptographically verified to match, that round checkpoint and the same80 inputs. Do not label round-0 observations as round-specific. Package the code needed to prepare matching observations on the server; do not run observation generation or any attack locally.
6. Include exactly the four requested FedAvg round checkpoint files and the 80 original JPEGs with manifest/protocol/loader. Include the minimal source/dependency files and a Korean README with extraction and per-round run commands. Use relative paths in the package; avoid workstation-specific absolute paths.
7. Do not include full ImageNet tar files, unrelated experiments, cached duplicate models, old result trees, private result artifacts, or nested historical ZIPs unless a runtime dependency makes one necessary. Document any such necessary inclusion.
8. Produce a ZIP64 archive plus adjacent SHA-256 file and an internal file manifest with sizes and SHA-256 hashes. Use a new output such as `C:\Users\조상제\Desktop\fedavg_cora_transfer_20261006\fedavg_cora_transfer_20261006_same80_r1357_allFIA25k_20261007.zip`; never overwrite a prior archive.

## Boundaries

- This task is code packaging only. Do not launch FedAvg training, observation generation, CPA/FIA reconstruction, GPU work, or a benchmark. Do not add or run a test suite.
- Static checks of file presence, archive entries, byte counts, ZIP CRC, and SHA-256 manifest consistency are permitted and required for delivery.
- Preserve the user-defined scientific constraints and existing results. The only requested algorithmic change is to remove 5k/top-3 screening so all candidates complete FIA 25k before public ranking.
- If the round checkpoint format or current QACPA source/observation contract prevents implementing this package without changing additional scientific assumptions, stop and report `NEEDS_REVIEW` with the exact blocker; do not substitute the already-full-25k FedAvg runner or silently change the model/update protocol.

## Deliverables and completion criteria

- New ZIP and adjacent `.sha256` at the specified output path.
- New packaged source/config/README, preserving historical configs and archives.
- Manifest verifies exactly four round checkpoints and 80 JPEGs, lists each file hash, and has no missing/extra source dependencies.
- Static archive/hash checks pass; no research execution or tests are claimed.
- Record the code/config files changed inside the staged package, actual archive size, SHA-256, and all verification results in the execution record. Set `COMPLETED` only if all criteria pass; otherwise use `NEEDS_REVIEW` or `FAILED` accurately.
