# Planner review 01 — existing task implementation checks

TASK_ID: CPA_SIGNED_BASELINE_B16_T10_20260914
PLAN_VERSION: 1
FROM: CODEX_PLANNER
This is a bounded review of the current authorized implementation, not a new task,
new experiment, changed handoff, or authorization to rerun completed arms.

[FACT] Read-only review of signed_baseline_evaluation.py found that final candidate
hashes and FIA initial/runtime equality are checked, but arm/input provenance is not
yet tied to the sender and actual observation. Passing the same completed arm twice
could pass. Output currently only checks existence and could write original-image
figures under a public observation/attack directory if a CLI path were mistaken.
The current registered paths may be correct; this is a validation gap, not evidence
that an actual experiment or result was contaminated.

[PROPOSAL] Within the already authorized implementation/tests, before offline
evaluation, validate distinct naive/reference directories and correct arm identities;
bind reference result.input.raw_file_sha256 to sender.raw_updates_sha256 and the
actual raw file, and naive result.input.public_manifest_sha256 to
sender.public_manifest_sha256 and actual public manifest. Check the registered
checkpoint/seed/CPA/FIA/input conversion/source versions agree as required by the
approved comparison. Keep evaluation output in the declared private tree, outside
all public observation/naive output trees. Add focused rejection checks for swapped
or duplicated arms/mismatched inputs and a public output location as appropriate.

No change to codec, CPA/FIA math, iteration budget, data, or order is requested.
Do not restart successful collection or completed optimization solely for these
offline guards. Preserve the running task ID/version and continue the same task.

[FACT] Separate read-only reviews found no blocking issue in the normal sender,
signed wire format, raw-Delta CPA/FIA path, or label-free image matching so far.
