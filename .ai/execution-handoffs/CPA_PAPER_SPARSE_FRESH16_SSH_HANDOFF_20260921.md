# SSH handoff: fresh16 sparsification experiment

STATUS: LOCAL_STOPPED; REMOTE_TARGET_PENDING
TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
SCIENTIFIC_PLAN_VERSION: 4
USER_REQUEST: 루나 실행 멈추고 SSH에 넘길래
LOCAL_RESTART_ALLOWED: false
REMOTE_TRANSFER_OR_LAUNCH_PERFORMED: false

## Stop receipt and authority

On 2026-09-21 at 01:49:57 UTC (10:49:57 KST), the exact local v4 callback
PID21688, wrapper PID13636 and experiment worker PID2604 were terminated,
in that order, after checking creation times, commands and process ancestry.
Each exited with code15. No experiment files were deleted. The local Luna
terminal was left open; a no-restart notice was queued to its verified thread.
Other Codex sessions, the messaging broker, SSH and GPU monitor were not stopped.

Evidence: `quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921/user_stop_for_ssh_20260921.json`.
That receipt preserves the pre-stop status records. Current live status files
now distinguish this user cancellation from success or an experiment failure.
Do not reactivate historical callback reservations or local workers.

## Scope to retain on SSH

The authoritative scientific contract is
`.ai/execution-handoffs/CPA_PAPER_SPARSE_FRESH16_R8_AMENDMENT_20260921.md` and
`quantization_aware_cpa/configs/cpa_fia_fresh16_R8_S5k_K3_F25k.json`.
Use original batch00/global indices0..15, B16/T10, fresh v4 victim update,
global-within-FC1 Top-k keep50/30/20, shared FP32 packets for naive/random,
one naive control and eight random candidates0..7 per fraction, CPA25000,
FIA5000 screen/top3 continuation to total25000, observable-only whole-batch
selection, and evaluation only after selection. No new scientific arms or
changes are authorized. Historical same80/R20 tokens in the task ID are not
the current scope. Do not import v1/v2/v3 attack artifacts or old metrics.

## Preserved v4 state

- Public run tree: `quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921/`.
- Private victim/evaluation tree: `quantization_aware_cpa/results_private/paper_sparse_fresh16_keep50_30_20_R8_20260921/`.
- Fresh victim update/truth and all three newly encoded packets exist.
- keep50/random/candidate00 and candidate01 have screen records, reconstructions
  and FIA checkpoints; their logs reached5000 screening iterations.
- keep50/random/candidate02 has CPA sources/checkpoint and a partial FIA
  checkpoint. Its last saved progress record reports iteration1; in-memory
  work after the last save is not recoverable from process termination.
- No final three-fraction comparison is complete. File presence is not a
  completed-result claim; checkpoint integrity and portability remain untested.

## Transfer preparation (not yet executed)

Obtain the user's SSH alias or user@host and remote project directory before
transferring private experiment data or launching remotely. An existing SSH
process is not confirmation of the intended destination. Preserve local files.

Transfer the current code/configs and their dependencies, `official_cpa/`,
the v4 public/private trees, original batch00 `inputs.pt`, and the required
public VGG16 checkpoint/environment specification. Do not transfer credentials,
SSH private keys, unrelated datasets or old experiment trees indiscriminately.
Source input SHA256:
`24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141`.

Runner: `quantization_aware_cpa/qacpa/run_paper_sparse_fresh16_r8.py`.
Report: `quantization_aware_cpa/qacpa/report_paper_sparse_fresh16_r8.py`.
The Python module exposes `--phase validate|run|report`; inspect entry-point
side effects before invoking it on the destination. Do not invoke the wrapper
with `-h` for inspection: its script does not parse help and launches work.

Windows absolute paths in manifests, upstream CPA/FIA `cache_root` references,
private paths and source provenance require a relocation audit. A previous v3
import failed the private-cache containment guard; do not disable that guard.
Keep immutable local evidence and record any remote-only path mapping. Verify
checkpoint integrity and resume semantics before claiming saved work can
continue. Do not silently reset the v4 fresh observation or replace its data.
No remote command, package transfer, environment setup or restart has yet run.
