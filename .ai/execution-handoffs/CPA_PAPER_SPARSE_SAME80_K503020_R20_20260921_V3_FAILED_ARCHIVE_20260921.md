# Paper ImageNet random filling at keep 50%, 30%, 20% (v3/SAME16/R8)

TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
PLAN_VERSION: 3
STATUS: FAILED
TASK_KIND: USER_AUTHORIZED_MATCHED_PAPER_SPARSIFICATION_SWEEP
ASSIGNED_ROLE: CODEX_EXECUTOR
SESSION_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: max
CLAIMED_BY: CODEX_EXECUTOR:01a0c0c3-d687-7d92-824c-7499758967d5/PID9296
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_PAPER_ASSET_DATA_KEEP50_30_20_RERUN_20260921
PLANNER_THREAD: 01a0bffa-40dd-7213-b387-7022055cec39
EXECUTOR_TARGET: USER_DESIGNATED_EXISTING_TERMINAL_LUNA
RESULT_REPORT_PATH: .ai/execution-results/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/report.md
SWEEP_OUTPUT_ROOT: quantization_aware_cpa/results/paper_sparse_same16_keep50_30_20_R8_20260921

## Active v3/SAME16 amendment

AMENDMENT_ID: CPA_PAPER_SPARSE_SAME16_R8_AMENDMENT_20260921
BASE_PLAN_VERSION: 1
SUPERSEDED_PLAN_VERSION: 2
REPLACEMENT_PLAN_VERSION: 3
AMENDMENT_STATUS: FAILED_AT_V3_RUN_TOPK50_NEEDS_REVIEW
AMENDMENT_RECEIPT: .ai/execution-handoffs/CPA_PAPER_SPARSE_SAME16_R8_AMENDMENT_20260921.md
AMENDMENT_RECEIPT_SHA256: 004266033f9b054df392fc5a7d61c30ee85f4ba46256941bb4688b1d8372230f
V1_HANDOFF_ARCHIVE: .ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V1_CLAIMED_ARCHIVE_20260921.md
V1_HANDOFF_ARCHIVE_SHA256: ba63f85795c8ba8b4cbdceea2018dad809a4e090613f04430504862dddea9014
V2_HANDOFF_ARCHIVE: .ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V2_R8_CLAIMED_ARCHIVE_20260921.md
V2_HANDOFF_ARCHIVE_SHA256: 87210156508b18ab8886a36c4094dd9ed8cc2ed37d869486bda5e707a51bed2f
V2_SUPERSESSION_RECORD: quantization_aware_cpa/results/paper_sparse_same80_keep50_30_20_R8_20260921/v2_supersession.json
V2_CALLBACK_SUPERSESSION_RECORD: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_sweep_r8/supersession.json
V2_CALLBACK_STATUS: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_sweep_r8/status.json
V2_ACTUAL_TRANSITION: wrapper PID11696 and task child PID9156 safely stopped with exit15; callback receipt 797580f5-25bc-45d3-ab80-f67e6c813c6c preserved and queued
V3_FAILURE_RECORD: quantization_aware_cpa/results/paper_sparse_same16_keep50_30_20_R8_20260921/v3_failure_record.json
PLANNER_NOTICE_STATUS: UNAVAILABLE; pinned Planner CLI PID3348 is no longer running; one authorized handoff attempt recorded; no retry
V3_TRANSITION_STATUS: FAILED_RUN_TOPK50_EXIT1_UPSTREAM_CACHE_GUARD; ARTIFACTS_PRESERVED_NO_RESTART

The user explicitly replaces the evaluated same80 image scope with the original
batch00 same16 scope while preserving the already authorized candidate IDs0..7.
This v3 plan supersedes the queued v2/R8 same80 schedule. It evaluates only
`results/m6_uia_fp32_B16_fia/batch00/inputs.pt`, original global_index0..15,
with no batches01..04 and no preview or second dataset. It keeps retained
fractions50/30/20, the authenticated received Top-k packets, candidate seeds
`100000 + candidate_id` for batch00, CPA25000, FIA screen5000, public top3
continuation to total25000, the same truth-free public ranking and private
evaluation boundary. No candidate selection, image selection, extra arm,
iteration change or dataset change is authorized.

The authenticated batch00 input hash is
`24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141`.
The packet sources remain the existing baseline files:

* `results/cpa_fia_topk50_same80_B16/batch00/observation.bin` ?
  `9bd2646bb1d9f185b503a036f678d30dce389dae990add706795a4b03d15dec7`,
  218365984 bytes.
* `results/cpa_fia_topk30_same80_B16/batch00/observation.bin` ?
  `598998b2e634f95cfb74d3f44852c0d85dece073973237e849fe75d71eaeae3f`,
  136157624 bytes.
* `results/cpa_fia_topk20_same80_B16/batch00/observation.bin` ?
  `13e1e6bf34547555965b1c7d9a9e1bf9867474c3d77a59597fe77039d9426fac`,
  95053444 bytes.

Naive comparisons use the existing matched baseline per-image CSV rows and
saved reconstruction for exactly those same16 original rows. Reports must never
compare a new16 mean with a historical same80 mean. Every v3 fraction saves
`all16_rec.pt`, `naive_all16_rec.pt`, original batch00 tensors, per-image,
per-batch and paired CSVs, contact sheets, candidate state, selection records,
packet audit and source manifests. Historical Top-k10 remains its R20 result;
if referenced, only its saved batch00 provenance is linked.

## Fixed v3 implementation

Task-local files are isolated from v1/v2 code and result trees:

* `quantization_aware_cpa/qacpa/run_paper_sparse_same16_r8.py`
* `quantization_aware_cpa/qacpa/report_paper_sparse_same16_r8.py`
* `quantization_aware_cpa/scripts/run_paper_sparse_same16_r8.py`
* `quantization_aware_cpa/configs/cpa_fia_random_topk{50,30,20}_same16_R8_S5k_K3_F25k.json`
* `quantization_aware_cpa/tests/test_paper_sparse_same16_r8.py`

The v3 runner uses the authenticated v2 attack adapter only for the CPA/FIA
kernel. Its own validator, input manifest, migration, batch loop, budgets,
markers, report and wrapper enforce `n_images=16`, `batch_indices=[0]`,
candidate IDs0..7 and no later batch. Migration prefers the complete
authenticated v1 R20 candidate state and records v2/v1 provenance; v1/v2
selection decisions are never reused. The v2 same80 tree and its partial
artifacts remain preserved.

Preflight evidence:

* Focused v3 suite: `7 passed`.
* Top-k50 manifest SHA256:
  `d2ad849b9037db7ca8e0e0c912adddb066a05cb24bea9a21ca36abe47fe20896`;
  migration SHA256:
  `aa4da2fb73421e58ac46a6d26281e73fc8d6fe68a1bf7becdbb0e48beb118247`.
* Top-k30 manifest SHA256:
  `5555636f44828a6715d043644e27a588ecad477c469cdcc2367fad17e22f9c70`;
  migration SHA256:
  `137f418e12d330f9b378b5b991382474a2e84ec67d97c33bb625f0dfa266a668`.
* Top-k20 manifest SHA256:
  `b035a33e0be22918997f8822cd51b16ea82fcd2162b747bb47e1ecc553d33435`;
  migration SHA256:
  `c1b04a2710d09253fc2926128c491614dfde8539170cfe585e3c34f9489083ac`.
* Top-k50 migration imports authenticated candidate00/01 completed screen
  state and candidate02 checkpoint state; candidates03..07 are fresh. Top-k30
  and Top-k20 have eight fresh batch00 candidate slots each.

## Launch and completion safeguards

The v3 budget is 3 fractions ? 1 original batch ? 8 candidates = 24 CPA
trajectories / 600000 CPA iterations; 24 screen FIA trajectories ?5000 plus
9 public finalists ?20000 = 300000 FIA iterations; and 33 public ranking
replays. Expected active compute is approximately6.9 hours. Run serially with
one hidden persistent wrapper and no concurrent GPU workers.

After the gates pass, launch
`quantization_aware_cpa/scripts/run_paper_sparse_same16_r8.py`, verify its exact
PID, process start time, command line, startup receipt and v3 stage record.
Register exactly one model-free process-exit callback for this v3 wrapper at a
distinct callback stage. Do not poll from the model turn and do not launch a
monitoring agent. On exit, the callback records the actual terminal status and
the final report path.

