# Paper sparse fresh16 naive versus CoRA8 (v4)

TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
PLAN_VERSION: 4
STATUS: STOPPED_BY_USER_FOR_SSH
TASK_KIND: USER_AUTHORIZED_FRESH16_PAPER_SPARSIFICATION_SWEEP
ASSIGNED_ROLE: CODEX_EXECUTOR
SESSION_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: max
CLAIMED_BY: CODEX_EXECUTOR:01a0c0c3-d687-7d92-824c-7499758967d5/PID9296
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_FRESH16_REPLACES_ALL_REUSE_PLANS_20260921
PLANNER_THREAD: 01a0bffa-40dd-7213-b387-7022055cec39
EXECUTOR_TARGET: USER_DESIGNATED_EXISTING_TERMINAL_LUNA
LOCAL_EXECUTION_AUTHORIZATION: REVOKED_BY_LATEST_USER_STOP_AND_SSH_REQUEST
LOCAL_RESTART_ALLOWED: false
LOCAL_STOP_CONFIRMED_UTC: 2026-09-21T01:49:57.506091+00:00
LOCAL_STOP_RECEIPT: quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921/user_stop_for_ssh_20260921.json
SSH_HANDOFF: .ai/execution-handoffs/CPA_PAPER_SPARSE_FRESH16_SSH_HANDOFF_20260921.md
EXECUTOR_STOP_NOTICE: QUEUED_TO_EXISTING_LUNA; message01a0c1a8-1f73-7c70-8b1c-a456fb2c2408
SSH_TRANSFER_STATUS: WAITING_FOR_USER_TARGET_AND_REMOTE_DIRECTORY
STOP_REQUEST_USER_QUOTE: 루나 실행 멈추고 SSH에 넘길래
STOP_REQUEST_SESSION: 01a0c10d-f041-7692-8d3b-0a6cc1d92293
STOP_REQUEST_PRECEDENCE: This administrative user stop overrides the historical launch and callback states below. Do not resume any local v1/v2/v3/v4 worker, act on an obsolete callback, or claim another local job. Preserve all artifacts. The scientific v4 scope is unchanged; remote transfer/launch awaits the SSH destination.
AMENDMENT_RECEIPT: .ai/execution-handoffs/CPA_PAPER_SPARSE_FRESH16_R8_AMENDMENT_20260921.md
AMENDMENT_RECEIPT_SHA256: b5bfbdb063de90c8360e71cb6246ea43e06f872bd7265c7045aef25baaa38bbc
AMENDMENT_RECEIVED_UTC: 2026-09-21T00:04:21Z
V1_HANDOFF_ARCHIVE: .ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V1_CLAIMED_ARCHIVE_20260921.md
V2_HANDOFF_ARCHIVE: .ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V2_R8_CLAIMED_ARCHIVE_20260921.md
V3_HANDOFF_ARCHIVE: .ai/execution-handoffs/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921_V3_FAILED_ARCHIVE_20260921.md
V3_HANDOFF_ARCHIVE_SHA256: dcc4bf8dfe6de326f1cd86b6672291f86ed4d18382df194303a892010831885d
V3_FAILURE_RECORD: quantization_aware_cpa/results/paper_sparse_same16_keep50_30_20_R8_20260921/v3_failure_record.json
V3_CALLBACK_STATUS: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_sweep_v3_same16/status.json
V3_CALLBACK_STATE: NOTIFICATION_QUEUED_PROCESS_EXIT1_PRESERVED_NO_RESUME
OBSOLETE_CALLBACKS_RESUMED: false
OBSOLETE_WORKERS_RUNNING_AT_V4_REGISTRATION: false

## Authoritative v4 scope

This v4 amendment supersedes every reuse permission in v2 and v3 and the
same80/R20 v1 schedule. It runs one fresh experiment over only the original
batch00 global indices 0..15. Batches01..04, intermediate v2/v3 jobs, parallel
workers, extra arms, extra seeds, extra images and extra10% work are prohibited.

The fresh private collection starts from the public VGG16 IMAGENET1K_V1 round0
checkpoint and performs one independent B16/T10 local SGD update with lr .001,
momentum0, weight_decay0, frozen convolutional layers, all classifier layers
trainable and dropout disabled. The transmitted tensor remains
Y=(W_initial-W_after10)/.001 for classifier.0.weight [4096,25088]. The update
is newly collected and saved under the private v4 root. It is newly encoded at
keep50/30/20 with the global magnitude Top-k bitmap plus unchanged FP32 values.

Each fraction uses the exact new packet for one fresh naive zero-fill CPA25000
+ FIA25000 control and eight fresh random-fill candidates IDs0..7 with seeds
100000+candidate. Random candidates screen FIA5000, rank only by public-model
recompressed-update relative L2, continue public top3 by 20000 steps to total
25000, and select one whole16 candidate. Truth is loaded only for evaluation
after public selection. No old update, packet, checkpoint, source, score,
reconstruction or metric is imported.

## v4 implementation and preflight

V4_RUNNER: quantization_aware_cpa/qacpa/run_paper_sparse_fresh16_r8.py
V4_REPORT: quantization_aware_cpa/qacpa/report_paper_sparse_fresh16_r8.py
V4_WRAPPER: quantization_aware_cpa/scripts/run_paper_sparse_fresh16_r8.py
V4_CONFIG: quantization_aware_cpa/configs/cpa_fia_fresh16_R8_S5k_K3_F25k.json
SWEEP_OUTPUT_ROOT: quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/paper_sparse_fresh16_keep50_30_20_R8_20260921
SOURCE_INPUT: quantization_aware_cpa/results/m6_uia_fp32_B16_fia/batch00/inputs.pt
SOURCE_INPUT_SHA256: 24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141
N_IMAGES: 16
BATCH_INDICES: [0]
CANDIDATE_IDS: [0,1,2,3,4,5,6,7]
KEEP_PERCENT: [50,30,20]
CPA_BUDGET: 27 trajectories / 675000 iterations
FIA_BUDGET: random300000 + naive75000 = 375000 iterations
PUBLIC_RANKING_REPLAYS: 33

Preflight passed before launch: Python syntax and locked-config validation,
 authenticated batch00 hash and identity/order checks, public checkpoint hash
 check, exact superseded PID check, and no old worker/callback resumption gate.
The persistent v4 wrapper is now running with the fresh private update collection
 and packet setup active; no historical attack artifact is counted as completed.

## Launch and callback state

V4_SUPERSESSION_RECORD: quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921/supersession_record.json
V4_INPUT_MANIFEST: quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921/input_manifest.json
V4_WRAPPER_PID: 13636
V4_WRAPPER_CREATED_MS: 1789954357647
V4_STAGE_EXECUTION: quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921/stage_execution.json
V4_COMPLETION_CALLBACK: .ai/completion_notify.py
V4_COMPLETION_CALLBACK_STAGE: paper_sparse_fresh16_v4
V4_CALLBACK_PID: 21688
V4_CALLBACK_CREATED_MS: 1789954675779
V4_CALLBACK_LAUNCH_RECEIPT: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_fresh16_v4/launch_receipt.json
V4_CALLBACK_STATUS_PATH: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_fresh16_v4/status.json
V4_COMPLETION_CALLBACK_STATUS: REGISTERED_ONE_MODEL_FREE_CALLBACK
V4_ACTUAL_TRANSITION_STATUS: RUNNING_WRAPPER_PID13636_CHILD2604_CALLBACK_PID21688_REGISTERED_FRESH_UPDATE_AND_PACKETS_STARTED

After the exact hidden wrapper PID and creation time are verified, register one
model-free OS exit callback at the distinct v4 stage. Do not poll from the
model turn, restart a failed arm, resume an obsolete callback, or start a
parallel worker. The callback records the actual terminal stage status once.


## Superseded v2 callback review

V2_CALLBACK_WAKEUP_RECEIVED_UTC: 2026-09-21T01:41:04.0175831Z
V2_CALLBACK_TRUE_HANDOFF_STATUS: SUPERSEDED_EXIT15_NO_REMAINING_V2_WORK
V2_CALLBACK_REVIEW_RECORD: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_sweep_r8/callback_review.json
V2_CALLBACK_REVIEW_SHA256: 527BD936BA4D9F75C17061E485BD4208C01610E57A47D1ED78FE0D0BEF567BCC
V2_STAGE_EXECUTION_STATUS: SUPERSEDED_AT_RUN_TOPK50_EXIT15
V2_COMPLETED_STAGES: prepare_topk50,prepare_topk30,prepare_topk20,migrate_topk50,migrate_topk30,migrate_topk20
V2_REMAINING_STAGES: NONE; run_topk30,run_topk20,combined_report were never started
V2_ACTIVE_REPLACEMENT_UNCHANGED: v4 fresh16 worker remains RUNNING
 
## Superseded v3 callback review

V3_CALLBACK_WAKEUP_RECEIVED_UTC: 2026-09-21T01:43:09.7196746Z
V3_CALLBACK_TRUE_HANDOFF_STATUS: FAILED_EXIT1_RUN_TOPK50_NO_REMAINING_V3_WORK
V3_CALLBACK_REVIEW_RECORD: .ai/relay-runs/completions/CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921/paper_sparse_sweep_v3_same16/callback_review.json
V3_CALLBACK_REVIEW_SHA256: 9E6D4273ACBDF13FEFA998B16ACAE9E9B29F045AED77203FC2597B2F5DD8820D
V3_STAGE_EXECUTION_STATUS: FAILED_AT_RUN_TOPK50_EXIT1
V3_COMPLETED_STAGES: prepare_topk50, prepare_topk30, prepare_topk20, migrate_topk50, migrate_topk30, migrate_topk20
V3_REMAINING_STAGES: NONE; run_topk30, run_topk20, combined_report never started
V3_FAILURE: imported upstream FIA cache_root was outside the private v3 candidate directory
V3_ACTIVE_REPLACEMENT_UNCHANGED: v4 fresh16 worker remains RUNNING
V3_CALLBACK_REVIEW_ACTION: report failure and preserved artifacts; no relaunch or polling

V3_PLANNER_HANDOFF_ATTEMPTED_UTC: 2026-09-21T01:44:00Z
V3_PLANNER_HANDOFF_STATUS: FAILED_PEER_UNAVAILABLE_AFTER_ONE_ATTEMPT
V3_PLANNER_HANDOFF_TARGET: CODEX Planner peer PID 3348
V3_PLANNER_HANDOFF_ERROR: Connected CLI process 3348 is no longer running
V3_PLANNER_HANDOFF_RETRIES: 0
V3_TRUE_FINAL_HANDOFF_STATUS: REVIEW_RECORDED_MESSAGE_UNDELIVERED_PEER_UNAVAILABLE
