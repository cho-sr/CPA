# FINAL scope correction: fresh16-image naive and CoRA experiment

AMENDMENT_ID: CPA_PAPER_SPARSE_FRESH16_R8_AMENDMENT_20260921
TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
REPLACEMENT_PLAN_VERSION: 4
STATUS: READY_FOR_CURRENT_EXECUTOR
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: max
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_FRESH16_REPLACES_REUSE_PLAN_20260921
EXECUTOR_THREAD: 01a0c0c3-d687-7d92-824c-7499758967d5
PLANNER_THREAD: 01a0bffa-40dd-7213-b387-7022055cec39
REPLACEMENT_OUTPUT_ROOT: quantization_aware_cpa/results/paper_sparse_fresh16_keep50_30_20_R8_20260921
PRIVATE_OUTPUT_ROOT: quantization_aware_cpa/results_private/paper_sparse_fresh16_keep50_30_20_R8_20260921

## Latest user decision takes priority over every reuse plan

User first specified8 candidates, then16 images, and now says:
"뭘 유지해 16장짜리로 새로 해야하는거 아니야?"
Planner clarified that the16-image experiment will be newly run for BOTH naive
and CoRA and that old reconstructions/checkpoints will not be reused.

This v4 supersedes all candidate/checkpoint/metric reuse permissions in v2 and
v3 and all80/R20 scope in v1. Execute one FRESH16-image experiment at each of
keep50/30/20 with naive and8-candidate random comparisons. Do not migrate any
old CPA state, sources, FIA state, candidate scores, selections, reconstructions
or final metrics into this experiment. Do not create an intermediate v2/v3 run.
Historical task ID tokens SAME80/R20 are not the new method; use fresh16/R8 and
plan_version4 in all new metadata. No further scientific approval is needed.

## Dataset and fresh victim observation

Use the fixed original batch00 images global_index0..15 from
quantization_aware_cpa/results/m6_uia_fp32_B16_fia/batch00/inputs.pt,
SHA25624995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141.
These are the16 whose first two examples occur in paper_assets and whose pixels
match the quantization original batch. Reusing DATA identities and the public
pretrained model is required; selecting different16 images is not authorized.
No new public model training or data download is needed.

Register n_images16, B16, batch_indices[0], one independent client update.
Start from the same VGG16 IMAGENET1K_V1 public round0 checkpoint and perform
fresh local SGD on these16 inputs,10 full-batch steps, lr0.001, momentum0,
weight_decay0, frozen conv, all classifier parameters trained, dropout disabled.
This fresh update collection replaces the earlier prohibition on recollection.
Labels are allowed for victim training only and remain private to the attacker.
Save fresh private truth/update separately; do not reuse historical update or
packet files as the actual new observation. Retain the original Y convention:
Y=(W_initial-W_after10)/0.001 for classifier.0.weight [4096,25088].
Encode that SAME newly collected Y once at each keep50/30/20 using the audited
global-within-FC1 magnitude Top-k bitmap+unchangedFP32 codec, original cutoff
tie rule and byte accounting. Naive and CoRA for a fraction receive the EXACT
same freshly saved packet; no private truth in attacker-side filling/ranking.
Compare provenance/settings to old same16 identities as an audit, not as input
substitution. Investigate unexpected differences without outcome-based tuning.

## Fresh naive and random attacks

For each fraction, run one fresh naive zero-fill CPA25000 + FIA25000 control.
Separately run8 fresh random-fill candidates, IDs0..7, fill seeds100000+ID.
Use the unchanged uniform_tau_pcg64_v1 missing-coordinate sampler and preserved
transmitted FP32 values. CPA seed0, FIA seed42, same original upstream CPA/FIA
backend and all optimization hyperparameters as v1. Candidate count is8.

Each random candidate receives CPA25000 and FIA5000 screening, with cosine
horizon25000 from step1. Rank by the same observable recompressed-update
relative-L2 score, public model predicted labels, ascending candidate-index
ties. Continue top3 for20000 additional steps to total25000 and select one
whole16 reconstruction by the same final observable score. No old selections,
metric-driven choices, averaging, new seeds or changed loss functions.
Fresh naive and random inputs use the same model, original16 and observations.

## Transition, ownership and validation

Current Executor owns the safe transition. Stop only the exact superseded
task wrapper/child after verifying PID/start/command/parent. Preserve all old
files and log the supersession; do not let its callback restart old work.
No old checkpoint migration is wanted now. New attacks start from registered
initial states even if corresponding old candidates were already completed.
Do not leave any old80-image or same16-reuse worker running alongside the new
GPU worker. Do not stop an unrelated process or terminal.

Archive the current owned handoff and publish this authoritative v4 there.
Use isolated fresh16/R8 runner/config/report/output paths; do not edit frozen
R20/v2/v3 artifacts or old original10 results. Existing implementation functions
may be reused after source verification; attack states/results may not.
Update validation to one16-image batch,3 fractions,1 naive+8 random candidates,
top3 continuation and no hidden use of historical summaries for new metrics.
Check the new local update rule, codec roundtrip, exact shared packet, candidate
coverage, staged optimizer continuation and no truth in selection. No extra
real-image optimization pilot or additional scientific arm.

## Outputs and budget

Fresh total CPA trajectories:3x(1+8)=27, or675000 CPA iterations.
Random FIA total:3x(8x5000+3x20000)=300000 iterations.
Fresh naive FIA total:3x25000=75000 iterations. Combined FIA=375000.
Random ranking replays:33. Victim update collection:one B16/T10 update.
Measured rates imply about8.6h of total compute plus setup/export, approximately
8--9h; this is longer than v3's~7h because naive is also newly computed.
Do not count old work as new completed computation or as a reuse time saving.
Keep original cumulative resource ceilings including superseded work; no
additional images, fractions, seeds, finalists, epochs or automatic restarts.

Save fresh per-image metrics and aggregate exactly these16 for BOTH methods,
with identical metric conventions and saved evaluation correspondence per
candidate. PSNR/SSIM/VGG LPIPS only; no success rate. Save all16 reconstructions,
candidate scores/checkpoints, fresh packet/input/source manifests and timing,
paired naive/random per-fraction and combined CSV, fixed paper examples0/1,
and all16 contact sheets under REPLACEMENT_OUTPUT_ROOT. No old80 mean is valid
as a fresh16 baseline. No borrowed metrics or reconstructions in these outputs.

Historical10% and CIFAR tasks remain separate and unchanged; do not run or
relabel them as part of this fresh three-fraction task. Leave shared manuscript
and paper_assets updates to Planner's completed-result review.

After checks, launch one hidden persistent fresh16 wrapper and register one
distinct v4 process-exit callback to this existing Executor. Verify startup
once, end the model turn, and report actual final completion/failure once.
No model progress-polling loop. Distinguish amendment queue receipt from actual
process transition and from scientific completion.
