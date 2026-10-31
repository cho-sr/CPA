# Latest user-authorized scope: one original16-image batch and8 candidates

AMENDMENT_ID: CPA_PAPER_SPARSE_SAME16_R8_AMENDMENT_20260921
TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
REPLACEMENT_PLAN_VERSION: 3
STATUS: READY_FOR_CURRENT_EXECUTOR
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: max
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_REPLACE80_WITH16_AND_KEEP_CANDIDATES8_20260921
EXECUTOR_THREAD: 01a0c0c3-d687-7d92-824c-7499758967d5
PLANNER_THREAD: 01a0bffa-40dd-7213-b387-7022055cec39
REPLACEMENT_OUTPUT_ROOT: quantization_aware_cpa/results/paper_sparse_same16_keep50_30_20_R8_20260921

## Latest instruction overrides both earlier dataset scopes

User now explicitly says "80장 말고 16장짜리로 교체하자", after the earlier
"후보 8개로 정정". Execute only ONE original16-image batch, with8 candidates,
at keep50/30/20. This REPLACES the all80 experiment; it is not just changing
execution order or making a preview followed by the remaining64 images.
Do not launch or continue batches01..04 under either R20 or R8.

This v3 amendment supersedes all80 instructions in both the active v1 handoff
and CPA_PAPER_SPARSE_SAME80_K503020_R8_AMENDMENT_20260921.md (v2). If v2 is not
yet running, transition directly to v3 without creating an intermediate all80
R8 run. All non-conflicting victim, packet, attack, selection, integrity,
migration and role instructions from those documents remain in force.
No further approval is needed for the authorized scope change or safe transition.
The historical task ID's SAME80/R20 tokens no longer describe the actual v3
method; use truthful same16/R8 names and version3 in new artifacts and reports.

## Fixed16-image identities and observations

Use batch00 ONLY from
quantization_aware_cpa/results/m6_uia_fp32_B16_fia/batch00/inputs.pt.
SHA25624995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141.
Its16 entries are global_index0..15 in original source order. B remains16;
n_images=16, batch_indices=[0], independent_client_updates=1.
Do not sample a new batch or pick16 by reconstruction quality.

This is the exact batch containing paper_assets' fixed displayed indices0/1
(ImageNet dataset_index21950 and31051). It is also pixel-identical to the
quantization panel's original16, as previously verified. Preserve data,
checkpoint, local T10 SGD and the saved updates; no new victim training.
For each keep percentage use its original batch00 observation.bin at
results/cpa_fia_topk{50,30,20}_same80_B16/batch00/observation.bin. The original
same80 input manifests may remain parent provenance but do not imply evaluating
or attacking the other batches. Report one packet per condition.

## Unchanged8-candidate reconstruction method

Candidate IDs exactly0..7 with seeds100000+candidate_id for batch00; not best8
chosen from20. Original CPA25000 per candidate. FIA5000 screening at the fixed
25000 cosine horizon; advance the lowest public-score3 and continue each for
20000 additional steps. Final whole16-candidate selection uses the same public
recompressed-update relative-L2 score, with candidate-index ties. No truth
selection, averaging, objective change, new seed or reduced final iterations.

Total production budget including reused work:3 conditions x8=24 CPA runs,
600000 CPA iterations;24x5000 screen FIA +9x20000 continuation FIA=300000 FIA
iterations;3x(8+3)=33 ranking replays. Measured rates imply roughly6.9h for the
complete new scope, minus authenticated reused work, plus implementation/export
overhead. This is an estimate, not a guarantee. Keep original cumulative
resource ceilings; do not restart them or spend the saved time on extra arms.

## Transition and reuse owned by the existing Executor

Planner has left the claimed RUNNING handoff and research processes untouched.
The same Executor must safely supersede the old wrapper, archive claimed plans,
publish authoritative v3 in EXECUTION_HANDOFF.md, then resume only the new scope.
Use the exact process identity checks, callback supersession and no-concurrent-
GPU-worker procedure of the previous amendment. Revalidate live PIDs before
control; no unrelated process or CLI may be stopped.

Preserve and reuse authenticated batch00 candidate0..7 CPA/screen states from
R20 or intermediate R8 wherever settings, hashes and seeds match. Keep explicit
old/new identity and imported-artifact provenance; never rewrite old manifests.
Recompute candidate selection among exactly these8. Preserve any other-batch
artifacts as historical only; do not include them in v3 means or output coverage.

Use new same16/R8 config/lock/output names:
results/cpa_fia_random_topk{50,30,20}_same16_R8_S5k_K3_F25k.
Update task-local validation, loops, reports, contact sheets and coverage checks
to exactly batch00/16,8 candidates and3 finalists. Do not blindly replace all80
or20 literals: old provenance paths, keep20%,20k continuation and shape values
are different concepts. Keep frozen old source/config/result files intact.
Checks must validate exact identities, one packet,8 coverage, top3 selection,
same scheduler continuation and truthful migration without another image pilot.

## Matched naive comparison and outputs

Naive references MUST use the same batch00/global_index0..15. Read the existing
per-image CSV and saved reconstructions and aggregate exactly these16 rows;
never compare the new16 mean with an old80 mean. Use the original saved
correspondence for each naive image and unchanged final offline matching for
random candidates. Same PSNR/SSIM/VGG LPIPS definitions. No success-rate metric.

Save all16 originals/reconstructions in original order, per-image measurements,
one-batch summaries, paired naive/random CSV for each fraction, combined CSV,
packet/code/input hashes, candidate scores/selections/checkpoints, migration
provenance, fixed-index0/1 panel and all16 contact sheets. Use all16_rec.pt and
same16 labels; do not leave all80 claims or pretend new values are already
available in the manuscript. Keep paper_assets rendering to Planner review.

Historical10% remains R20. If referenced for this dataset, use its saved batch00
rows/images and label its different candidate budget accurately; do not insert
its80-image mean or relabel it R8. No new10% arm is authorized by this amendment.
Deferred CIFAR work remains preserved and is not launched by this task.

Register one distinct v3/same16 process-exit callback after the new hidden
persistent wrapper starts, verify one startup receipt, then end the model turn.
Old wrapper completion notices must not resume superseded80-image execution.
On genuine final completion/failure, write the result report and send the one
previously authorized notice. QUEUED alone is not proof of an applied transition.
