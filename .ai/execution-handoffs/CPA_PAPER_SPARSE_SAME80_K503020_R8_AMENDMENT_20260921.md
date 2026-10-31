# Authorized amendment: reduce paper sparse candidates from20 to8

AMENDMENT_ID: CPA_PAPER_SPARSE_SAME80_K503020_R8_AMENDMENT_20260921
TASK_ID: CPA_PAPER_SPARSE_SAME80_K503020_R20_20260921
BASE_PLAN_VERSION: 1
REPLACEMENT_PLAN_VERSION: 2
STATUS: READY_FOR_CURRENT_EXECUTOR
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: max
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_CANDIDATES8_CORRECTION_20260921
EXECUTOR_THREAD: 01a0c0c3-d687-7d92-824c-7499758967d5
PLANNER_THREAD: 01a0bffa-40dd-7213-b387-7022055cec39
REPLACEMENT_OUTPUT_ROOT: quantization_aware_cpa/results/paper_sparse_same80_keep50_30_20_R8_20260921

## User decision and immediate priority

User explicitly instructed: "후보 8개로 정정" after discussion of the currently
running keep50/30/20 sweep's runtime. This authorizes reducing candidates20->8
and the necessary safe stop/migration/restart of this specific running job.
Do not request the same approval again. All other scientific settings remain
as specified in the active v1 handoff, subject only to the changes below.
This is an amendment to the current owned task, not a second concurrent sweep.
The R20 text in the historical task ID is an identifier, not the new budget.
Record explicit v2/R8 metadata and human-readable names for the replacement.

Planner did not overwrite the RUNNING handoff, edit research code or stop its
processes. This separate signed-scope amendment is delivered to its existing
Executor, who owns the safe transition and updates the active handoff/version.

## Safe transition before continuing research

At the latest read, wrapper PID17136/start1789943901383ms owns the R20 serial
sweep; child PID14888/start1789944002099ms is running keep50/batch00, candidate1
(zero-based), FIA screen5000. Revalidate PID, creation time, command and parent
relationship before any process control; do not act on stale/reused PIDs.
No other experiment or terminal may be stopped.

Prevent the old wrapper from launching further R20 children. Stop/suspend and
checkpoint the exact task safely, preserving every completed marker/artifact.
Prefer a checkpoint boundary when immediately available; do not continue all20
candidates just to finish the old schedule. If the current partial trajectory
has no recoverable checkpoint, record only that partial loss and restart that
candidate as necessary; never discard completed authenticated candidates.
Do not modify code that a still-running worker may import. Ensure there is at
most one active GPU worker when the R8 replacement starts.

The existing model-free callback watches old wrapper PID17136 under stage
paper_sparse_sweep. Record its intentional supersession so its one exit notice
cannot trigger resumption/evaluation of the obsolete R20 plan. Do not delete
callback receipts or dedupe records. Register a distinct v2 stage callback for
the replacement wrapper after launch. A stale callback is not new work approval.

Archive the current claimed v1 handoff byte-for-byte, then have the current
Executor update EXECUTION_HANDOFF.md to version2, retaining ownership and the
user authorization history. Explain this amendment prominently; do not leave
contradictory R20 budget instructions active. Preserve all R20 source/config/
manifest/result evidence and the deferred CIFAR task. No new research arm.

## Exact candidate and optimization budget

Dataset: unchanged paper_assets ImageNet same80, original5 B16 batches and all
identities, including fixed displayed global indices0/1. Retained fractions
remain50,30,20. Same received packets and public VGG16 checkpoint, local SGD
T10/lr0.001, original CPA backend, uniform_tau_pcg64_v1 sampler and evaluation.

For EVERY fraction/batch, use exactly candidate IDs0..7, selected in advance
independent of scores or images. Seeds remain100000+100*batch_index+candidate_id.
No renumbering, best8-of20 selection, replacement seed or extra random candidate.
CPA25000 per candidate; FIA screen5000 with total cosine horizon25000; advance
the best3 of these8 by the same public recompressed-update relative-L2 score;
continue these3 for20000 more steps to total25000 and select the best full16
candidate by the same final observable score. Same tie-breaking and truth-free
selection. Final private matching/PSNR/SSIM/LPIPS stays unchanged.

New full scientific budget, including reused computation:
- 3 fractions x5 batches x8 =120 CPA trajectories, 3000000 CPA iterations.
- 120x5000 screen FIA +45x20000 continuation FIA =1500000 FIA iterations.
- 15 batches x(8 screen +3 final) =165 public ranking replays.
- Per fraction40 candidates and15 finalists; per batch8 candidates and3 finalists.

Initial measurements: CPA42.37s/candidate, FIA5000=392.61s, replay7.92s.
These imply ~34.5h for the full R8 sweep before minor overhead, minus verified
reused work. Communicate approximately35h, not a guaranteed completion time.
The existing resource ceilings are upper limits, not a target; the original
96h active-compute ceiling includes v1 work and is not restarted at migration.
No other sampling, image, iteration or top3 reduction is authorized.

## Reuse with truthful provenance

Use new R8-named configs/locks/results and a task-local R8 runner/report/wrapper
or an explicitly versioned replacement, keeping the R20 files immutable:
results/cpa_fia_random_topk{50,30,20}_same80_R8_S5k_K3_F25k.
Combined output is REPLACEMENT_OUTPUT_ROOT above. Freeze v2 registration before
new optimization and document exactly the candidate-count-only method change.

Reuse completed CPA, sources, FIA screen checkpoints and observable scores for
candidate0..7 only after verifying old output hashes, same input packet, seeds,
source adapter, settings, actual algorithm dependencies and scheduler horizon.
Record source artifact path/hash, old identity and new identity in a migration
manifest. Code/config identity hashes WILL differ under R8; do not rewrite old
manifests, suppress integrity checks or pretend old identity equals new identity.
Any newly written v2 marker must explicitly retain imported-artifact provenance.
Copy only necessary artifacts; do not duplicate dense datasets/filled FC1 arrays.

Old screen/final SELECTION decisions must never be reused: rank the specified
eight screen candidates under v2. A prior finalist's final trajectory can only
be reused if it was legitimately advanced from the v2 eight and its continuation
state/settings match. Existing candidates8..19, if any, remain historical and
cannot enter R8 selection, metrics or budgets. No prior metric influences this.

Audit all hardcoded20s that mean CANDIDATE COUNT in run_paper_sparse_same80.py,
report_paper_sparse_same80.py and scripts/run_paper_sparse_same80.py plus new
configs, status, validation and reports. Do not globally replace20: keep20%,
20000 continuation steps and tensor/data settings have different meanings.
The imported random_fill.candidate_seed upper limit20 is a legacy bound; enforce
0..7 in the new caller. Preserve the original10% run and its frozen dependencies.

Focused checks should establish exact eight-candidate coverage, top3 from these
eight, all80 identities, authenticated migration equivalence and unchanged
optimizer continuation/evaluation. No extra real-image optimization pilot.
Keep the existing matched naive baselines; no fresh naive training or attacks.

## Deliverables, historical10 and reporting

Keep the v1 deliverables with R8 names and truthful budgets/provenance. Report
both reused and newly computed work. Save per-image/per-batch metrics, selected
reconstructions, all candidate scores, matched naive comparisons, fixed paper
examples and all80 contact sheets. Do not introduce reconstruction success rates.

The old10% result is R20 and has NOT become R8. Preserve it and flag the budget
difference in the combined manifest/report. Do not relabel it R8 or launch a
fourth condition or new finalists for10 without a separate scope instruction.
Planner will review how to describe this in the paper after genuine results.

After the safe transition and meaningful checks, launch one hidden persistent
R8 wrapper, record actual PID/start, register one model-free v2 completion
notification, and end the Executor model turn. No progress polling loop.
Record v1 as superseded/partial, not completed. Notify current Planner once on
final completion/failure according to the existing handoff. Receiving this queue
notice alone does not prove that the process/config switch already happened.
