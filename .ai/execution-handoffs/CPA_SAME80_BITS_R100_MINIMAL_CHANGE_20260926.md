# User correction: reuse existing implementation; minimal condition changes

AMENDMENT_ID: CPA_SAME80_BITS_R100_MINIMAL_CHANGE_20260926
TASK_ID: CPA_SAME80_BITS87654_R100_SCREEN3_20260926
ASSIGNED_ROLE: CODEX_EXECUTOR
EXECUTOR_THREAD: 01a0dcd1-66a4-7b52-a6c7-0e9d0a5c096e
STATUS: USER_CORRECTION_FOR_RUNNING_TASK

Latest user: "아니 그냥 조건만 다르게 하면 되는거 아니야? 뭘 이렇게[ 수정해"
This instruction narrows IMPLEMENTATION scope immediately. The approved scientific
settings and authorization to run remain unchanged. Planner acknowledges the earlier
handoff prescribed excessive new infrastructure. Do not keep expanding the new
~1900-line runner/core architecture merely to satisfy that earlier file inventory.

1. Stop expanding new runner/core/report infrastructure. Preserve current files and
   partial work; do not wholesale revert/delete them or touch unrelated changes.
2. Reuse existing original CPA, staged FIA/checkpoint continuation, signed codec and
   sampler, public feature+TV scoring, and existing same80 evaluation/report helpers.
   Prefer parameters/configuration in existing orchestration plus a thin connection
   layer over copying these implementations or building new general infrastructure.
3. Limit code changes to actual incompatibilities: existing80 inputs across5 batches,
   8..4-bit quantized input instead of Top-k filling,100 candidates instead of8/20,
   existing5000->top3->25000 stages, final public-score reranking, and separate output
   names. Use existing lock/hash/save/resume/report mechanisms wherever available.
   Preserve historical default behavior/results. Do not reimplement objective,
   optimizer, whitening, matching, generic validators or resource management.
4. Existing7..4-bit sampling is already reusable; connect established8-bit sampler
   if compatible, or make the smallest bit-support adaptation validated against it.
   Do not silently replace original CPA with the old FastICA-based bit runner.
5. First inspect the actual minimal diff and identify each unavoidable adapter in
   the Executor's visible progress message; this is a scope check, not a request
   for repeat user permission. Then finish only necessary tests and the already
   authorized execution. No new GPU launch on the sprawling implementation before
   this simplification check. Do not kill an existing experiment without user order.
6. Acceptance remains original algorithms and data/conditions, exact code+scale
   consistency, same source matching, public selection, valid continuation/restart,
   all expected candidates and truthful reports. Meet these using existing tested
   functions rather than new duplicate machinery. Report what is reused and the
   final actual change size. Do not treat the earlier planned filenames/line counts
   as mandatory deliverables.

Do not restart the task under a duplicate ID or dispatch another agent. Acknowledge
this correction within the already claimed task and record its receipt there.
