# Recover the approved bit sweep from its completed 7-bit computation

TASK_ID: CPA_SIGNED_BITS7654_RECOVERY_20260918
PLAN_VERSION: 1
STATUS: RUNNING
TASK_KIND: SAME_SCOPE_IMPLEMENTATION_REPAIR_AND_RESUME
PARENT_TASK_ID: CPA_SIGNED_BITS7654_SCREEN3_20260918
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
USER_EXECUTION_AUTHORIZATION: EXISTING_BITS7654_EXECUTION_AND_USER_AUTOMATIC_REPAIR_DIRECTION
ACTIVATION_POLICY: EXISTING_EXECUTOR_OWNED_RECOVERY_NO_DUPLICATE_EXPERIMENT
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_BITS7654_RECOVERY_20260918/report.md

## Authorization and ownership

The user authorized the complete 7 -> 6 -> 5 -> 4 sweep, including necessary
ordinary isolated implementation repairs. After the status report the user said
"오류가 났는데 왜 자동적으로 수정을 안해?". Continue the already authorized work;
do not ask for this same permission again. This is a scoped recovery instruction,
not a new scientific experiment or an expanded compute budget.

Recipient is the currently claimed Executor: Luna/xhigh, thread
01a0a799-f1ce-7b31-8c46-b4ddef40aeab, PID14236, start1789518279030ms. Verify actual
identity/runtime. The requesting chat forwards this user instruction and does
not take over the Executor's research work. Do not start another AI, change a
model, or rebind Planner/Claude/shared Executor peer files.

Parent immutable plan:
.ai/execution-handoffs/CPA_SIGNED_BITS7654_SCREEN3_20260918_v1.md
SHA256 f3f8c60e36f6fd1cece6d30acf6429a15ce0dc14a0ad5bac744f743f6d751612.
Read it fully; all scientific settings, protected sources, output/privacy/ranking
boundaries and ordered scope remain binding. After checking no wrapper is active,
archive the parent's failed wrapper evidence and record this recovery under the
same owned sweep. Do not blindly replace an active/newer task or rerun the parent
wrapper from its beginning.

## Verified cause and completed work

1. qacpa/signed_bits_screen3_runner.py:384 returns a successful public_gate result
   with status, bits and candidate evidence, but no task_id. Its main():411 then
   unconditionally indexes result['task_id'] when printing. The real gate command
   reached this print and raised KeyError after public_gate returned. Do not
   misreport this as failed optimization or failed numeric code consistency.
2. .ai/completion_notify.py used obsolete shared peer PID18272, not current
   Luna PID14236. completion_callback.log shows failure before any OS wait was
   registered. completion_callback_launch.json merely recorded Popen creation;
   this was not proof of a working completion callback. Therefore no exit event
   woke the designated Executor to perform ordinary repair.
3. Parent wrapper26424 has exited. Its execution_summary is FAILED, while the
   shared handoff still says RUNNING. Correct both the stale and misleading
   per-bit status in recovery reporting; do not overwrite the original evidence.

Bit7 public_execution.json reports PUBLIC_COMPLETED, nine separations, all eight
FIA5000 prefixes, center25000 and all three selected continuations to25000;
total125000 FIA updates. Frozen selection is [0,5,3]. Its thirteen saved candidate
sets need authentication and final evaluation, NOT further optimizer steps.
Bits6,5,4 currently have REGISTERED configurations and no executed public stage.
All four configurations were fixed before private loading; retain this fact.

Verified SHA256 inventory (paths relative to the original public/results roots):
- Parent execution-results/execution_summary.json:
  5c43c9c5fc58af6280cd2f7b4563e71c267e87699a1c4b9527edc0b53031882e.
- Bit7 public_execution.json:
  863e7ca84b5e805b22b45713b3b13b37b48f7558b70aa1d668a4412d24c75308.
- Bit7 selection.json:
  e3e296851bc8316a851671a66643823f3edd6e863ffd76407e946d018efcda66.
- Bit7 registration.json:
  cc4ccb524cf6b04bd7c71bd6d54531529e4d40263b0cea550ee2007a854eca8c.
- Bit6 registration.json:
  88d745533e9a609e544b5662c5e381a59b863bebc4ba57c7f9423aae307e2e0b.
- Bit5 registration.json:
  345d50f7e8c02dad6c2f489baa6f88033ad0682cb3489f07ea7df08504b269cd.
- Bit4 registration.json:
  c53b6c0b93e220ded607bb76b989941132d78bc327d846f5ef264b9dd388209b.
- signed_bits_screen3_runner.py:
  0ab6bdac5e121b1713e7124bc8dcc958e9b962d6eb667676389823e87d4ff984.
- signed_bits_screen3_evaluation.py:
  811b2d45b0b5c98b46d018b11e0ff80dec8314364968cba9f43b266d76b51120.

## Required narrow recovery

Prefer a task-local corrected gate entrypoint which invokes the EXISTING
public_gate(bits, root), enforces the successful return contract and prints its
real status with task_id derived from the authenticated bit task. The existing
evaluator already calls public_gate directly and does not need its broken CLI.
This avoids modifying any source hashed in the four original registrations.
Do not bypass, weaken or monkeypatch public_gate. If a research-source change is
actually necessary, record old/new revisions and strict provenance migration;
never edit old registered hashes in place or falsely authenticate old artifacts
against a new source revision.

Create a separate resumable wrapper: authenticate all existing bit7 observations,
source/target/prefix/final outputs and frozen selection; run corrected public
gate -> existing x-only evaluator -> report for bit7 with ZERO optimizer steps;
then execute registered bits6,5,4 in order using the unchanged run_public method,
corrected gate adapter and existing final evaluation. Keep exactly125000 FIA
updates per bit and375000 remaining, never rerun bit7 sender/ICA/FIA or reselect
its top3. Do not call _register_all on occupied roots. Verify registered configs
and source hashes before reusing them. No broad exists-based skip logic.

Keep final artifacts immutable and use new task-local recovery logs, receipts,
hash manifests and report paths. Mark bit7 PUBLIC_COMPLETED_PENDING_EVALUATION
until evaluation completes; mark bits6/5/4 queued, running, completed or actually
blocked with precise cause. Save schedule_state on every real state transition,
including errors, so it cannot remain a stale RUNNING file after wrapper exit.

The original budget remains: bit7 has used11021.525099699968 seconds out of12600;
its remaining allowance is1578.474900300032 seconds before new recovery work.
Aggregate used11024.35411100008 of50400, remaining39375.64588899992 seconds.
Count all subsequent real validation/evaluation/execution as prescribed. Keep
12600 per fresh remaining bit and12GiB CUDA. Development/synthetic checks and
idle time stay excluded as before. Do not reset bit7 or aggregate budgets.

## Completion reliability and automatic same-scope repairs

Replace the failed callback route with a task-local one pinned explicitly to the
CURRENT Executor thread/PID/start and new wrapper PID/start. Do not use or change
the stale generic high peer. The working prior scheduling helper at
.ai/relay-runs/schedules/CPA_SIGNED_BITS7654_SCREEN3_20260918/schedule.py illustrates
exact process handles, recipient checks and a startup acknowledgement, but its
old parent PID/task trigger must NOT be reused as a fresh completion target.

Before ending a launched turn, confirm actual WAITING_FOR_PROCESS_EXIT status,
callback identity/liveness, target PID/start and recipient binding. A launch PID
alone is insufficient. Use one non-model OS wait; no model polling. If callback
startup fails, fix the same authorized transport and verify it before ending the
turn. On exit notify this Executor ONCE for either success or failure. The notice
must direct autonomous repair of ordinary same-scope implementation/export/gate/
report defects using saved states, followed by the remaining registered jobs.
It must not say every failure permanently ends the approved task or requires
fresh permission. Preserve failure evidence and never duplicate already queued
notifications or completed optimization. On unrecoverable corruption, changed
scientific scope, exhausted budget or unavailable recipient, report the actual
blocker rather than retrying blindly.

Finish the parent sweep's reports and send one final result to the existing
Planner. Include the user's originating thread01a0a7ab-fa58-7d22-ac52-3247b1dd2255
in the result context. No new experiment follows these four bits automatically.

## Focused checks before remaining expensive execution

- Exercise actual gate CLI/adapter success and failure paths on a tiny fixture;
  catch missing task_id and assert exit0 only for PUBLIC_GATE_PASSED.
- Mock the resume wrapper with completed bit7 and registered bits6/5/4; assert
  bit7 has zero sender/separation/optimizer calls and exactly one remaining eval,
  then three fresh bit public calls in order and final evaluations.
- Check input/hash/selection mutation rejection, no private load before gate,
  preservation of old registrations and correct per-bit/aggregate accounting.
- Test callback stale-recipient rejection, real ready-acknowledgement handling,
  success/failure wakeups and exactly-once delivery with mocked transport; use a
  short harmless local process for the OS waiting path, not a model/experiment.
- Run only relevant checks and report results. After validated hidden launch,
  one confirmed startup/callback receipt then END turn as instructed. No polling.


ACTIVE_CLAIM_STATUS: RUNNING
CLAIMED_BY: 01a0a799-f1ce-7b31-8c46-b4ddef40aeab
CLAIMED_AT_EPOCH: 1789691653.6271706
