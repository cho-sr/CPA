# Authorized amendment: automatic recovery of the existing Same80 experiment

AMENDMENT_ID: CPA_SAME80_AUTORESTART_20260926
TASK_ID: CPA_SAME80_BITS87654_R100_SCREEN3_20260926
PLAN_VERSION: 1 (scientific contract unchanged)
STATUS: INSTALLED_DISABLED_NEEDS_REVIEW
ASSIGNED_ROLE: CODEX_EXECUTOR
EXECUTOR_THREAD: 01a0dcd1-66a4-7b52-a6c7-0e9d0a5c096e
USER_AUTHORIZATION: EXPLICIT_USER_AUTOMATIC_RESTART_20260926

User: "자동 재시장까지 넣어줘" (add automatic restart too), following the audit
that only exit notification, rather than automatic recovery, was installed.
Implement, test and INSTALL this operational amendment for the ALREADY RUNNING,
already claimed task. Do not replace the scientific task, create a new AI session,
reclaim it, or request the same authorization again. This supersedes the old ban
on automatic restart for this exact task and the existing report-only callback.
It does not authorize new experiments, candidates, seeds, or scientific changes.

## Current live evidence; reverify exact PID/start before acting

- Worker PID14680, created_ms1790420245093, hidden DLG Python; child PID22048.
- Callback PID8300, created_ms1790420314692, completion_notify.py, worker_exit.
- Public root: quantization_aware_cpa/results/same80_bits87654_R100_S5k_K3_F25k.
- Private root: quantization_aware_cpa/results_private/same80_bits87654_R100_S5k_K3_F25k.
- Current orchestrator skips prepare and executes run -> gate -> report.
- Current Planner thread01a0dcd0-1857-7f93-840e-fe277062fa63/PID5888.
- Existing bindings are in .ai/relay-runs/{executor/peer.json,visible/peers.json}.

The current worker must keep running while this feature is installed. Attach to
the exact existing process; do not stop/restart production to demonstrate recovery.

## Required behavior

1. Add a small task-specific, non-model OS supervisor. It owns the recovery decision
   and uses a durable exclusive lock, verified PID/start identity and task/config
   identity. Exactly one research worker is permitted across manual launches,
   supervisor restarts and scheduled invocations. No periodic AI calls or extra agents.
2. On an unexpected process exit or confirmed interrupted run after login/reboot,
   verify that BOTH the prior wrapper and research descendants/lock owners are gone
   before starting a replacement. A surviving phase child must be adopted/waited on
   or reconciled safely; never duplicate its work just because its wrapper died.
3. Automatically resume only the same incomplete authorized task using authenticated
   saved checkpoints, optimizer/scheduler/RNG and completion records. Skip completed
   stages/candidates and prepare. If no valid checkpoint was saved for an interrupted
   stage, the handoff already allows the SAME deterministic seed/settings attempt;
   record lost work and do not count it as a new candidate. Do not repeat a completed
   experiment or restart a finished task. Gate/report failures resume at the proper
   phase without discarding the frozen public selection.
4. Durable bounded recovery: at most 3 consecutive automatic restart attempts
   without authenticated forward progress; backoff 30s, 120s, 600s. Retain the
   attempt count across supervisor/process/OS restarts. Reset the consecutive-failure
   budget only when verified work progresses beyond the prior high-water mark.
   Exhaustion records NEEDS_REVIEW and notifies the user via existing bound sessions.
   Do not silently shrink the experiment to fit the budget.
5. Clearly classify failure causes. Unexpected OS/process loss and demonstrated
   transient infrastructure failures may recover. Nonfinite/numerical results,
   input/config/code/checkpoint hash corruption or changed scientific identity,
   unknown deterministic errors and user-requested stop must NOT loop. Resource
   exhaustion may wait for availability within a bounded policy; do not change
   batch size, algorithms, precision, seeds or delete historical artifacts.
   Preserve FAILED_ERROR/FAILED_NUMERICAL records. Never erase or rewrite failures
   as RUNNING to bypass _start_attempt's existing protection. If a narrow operational
   recovery adapter is necessary, preserve failed evidence and require an explicit
   authenticated recovery record for the exact recoverable attempt.
6. A user's stop/cancel must persistently disable automatic recovery until explicit
   resume. Provide a documented task-specific stop/disable control and honor it in
   the supervisor and every OS launch entry point. Do not interpret intentional
   user cancellation as a crash. Do not interrupt this running experiment now.
7. Keep recovery independent of Luna, Opus or Astra remaining open. Install a
   task-specific current-user Windows scheduled launch at logon and a bounded
   non-model health/restart trigger for the supervisor itself. Use no passwords,
   privilege/policy changes, visible pop-up windows or replacement AI sessions.
   Verify actual OS registration and report whether recovery is available at boot
   or only after the user logs in; do not claim boot recovery from an in-memory
   helper. If Task Scheduler cannot be configured under current rights, use an
   appropriate current-user persistent launch mechanism and disclose its limit.
8. Hangs: no killing based only on stale stage_execution.json/log mtime; existing
   steps/checkpoint intervals can be long. Where reliable multi-signal liveness
   cannot establish a hang, record it as needing review rather than kill a healthy
   worker. State exactly what hang detection is implemented versus unsupported.
9. Coordinate the existing one-shot callback: establish the supervisor first, then
   retire/supersede ONLY the verified PID8300 notification helper if required to
   avoid stale failure notifications or competing recovery. Preserve its logs and
   reservation; this permission is limited to the notification helper, not research
   processes. Updating its source alone will not update the already running helper.
10. Operational status/notifications: append restart/failure evidence and expose
    supervisor PID, watched worker PID, restart count, next retry, disabled/completed
    state and notification state. Use exact pinned peers; unavailable AI sessions
    must not block OS recovery. Persist pending notifications without spawning new
    sessions or flooding queues. Luna remains executor, Astra planner, Opus reviewer;
    none should launch an additional worker independently of the single supervisor.
    Prefer Luna for execution reports, fall back to a live verified Planner/Reviewer
    for notification only if necessary. Do not auto-rebind to an unrelated new session.

## Scope, tests and installation evidence

Reuse existing lock, identity, process, checkpoint, stage and message helpers.
Prefer isolated additive operational scripts/state outside the sealed scientific
code manifest. Do not rewrite or expand the research runners. Do not modify hashed
production files under the current worker. If any operational migration is truly
necessary for a future attempt, retain the original identity and before/after
hashes and explicitly prove the scientific/config/input identity is unchanged;
never silently refresh a manifest to accept changed code.

Use non-production temporary roots and dummy child processes for meaningful tests:
unexpected exit -> exactly one restart; saved-state continuation and completed-stage
skip; duplicate supervisors and orphan-child prevention; retry exhaustion across
supervisor restarts; user stop; numerical/corruption rejection; absent CLI notification
handling; persisted login trigger. Never deliberately crash the real GPU worker.
No extra scientific pilot, GPU fault injection or repeated full research tests.

Finish installation on the live task, not just code creation. Reverify the original
worker is unchanged, one supervisor is active, the OS launch registration exists,
the old notifier is coordinated, and restart state is durable. Record actual tests,
paths, PID/start identities, restart policy, disable command and any residual limits.
Update the claimed handoff's operational status as Executor without changing the
scientific plan/version. Send ONE implementation/installation completion (or concrete
failure) to the current Planner and end the turn; no model polling afterwards.

## Executor installation result — 2026-09-26

The callback arrived after the original worker had already exited. Therefore the
supervisor was installed after PID14680/PID22048 ended; this is not retroactive
recovery of the failed worker. The exact deterministic `_device_state` error is
persisted as `NEEDS_REVIEW`, automatic retries are disabled, and no research
worker was launched. The existing one-shot callback had already queued its
failure notice to Luna, so it was preserved and not duplicated.

Task Scheduler registration failed with `ERROR: Access is denied`. A current-user
HKCU Run entry was installed and verified instead, launching hidden DLG
`pythonw.exe --watch` after login. Watchdog PID18648 / created_ms1790425243758
owns the verified lifetime lock. It checks every 60 seconds; its own caught
health errors use the durable 30/120/600-second retry schedule and then disable
recovery. The fallback is not available at boot, and a hard watchdog exit is
recovered at the next user logon. No stale-log hang killing is implemented.

The 10 focused supervisor tests passed, including duplicate-lock rejection,
orphan-process adoption, retry persistence/exhaustion, stop persistence,
completed-stage selection, error classification, notification persistence and
the task launch guard. Registration SHA256 remains
`9d445f925aa43e2cd00a8464e5345a72e489d51a3bd5ffd36956aaa1c8cda235`; all
registered source/config/checkpoint/packet/runtime hashes passed verification.
