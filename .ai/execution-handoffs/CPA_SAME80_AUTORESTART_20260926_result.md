# Automatic recovery amendment result ? CPA_SAME80_AUTORESTART_20260926

TASK_ID: CPA_SAME80_BITS87654_R100_SCREEN3_20260926  
PLAN_VERSION: 1  
AMENDMENT: CPA_SAME80_AUTORESTART_20260926  
VERIFIED_AT_UTC: 2026-09-26T12:39:20+00:00  
RESULT: INSTALLED_BUT_CURRENT_ATTEMPT_FAIL_CLOSED_NEEDS_REVIEW

## Current execution state

The research worker is not running. Original wrapper PID 14680 (created_ms 1790420245093) exited with code 1; phase child PID 22048 and the one-shot notifier PID 8300 are also gone. The preserved `worker_status.json`, `run.stderr.log`, and callback receipt remain authoritative. The traceback records `AttributeError: module 'qacpa.staged_fia' has no attribute '_device_state'` in `public_attack`. This is a deterministic code error, so the amendment's recovery policy forbids retrying it automatically. No replacement worker was launched and no completed artifact or failure record was changed.

## Installed recovery adapter

The task-specific model-free supervisor is `.ai/cpa_same80_supervisor.py`; its durable state and exclusive locks are under `.ai/relay-runs/supervisors/CPA_SAME80_BITS87654_R100_SCREEN3_20260926/`. The live hidden watchdog is PID 18648, created_ms 1790425243758, running DLG `pythonw.exe ... --watch`; its latest durable health tick is `2026-09-26T12:38:44.087973+00:00`. Its state is `NEEDS_REVIEW`, blocker `DETERMINISTIC_ERROR_REVIEW`, restart_count 0, and active_worker null. The exact Same80 configuration/registered research source hashes and prepared registration were rechecked: 23 file hash entries match, config hash matches, and registration SHA256 is `9d445f925aa43e2cd00a8464e5345a72e489d51a3bd5ffd36956aaa1c8cda235`. The research code, scientific identity, and prepared inputs were not modified for this result.

For an eligible unexpected OS/process loss or demonstrated transient infrastructure failure, the persisted limit is at most 3 consecutive retries with 30/120/600-second backoff; the budget resets only after verified progress. Numerical, corruption, unknown deterministic failures, user stop, completed tasks, and `NEEDS_REVIEW` are fail-closed. Recovery is therefore installed but inactive for this failed attempt pending review. The supervisor never kills on stale log timestamps; reliable hang detection is unsupported.

Task Scheduler registration failed with `ERROR: Access is denied`. The verified fallback is HKCU Run value `CPA_SAME80_BITS87654_R100_SCREEN3_20260926`, launching hidden DLG `pythonw.exe -B .ai/cpa_same80_supervisor.py --watch`. It starts after the current user logs in, not at boot; if the watchdog exits, the Run entry restarts it on the next login. Its in-process model-free health tick interval is 60 seconds. The previous one-shot callback's failure notice was already queued once to Luna and was preserved; it was not duplicated or used to restart research.

Persistent user-stop control (blocks future recovery and direct exact-task DLG launches; does not kill a live process):

```powershell
C:/miniconda/envs/DLG/python.exe -B .ai/cpa_same80_supervisor.py --disable
```

The existing isolated supervisor suite previously passed 10/10 tests, including dummy-process duplicate-lock/orphan handling, retry persistence/exhaustion, continuation selection, stop persistence, fail-closed errors, pending notification, and launch guard. No production experiment or broad test suite was run during this verification.

## Handoff

The scientific task remains `NEEDS_REVIEW`; it is neither running nor complete. No new worker, retry, model session, or scientific condition change is authorized by this operational result. Current bound Planner: thread `01a0dcd0-1857-7f93-840e-fe277062fa63`, PID 5888. One final operational result is being sent to that binding.

PLANNER_RESULT_STATE: QUEUED_ONCE
PLANNER_THREAD: 01a0dcd0-1857-7f93-840e-fe277062fa63 / PID 5888 (current binding verified)
PLANNER_RECEIPT: 01a0ddba-91ec-7f61-b3f3-2951caa43c1b
PLANNER_MESSAGE_SHA256: b71cc191bd5fb70e6901ab63818b0a313f17bc18df74bead4e6c40ab7b146367
