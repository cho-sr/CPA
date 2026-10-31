from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / ".ai/execution-results/CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918"
SCHEDULE = REPO / ".ai/relay-runs/schedules/CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918"
PLAN = REPO / ".ai/execution-handoffs/CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918_v1.md"
SNAPSHOT = REPO / ".ai/relay-runs/executor/CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918_v1/handoff.md"
ACTIVE = REPO / ".ai/EXECUTION_HANDOFF.md"
PARENT_SCHEDULE_STATUS = REPO / ".ai/relay-runs/schedules/CPA_SIGNED_BITS7654_SCREEN3_20260918/status.json"
TASK = "CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918"
PARENT = "CPA_SIGNED_BITS7654_RECOVERY_20260918"
EXECUTOR = "01a0a799-f1ce-7b31-8c46-b4ddef40aeab"
PLANNER = "01a0a799-d62c-7ab0-9c2a-628acf6d2c95"
ORIGIN = "01a0a7ab-fa58-7d22-ac52-3247b1dd2255"
HANDOFF_SHA = "75acba8097331f0bf9b91d3f9f20eaa7a3c4f8a2f6c4f8015803d46da3cef7aa"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


now = datetime.now(timezone.utc).isoformat()
if sha(SNAPSHOT) != HANDOFF_SHA:
    raise RuntimeError("INT4 immutable snapshot hash mismatch")
if "TASK_ID: " + TASK not in PLAN.read_text(encoding="utf-8"):
    raise RuntimeError("INT4 READY plan identity mismatch")
if "TASK_ID: " + PARENT not in ACTIVE.read_text(encoding="utf-8-sig"):
    raise RuntimeError("active parent handoff changed")

parent_schedule = None
if PARENT_SCHEDULE_STATUS.is_file():
    parent_schedule = json.loads(PARENT_SCHEDULE_STATUS.read_text(encoding="utf-8-sig"))

RESULTS.mkdir(parents=True, exist_ok=True)
SCHEDULE.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SNAPSHOT, RESULTS / "immutable_sidecar_snapshot.md")

registration = {
    "task_id": TASK,
    "status": "DEFERRED_PENDING_PARENT_FINALIZATION",
    "task_kind": "USER_AUTHORIZED_DEFERRED_SUCCESSOR",
    "depends_on": PARENT,
    "plan": str(PLAN),
    "plan_sha256": sha(PLAN),
    "immutable_snapshot": str(SNAPSHOT),
    "immutable_snapshot_sha256": sha(SNAPSHOT),
    "required_handoff_sha256": HANDOFF_SHA,
    "executor": {
        "thread": EXECUTOR,
        "pid": 14236,
        "started_ms": 1789518279030,
        "model": "gpt-5.6-luna",
        "effort": "xhigh",
    },
    "planner_thread": PLANNER,
    "originating_user_thread": ORIGIN,
    "registered_at": now,
    "launch": "NOT_LAUNCHED",
    "optimizer_steps": 0,
    "research_process_started": False,
    "active_parent_handoff_modified": False,
    "activation_gate": "parent public/evaluation/integrity/final-report success verified by designated Executor",
    "activation_policy": "adopt only after parent success and no other active task owns execution",
    "callback_policy": "reuse a valid parent completion wakeup; do not create a competing watcher",
    "one_time_identity_check": {
        "executor_runtime": "verified from designated thread/runtime record",
        "historical_parent_pid": 25160,
        "historical_parent_start_ms": 1789703008341,
        "historical_callback_pid": 25904,
        "historical_process_presence_at_registration": "not_present",
        "new_os_wait_registered": False,
        "reason": "parent process was not present at the single registration-time check; no invalid wait was launched",
    },
    "parent_schedule_reference": {
        "path": str(PARENT_SCHEDULE_STATUS),
        "state_at_registration": parent_schedule.get("state") if parent_schedule else "MISSING",
        "reused": False,
        "reason": "existing schedule is for the earlier four-bit notification and its wait pin is not the current parent attempt",
    },
    "scientific_scope": {
        "method": "original upstream_naive.run_upstream_cpa gradient-only CPA_CONFIG, zero FastICA",
        "bit": 4,
        "separations": 9,
        "new_fia_updates": 125000,
        "private_evaluation": "deferred",
    },
}
write_json(SCHEDULE / "reservation.json", registration)
write_json(SCHEDULE / "status.json", registration)
write_json(RESULTS / "pending_schedule.json", registration)

activation = {
    "task_id": TASK,
    "status": "PENDING",
    "activation_receipt_type": "DEFERRED_SCHEDULE_REGISTRATION",
    "registered_at": now,
    "parent_task": PARENT,
    "parent_success_required": True,
    "parent_success_not_verified": True,
    "research_launch_receipt": None,
    "callback": {
        "mode": "NO_NEW_WAIT_REGISTERED",
        "existing_parent_callback_reference": str(PARENT_SCHEDULE_STATUS),
        "exact_identity_verified_for_executor": True,
        "parent_wait_identity_available": False,
        "model_calls_while_waiting": 0,
    },
    "next_action": "On a verified parent completion wakeup, inspect actual parent outputs and final report; mark BLOCKED_BY_PARENT on failure, otherwise adopt and execute this single INT4 condition.",
    "no_experiment_started": True,
}
write_json(SCHEDULE / "activation_receipt.json", activation)
write_json(RESULTS / "activation_receipt.json", activation)

claim = {
    "task_id": TASK,
    "status": "DEFERRED",
    "assigned_role": "CODEX_EXECUTOR",
    "executor_thread": EXECUTOR,
    "executor_pid": 14236,
    "runtime": "Luna/xhigh",
    "handoff_sha256": sha(SNAPSHOT),
    "parent_task_id": PARENT,
    "active_parent_not_replaced": True,
    "research_launched": False,
    "pending_schedule": str(RESULTS / "pending_schedule.json"),
    "activation_receipt": str(RESULTS / "activation_receipt.json"),
}
write_json(RESULTS / "claim.json", claim)

report = f"""# {TASK}

Status: DEFERRED_PENDING_PARENT_FINALIZATION.

The READY handoff and immutable snapshot were verified for the existing CODEX_EXECUTOR runtime. The active `.ai/EXECUTION_HANDOFF.md` remains owned by `{PARENT}` and was not modified. No INT4 research process, optimizer step, FIA update, private evaluation, model call, or competing callback was started.

The successor is durably registered in `{SCHEDULE}` and `{RESULTS}`. Activation requires the designated Executor to verify actual parent public gate, offline evaluation, integrity and final-report success after the current 4-bit sweep. A historical parent PID check found no live process at registration, so no invalid OS wait was created. The earlier four-bit schedule receipt was preserved and not reused because its wait pin is for an earlier parent attempt.

The deferred scientific scope remains exactly the READY handoff: original upstream gradient-only CPA at INT4 with zero FastICA, nine separations, 125,000 new FIA updates, then gated private comparison and reporting. No successor source or output root has been adopted.

Files:

- pending schedule: `{RESULTS / 'pending_schedule.json'}`
- activation receipt: `{RESULTS / 'activation_receipt.json'}`
- schedule status: `{SCHEDULE / 'status.json'}`
- immutable snapshot: `{RESULTS / 'immutable_sidecar_snapshot.md'}`
- claim: `{RESULTS / 'claim.json'}`

Originating user thread: `{ORIGIN}`.
"""
(RESULTS / "report.md").write_text(report, encoding="utf-8", newline="\n")
