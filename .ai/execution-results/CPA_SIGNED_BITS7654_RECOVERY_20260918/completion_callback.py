"""Pinned, task-local one-shot callback for the authorized recovery wrapper."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any, Callable


TASK_ID = "CPA_SIGNED_BITS7654_RECOVERY_20260918"
EXECUTOR_THREAD = "01a0a799-f1ce-7b31-8c46-b4ddef40aeab"
EXECUTOR_PID = 14236
EXECUTOR_START_MS = 1789518279030
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / ".ai/execution-results" / TASK_ID
ACTIVE_HANDOFF = REPO_ROOT / ".ai/EXECUTION_HANDOFF.md"
WAIT_TIMEOUT = 258


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _kernel() -> Any:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    kernel.GetProcessTimes.restype = w.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = w.BOOL
    kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    kernel.WaitForSingleObject.restype = w.DWORD
    kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel.GetExitCodeProcess.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]
    return kernel


def checked_process(pid: int, expected_created_ms: int, executable: str) -> tuple[Any, Any]:
    kernel = _kernel()
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        times = [w.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        created_ms = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) // 10000 - 11644473600000
        buffer, size = ctypes.create_unicode_buffer(32768), w.DWORD(32768)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if created_ms != expected_created_ms or Path(buffer.value).name.lower() != executable.lower():
            raise RuntimeError("pinned process identity mismatch")
        return kernel, handle
    except BaseException:
        kernel.CloseHandle(handle)
        raise


def verify_executor() -> dict[str, Any]:
    kernel, handle = checked_process(EXECUTOR_PID, EXECUTOR_START_MS, "codex.exe")
    try:
        if kernel.WaitForSingleObject(handle, 0) != WAIT_TIMEOUT:
            raise RuntimeError("designated Executor is no longer running")
    finally:
        kernel.CloseHandle(handle)
    database = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "state_5.sqlite"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT cwd,model,reasoning_effort,rollout_path,source FROM threads WHERE id=?",
                                 (EXECUTOR_THREAD,)).fetchone()
    normalize = lambda value: os.path.normcase(os.path.abspath(str(value).removeprefix("\\\\?\\")))
    if (not row or normalize(row[0]) != normalize(REPO_ROOT)
            or row[1:3] != ("gpt-5.6-luna", "xhigh") or row[4] != "cli"
            or not Path(row[3]).is_file()):
        raise RuntimeError("designated Executor runtime/session mismatch")
    active = ACTIVE_HANDOFF.read_text(encoding="utf-8-sig")
    if f"CLAIMED_BY: {EXECUTOR_THREAD}" not in active:
        raise RuntimeError("active handoff is not claimed by the designated Executor")
    return {"thread": EXECUTOR_THREAD, "pid": EXECUTOR_PID, "started_ms": EXECUTOR_START_MS,
            "model": row[1], "reasoning_effort": row[2], "cwd": row[0]}


def build_notice(wrapper_pid: int, wrapper_created_ms: int, exit_code: int, summary: dict[str, Any]) -> str:
    return (f"[CPA recovery process completion | {TASK_ID}]\n"
            f"Wrapper PID: {wrapper_pid}; start: {wrapper_created_ms}; process exit: {exit_code}.\n"
            f"Execution summary: {RESULT_ROOT / 'execution_summary.json'}\n"
            f"Recovery report: {RESULT_ROOT / 'report.md'}\n"
            "This was one task-local OS process-exit callback with zero model calls while waiting. "
            "Inspect actual stage status and artifacts before concluding success. Preserve completed "
            "bit7 125000-update artifacts and any completed lower-bit stages. If the summary is PARTIAL "
            "or FAILED, autonomously repair ordinary same-scope gate/export/evaluation/report defects "
            "from saved artifacts and continue the authorized 7 -> 6 -> 5 -> 4 sweep; do not restart "
            "completed optimizer trajectories, add arms, seeds, methods, tuning or budget. On final "
            "validation send exactly one truthful result to Planner thread "
            "01a0a799-d62c-7ab0-9c2a-628acf6d2c95 and end. Originating user thread: "
            "01a0a7ab-fa58-7d22-ac52-3247b1dd2255.")


def queue_notice(notice: str) -> dict[str, Any]:
    executable = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/OpenAI/Codex/bin/codex.exe"
    if not executable.is_file():
        raise RuntimeError("existing Codex executable is missing")
    result = subprocess.run([str(executable), "queue", "--thread", EXECUTOR_THREAD, "--message", notice],
                            cwd=REPO_ROOT, capture_output=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"exit_code": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace")}


def run_callback(wrapper_pid: int, wrapper_created_ms: int, *, attempt: int = 1,
                 wait_fn: Callable[[Any, Any], int] | None = None,
                 queue_fn: Callable[[str], dict[str, Any]] | None = None) -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = "" if attempt == 1 else f"_attempt{attempt}"
    reservation = RESULT_ROOT / f"callback_reservation{suffix}.json"
    status_path = RESULT_ROOT / f"callback_status{suffix}.json"
    if reservation.exists():
        raise RuntimeError("a recovery callback is already reserved")
    executor = verify_executor()
    kernel, handle = checked_process(wrapper_pid, wrapper_created_ms, "python.exe")
    state = {"task_id": TASK_ID, "state": "WAITING_FOR_PROCESS_EXIT", "wrapper_pid": wrapper_pid,
             "wrapper_created_ms": wrapper_created_ms, "attempt": attempt, "callback_pid": os.getpid(),
             "executor": executor, "registered_at": now(), "model_calls_while_waiting": 0}
    reservation.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_json(status_path, state)
    try:
        wait_result = wait_fn(kernel, handle) if wait_fn else kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        if wait_result != 0:
            raise RuntimeError("wrapper wait failed")
        exit_value = w.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_value)):
            raise ctypes.WinError(ctypes.get_last_error())
        summary = read_json(RESULT_ROOT / "execution_summary.json")
        state.update({"state": "PROCESS_EXITED", "process_exit_code": exit_value.value,
                      "ended_at": now(), "recorded_status": summary.get("status"),
                      "recorded_bit_status": summary.get("bit_status")})
        write_json(status_path, state)
        notice = build_notice(wrapper_pid, wrapper_created_ms, exit_value.value, summary)
        (RESULT_ROOT / "notice.txt").write_text(notice, encoding="utf-8", newline="\n")
        state.update({"state": "NOTIFICATION_PENDING", "notification_started_at": now()})
        write_json(status_path, state)
        delivery = (queue_fn or queue_notice)(notice)
        state.update({"state": "NOTIFICATION_QUEUED" if delivery.get("exit_code") == 0 else "NOTIFICATION_FAILED",
                      "delivery": delivery, "notified_at": now()})
        write_json(status_path, state)
        return 0 if state["state"] == "NOTIFICATION_QUEUED" else 1
    except BaseException as error:
        state.update({"state": "NOTIFICATION_UNCONFIRMED", "error": str(error), "failed_at": now()})
        write_json(status_path, state)
        return 1
    finally:
        kernel.CloseHandle(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-pid", required=True, type=int)
    parser.add_argument("--wrapper-created-ms", required=True, type=int)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    return run_callback(args.wrapper_pid, args.wrapper_created_ms, attempt=args.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
