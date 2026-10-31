from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def process_creation_ms(kernel32, handle) -> int:
    times = [w.FILETIME() for _ in range(4)]
    if not kernel32.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return ticks // 10000 - 11644473600000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--expected-created-ms", type=int, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--execution-record", required=True)
    parser.add_argument("--executor-thread", required=True)
    args = parser.parse_args()
    record_path = (ROOT / args.execution_record).resolve()
    folder = ROOT / ".ai/relay-runs/completions" / args.task_id / args.stage
    folder.mkdir(parents=True, exist_ok=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel32.OpenProcess.restype = w.HANDLE
    kernel32.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    kernel32.WaitForSingleObject.restype = w.DWORD
    kernel32.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel32.GetExitCodeProcess.restype = w.BOOL
    kernel32.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, args.wait_pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created_ms = process_creation_ms(kernel32, handle)
        if created_ms != args.expected_created_ms:
            raise RuntimeError(f"wrapper creation time mismatch: {created_ms} != {args.expected_created_ms}")
        state = {"task_id": args.task_id, "stage": args.stage, "wait_pid": args.wait_pid,
                 "process_created_ms": created_ms, "executor_thread": args.executor_thread,
                 "registered_at": now(), "state": "WAITING_FOR_PROCESS_EXIT", "model_calls_while_waiting": 0,
                 "execution_record": str(record_path)}
        with (folder / "reservation.json").open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2)
        write(folder / "status.json", state)
        if kernel32.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        exit_code = w.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        state.update({"state": "PROCESS_EXITED", "process_exit_code": exit_code.value, "ended_at": now()})
        try:
            state["recorded_stage"] = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as error:
            state["record_read_error"] = str(error)
        write(folder / "status.json", state)
        notice = (f"[CPA INT4 GD-only completion wakeup]\nTASK_ID: {args.task_id}; stage: {args.stage}; process: {args.wait_pid}\n"
                  f"Process exit code: {exit_code.value}\nExecution record: {record_path}\n"
                  "This task-local callback waited once on the verified wrapper OS event with zero model calls. "
                  "Read the actual execution_summary.json and public/private artifacts, verify the claimed handoff, "
                  "finish only authorized ordinary evaluation/report repairs from saved states, and send exactly one "
                  "truthful final result to the existing Planner. Do not rerun successful optimization, add arms, "
                  "seeds, methods, tuning or budget, and do not poll or start another AI session.")
        (folder / "notice.txt").write_text(notice, encoding="utf-8")
        executable = shutil.which("codex.exe") or shutil.which("codex")
        if not executable:
            raise RuntimeError("codex executable not found for one completion wakeup")
        queued = subprocess.run([executable, "queue", "--thread", args.executor_thread, "--message", notice],
                                cwd=ROOT, capture_output=True, text=True, timeout=30,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state.update({"state": "NOTIFICATION_QUEUED" if queued.returncode == 0 else "NOTIFICATION_UNCONFIRMED",
                     "queue_exit_code": queued.returncode, "queue_stdout": queued.stdout[-2000:],
                     "queue_stderr": queued.stderr[-2000:], "notified_at": now()})
        write(folder / "status.json", state)
        return 0 if queued.returncode == 0 else 1
    finally:
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
