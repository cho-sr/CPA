"""Wait once for the hidden wrapper and wake the existing Executor."""
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
REPORT = ROOT / ".ai/execution-results/CPA_SIGNED_E2E_H1_NOFIA_20260916"
TASK_ID = "CPA_SIGNED_E2E_H1_NOFIA_20260916"
EXECUTOR_THREAD = "01a0a799-f1ce-7b31-8c46-b4ddef40aeab"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def process_created_ms(handle) -> int:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    kernel.GetProcessTimes.restype = w.BOOL
    times = [w.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    value = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return value // 10000 - 11644473600000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--expected-created-ms", type=int, required=True)
    args = parser.parse_args()
    folder = REPORT / "completion_callback"
    folder.mkdir(parents=True, exist_ok=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    kernel.WaitForSingleObject.restype = w.DWORD
    kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel.GetExitCodeProcess.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, args.wait_pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created = process_created_ms(handle)
        if created != args.expected_created_ms:
            raise RuntimeError(f"wrapper PID was reused: {created} != {args.expected_created_ms}")
        state = {"task_id": TASK_ID, "wait_pid": args.wait_pid,
                 "process_created_ms": created, "executor_thread": EXECUTOR_THREAD,
                 "state": "WAITING_FOR_PROCESS_EXIT", "model_calls_while_waiting": 0,
                 "registered_at": now()}
        reservation = folder / "reservation.json"
        with reservation.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, ensure_ascii=False)
        write_json(folder / "status.json", state)
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = w.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        state.update(state="PROCESS_EXITED", process_exit_code=code.value, ended_at=now())
        write_json(folder / "status.json", state)
        notice = (
            "[CPA noFIA process completion]\n"
            f"TASK_ID: {TASK_ID}\n"
            f"Wrapper PID: {args.wait_pid}; process exit: {code.value}\n"
            f"Report directory: {REPORT.as_posix()}\n"
            "One OS process-exit callback completed with zero model calls while waiting. "
            "Read execution_summary.json and the runner/evaluation logs, verify artifacts, "
            "update the claimed handoff truthfully, ensure report files exist, and send one "
            "final result notice to the existing Planner. Do not rerun, add arms, tune, or poll."
        )
        (folder / "notice.txt").write_text(notice, encoding="utf-8")
        executable = Path("C:/Users/조상제/AppData/Local/Programs/OpenAI/Codex/bin/codex.exe")
        if not executable.is_file():
            found = shutil.which("codex.exe") or shutil.which("codex")
            if not found:
                raise FileNotFoundError("installed codex executable is missing")
            executable = Path(found)
        result = subprocess.run([str(executable), "queue", "--thread", EXECUTOR_THREAD,
                                 "--message", notice], cwd=ROOT, capture_output=True)
        (folder / "queue_stdout.log").write_bytes(result.stdout)
        (folder / "queue_stderr.log").write_bytes(result.stderr)
        state.update(state="NOTIFICATION_QUEUED" if result.returncode == 0 else "NOTIFICATION_UNCONFIRMED",
                     queue_exit_status=result.returncode, notified_at=now())
        write_json(folder / "status.json", state)
        return 0 if result.returncode == 0 else 1
    finally:
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
