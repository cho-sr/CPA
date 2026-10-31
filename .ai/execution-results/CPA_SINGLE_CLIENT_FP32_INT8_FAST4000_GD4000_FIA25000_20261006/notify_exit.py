"""One OS exit wait, then one message to the user-authorized current conversation."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--created-ms", type=int, required=True)
parser.add_argument("--thread", required=True)
args = parser.parse_args()
folder = Path(__file__).resolve().parent


def save(value):
    temp = folder / "notification_status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(folder / "notification_status.json")


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
kernel.OpenProcess.restype = w.HANDLE
kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
kernel.CloseHandle.argtypes = [w.HANDLE]
handle = kernel.OpenProcess(0x00100000 | 0x1000, False, args.pid)
if not handle:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    times = [w.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    created_ms = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) // 10000 - 11644473600000
    if created_ms != args.created_ms:
        raise ValueError("Worker PID creation time mismatch")
    with (folder / "notification_reservation.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": args.pid, "created_ms": created_ms, "thread": args.thread}, stream)
    state = {"state": "WAITING_FOR_PROCESS_EXIT", "pid": args.pid, "created_ms": created_ms,
             "thread": args.thread, "model_calls_while_waiting": 0, "registered_unix": time.time()}
    save(state)
    if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
        raise ctypes.WinError(ctypes.get_last_error())
    code = w.DWORD()
    if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
        raise ctypes.WinError(ctypes.get_last_error())
    state.update(state="PROCESS_EXITED", exit_code=code.value, ended_unix=time.time())
    save(state)
    notice = ("[User-authorized one-time experiment completion]\n"
              "TASK_ID: CPA_SINGLE_CLIENT_FP32_INT8_FAST4000_GD4000_FIA25000_20261006\n"
              "SESSION_ROLE: CODEX_EXECUTOR, explicitly assigned to this conversation by the user.\n"
              f"Worker exit code: {code.value}. Record: {folder / 'execution_record.json'}\n"
              "Read that record and the completed report or failure artifact. Verify saved exact FastICA4000/GDICA4000/FIA25000 counts for each of the authorized FP32 and INT8 arms and report the results, including the cached INT4 comparison to the user in this conversation. "
              "Do not send to the historical Luna/Planner bindings, restart an optimization, run new experiments or poll. "
              "The wrapper already performs authorized frozen-image evaluation/reporting after attack. Complete only any necessary reporting repair without rerunning completed optimization.\n")
    (folder / "completion_notice.txt").write_text(notice, encoding="utf-8")
    cli = Path.home() / "AppData/Local/Programs/OpenAI/Codex/bin/codex.exe"
    result = subprocess.run([str(cli), "queue", "--thread", args.thread, "--message", notice],
                            cwd=folder.parents[2], capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    (folder / "notification.stdout.log").write_bytes(result.stdout)
    (folder / "notification.stderr.log").write_bytes(result.stderr)
    state.update(state="NOTIFICATION_QUEUED" if result.returncode == 0 else "NOTIFICATION_FAILED",
                 queue_exit_code=result.returncode)
    save(state)
finally:
    kernel.CloseHandle(handle)
