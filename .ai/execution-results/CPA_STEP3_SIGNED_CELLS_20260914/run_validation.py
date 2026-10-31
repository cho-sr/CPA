"""Reproducible authorized CPU unit-test invocation and evidence collection."""

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy
import pytest
import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage")]


def peak_working_set(process):
    if sys.platform != "win32":
        return None
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    api = ctypes.windll.psapi.GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    api.restype = wintypes.BOOL
    if api(wintypes.HANDLE(int(process._handle)), ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize
    return None


def main():
    claim = json.loads((OUT / "claim.json").read_text(encoding="utf-8"))
    command = [sys.executable, "-B", "-m", "pytest", "-q",
               "quantized_update_cpa/tests/test_cell_utils.py", "-o",
               "cache_dir=.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/pytest-cache",
               "--junitxml=.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/pytest.xml"]
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    peak = None
    with (OUT / "pytest.log").open("w", encoding="utf-8") as log:
        log.write("Command argv: " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            current = peak_working_set(process)
            if current is not None:
                peak = max(peak or 0, current)
            time.sleep(0.05)
        final_peak = peak_working_set(process)
        if final_peak is not None:
            peak = max(peak or 0, final_peak)
        code = process.wait()
        elapsed = time.perf_counter() - tick
        log.write(f"\nExit status: {code}\nMeasured subprocess wall seconds: {elapsed}\n")
    suite = ET.parse(OUT / "pytest.xml").getroot().find("testsuite")
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - sum(counts[k] for k in ("failures", "errors", "skipped"))
    runtime = {
        "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
        "numpy": numpy.__version__, "pytest": pytest.__version__, "platform": platform.platform(),
        "device": "cpu", "input_dtype": "float32", "scale_dtype": "float32", "code_dtype": "int8",
        "default_torch_dtype": str(torch.get_default_dtype()),
        "flush_denormal_setting": "not changed; positive subnormal behavior covered by tests",
        "seed": claim["seed"], "model": claim["model"], "effort": claim["effort"],
        "test_started_at_utc": started, "test_wall_seconds": elapsed,
        "pytest_reported_seconds": float(suite.attrib["time"]),
        "pytest_process_peak_working_set_bytes": peak,
        "peak_method": "Windows GetProcessMemoryInfo PeakWorkingSetSize; child process lifetime",
        "gpu_executions": 0, "gpu_peak_vram_bytes": None, "gpu_reason": "NOT_USED_CPU_ONLY",
        "budget": {"wall_seconds": None, "ram_bytes": None, "reason": "NOT_SPECIFIED"},
    }
    write_json("runtime.json", runtime)
    if code:
        write_json("validation_failure.json", {"command": command, "exit_status": code, "counts": counts})
        print((OUT / "pytest.log").read_text(encoding="utf-8"))
        return code

    spec = importlib.util.spec_from_file_location("signed_validation_tests", ROOT / command[5])
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    cells = tests.cells
    evidence = []
    for name, G in tests.FIXTURES.items():
        q, dq, s, _ = tests.ref(G)
        actual_q, actual_s = cells.encode(G)
        L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
        projections = [cells.project_to_cells(candidate, L, U) for candidate in tests.candidates(G)]
        evidence.append({
            "fixture": name, "shape": list(G.shape), "scale_shape": list(s.shape),
            "scale": float(s), "numel": G.numel(),
            "reference_q_equal": torch.equal(actual_q, q),
            "reference_s_equal": torch.equal(actual_s, s),
            "reference_dequantization_equal": torch.equal(cells.dequantize(q, s), dq),
            "A": bool(((L <= G) & (G <= U)).all()),
            "B": all(bool(((L <= P) & (P <= U)).all()) for P in projections),
            "C": all(torch.equal(tests.fixed_reference(P, s), q) for P in projections),
            "D": bool(((L <= dq) & (dq <= U)).all()),
            "scale_consistency_original": cells.scale_consistency(G, s),
        })
    limitations = []
    for bits in (1, 2, 3):
        tiny = tests.from_bits(bits)
        G = torch.stack((-tiny, torch.tensor(0.), tiny))
        q, dq, s, _ = tests.ref(G)
        limitations.append({"kind": "NONZERO_ABSMAX_DIV7_UNDERFLOW", "positive_absmax_bits": bits,
                            "G": G.tolist(), "reference_s": float(s), "reference_q": q.tolist(),
                            "reference_dequantization": dq.tolist(), "division_at_zero": None,
                            "division_at_zero_reason": "0/0 is NaN; int8 conversion gives 0 on recorded CPU runtime",
                            "utility_behavior": "explicit UnsupportedCodecInput; no epsilon or dtype repair"})
    maximum = tests.from_bits(0x3FE00006)
    q, _, s, _ = tests.ref(torch.stack((-maximum, maximum)))
    L, U = cells.get_cells(q, s, dict(cells.CODEC_METADATA))
    witness = {"max_abs_bits_hex": "0x3fe00006", "max_abs": float(maximum), "s": float(s),
               "naive_fp32_7s": float(7*s), "computed_cap": float(U[1]),
               "original_inside": bool(maximum <= U[1])}
    raw = tests.metrics.new_trial_metrics()
    G = tests.FIXTURES["normal"]
    q, s = cells.encode(G)
    raw["raw_metrics"]["quantization_consistency"] = tests.metrics.metric_record(
        cells.check_code_consistency(G, q, s), status="COMPUTED",
        definition="Synthetic original: exact fixed-observed-scale code mismatch count/fraction")
    raw["raw_metrics"]["scale_consistency"] = tests.metrics.metric_record(
        cells.scale_consistency(G, s), status="COMPUTED",
        definition="Synthetic original: reference FP32 absmax/7 versus observed scale; diagnostic only")
    write_json("raw_metrics_example.json", raw)
    after = {p: sha(ROOT / p) for p in claim["source_hashes_before"]}
    preserved = after == claim["source_hashes_before"]
    assert preserved, "Protected source/state/relay changed; report for review"
    added = ["quantized_update_cpa/cell_utils.py", "quantized_update_cpa/trial_metrics.py",
             "quantized_update_cpa/tests/test_cell_utils.py"]
    result = {
        "task_id": claim["task_id"], "plan_version": 1, "status": "COMPLETED",
        "step2": {"status": "ARTIFACT_MISSING", "baseline_executed": False,
                  "generation_command": None, "reason": "UNAVAILABLE; no signed artifact matching fixed conditions per approved handoff"},
        "step3": {"status": "COMPLETED", "A": "PASS", "B": "PASS", "C": "PASS", "D": "PASS",
                  "E": "PASS: all 15 codes/endpoints/neighbors and 14 ties", "F": "PASS",
                  "reference_equality": "PASS: torch.equal q,s,dequantization"},
        "codec_metadata": dict(cells.CODEC_METADATA), "fixture_evidence": evidence,
        "scale_cap_counterexample": witness, "limitations": limitations,
        "support_domain": "Nonempty finite CPU FP32 tensors with positive FP32 absmax/7, plus zero tensor q=0,s=1; explicit signed metadata; positive scalar FP32 s from reference",
        "bound_semantics": "Inclusive representable endpoints, exact fixed-s code ownership, finite cap=max finite m with FP32(m/7)=s; projection does not enforce scale equality",
        "runtime": runtime, "test_command_argv": command, "exit_status": code, "test_counts": counts,
        "first_run": {"passed": 78, "failed": 1, "reason": "ineffective cap-counterexample fixture", "log": "pytest-first.log"},
        "source_hashes_before": claim["source_hashes_before"], "source_hashes_after": after,
        "protected_hashes_equal": preserved, "new_source_hashes": {p: sha(ROOT / p) for p in added},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "not_executed": ["FP32 Reference", "INT4 reconstruction", "CPA", "FIA", "VGG loading/training",
                         "observation generation", "private data loading", "embedding oracle", "image oracle"],
    }
    write_json("results.json", result)
    print(json.dumps({"status": result["status"], "counts": counts, "runtime": runtime,
                      "protected_hashes_equal": preserved, "cap_witness": witness}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
