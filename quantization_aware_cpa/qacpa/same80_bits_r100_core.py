"""Task-local codecs, integrity checks, selection and execution bookkeeping."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator
import uuid

import numpy as np
import torch


BITS = (8, 7, 6, 5, 4)
N_BATCHES = 5
N_IMAGES = 16
FC1_SHAPE = (4096, 25088)
N_CANDIDATES = 100
PREFIX_STEPS = 5000
TOTAL_STEPS = 25000
TOP_K = 3
SAMPLER_CHUNK_ROWS = 32
MIN_FREE_BYTES = 8 * 1024**3
TASK_ID = "CPA_SAME80_BITS87654_R100_SCREEN3_20260926"
PLAN_VERSION = 1
CHECKPOINT_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
STAGE1 = {
    "n_components": 16, "seed": "batch_index", "n_iter": 25000,
    "lr": 0.001, "opt": "adam", "sch": "none", "decor": 5.3,
    "T": 7.7, "ne": 1.0, "nv": 0.13, "l1": 5.0, "tv": 0.1,
    "log_every": 5000, "debias": False, "mp_truncation": False,
    "refine_A": False, "stage2": False,
}
FIA = {
    "n_iter": 25000, "screen_steps": 5000, "continuation_steps": 20000,
    "lr": 0.1, "optimizer": "adam", "scheduler": "cosine", "tv": 0.1,
    "fi": 1.0, "seed": 42, "cosine_horizon": 25000,
    "log_every": 5000,
}
PUBLIC_SCORE = "feature_cosine_distance_mean + 0.1 * normalized_TV"


class TaskError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TaskError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_tensor(path: Path, value: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value.detach().cpu().contiguous(), temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def qmax_for_bits(bits: int) -> int:
    if type(bits) is not int or bits not in BITS:
        raise ValueError(f"bits must be one of {BITS}")
    return (1 << (bits - 1)) - 1


def codec_metadata(bits: int) -> dict[str, Any]:
    qmax = qmax_for_bits(bits)
    return {
        "codec": f"signed_symmetric_bits{bits}_qplus{qmax}_nearest_even_cpu_fp32_v1",
        "bits": bits, "qmin": -qmax, "qmax": qmax,
        "scale_scope": "per_tensor_absmax_scalar_fp32",
        "scale_operation": f"float32(max_abs/{qmax})",
        "code_operation": f"round(float32(G/s)).clamp(-{qmax},{qmax}).int8",
        "rounding": "nearest_even", "logical_code_bits": bits,
        "physical_code_storage": "int8", "zero_tensor": "q=0,s=1",
        "edge_policy": "largest_finite_fp32_absmax_rounding_to_observed_scale",
    }


def encode_tensor(value: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = qmax_for_bits(bits)
    if (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() == 0
            or not bool(torch.isfinite(value).all())):
        raise ValueError("input must be a finite nonempty CPU FP32 tensor")
    value = value.detach().contiguous()
    if bits != 8:
        # Preserve the established 7/6/5/4 implementation byte for byte.
        from .signed_bits_observation import encode as established_encode
        return established_encode(value, bits)
    maximum = value.abs().max()
    scale = maximum / qmax if bool(maximum > 0) else torch.ones((), dtype=torch.float32)
    if not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("invalid per-tensor scale")
    q = torch.round(value / scale).clamp(-qmax, qmax).to(torch.int8)
    if bool((q < -qmax).any()) or bool((q > qmax).any()) or bool((q == -128).any()):
        raise ValueError("8-bit signed code escaped the registered range")
    return q.contiguous(), scale


_MAX_FINITE_BITS = 0x7F7FFFFF


def _from_positive_bits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).view(torch.float32)


def _fixed_codes(value: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    return torch.round(value / scale).clamp(-qmax, qmax).to(torch.int8)


def _finite_cap(scale: torch.Tensor, qmax: int) -> tuple[torch.Tensor, int]:
    low, high = 0, _MAX_FINITE_BITS + 1
    while low < high:
        middle = (low + high) // 2
        if bool(_from_positive_bits(torch.tensor(middle, dtype=torch.int32)) / qmax <= scale):
            low = middle + 1
        else:
            high = middle
    cap_bits = low - 1
    cap = _from_positive_bits(torch.tensor(cap_bits, dtype=torch.int32))
    if not torch.equal(cap / qmax, scale):
        raise TaskError("observed scale has no finite FP32 absmax preimage")
    return cap, cap_bits


def cell_bounds_table(scale: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = qmax_for_bits(bits)
    if (scale.device.type != "cpu" or scale.dtype != torch.float32 or scale.ndim != 0
            or not bool(torch.isfinite(scale)) or not bool(scale > 0)):
        raise ValueError("scale must be a positive CPU FP32 scalar")
    if bits != 8:
        from .signed_bits_randomized import cell_bounds_table as established_bounds
        return established_bounds(scale, bits)
    cap, cap_bits = _finite_cap(scale, qmax)
    thresholds = torch.arange(1, qmax + 1, dtype=torch.int64)
    low = torch.zeros(qmax, dtype=torch.int64)
    high = torch.full((qmax,), cap_bits + 1, dtype=torch.int64)
    while bool((low < high).any()):
        middle = (low + high) // 2
        below = _fixed_codes(_from_positive_bits(middle), scale, qmax).long() < thresholds
        active = low < high
        low = torch.where(active & below, middle + 1, low)
        high = torch.where(active & ~below, middle, high)
    if bool((low > cap_bits).any()):
        raise TaskError("a signed 8-bit code has no bounded FP32 preimage")
    lower_positive = _from_positive_bits(low)
    preceding = _from_positive_bits(low - 1)
    lower = torch.cat((-torch.cat((preceding[1:], cap.reshape(1))).flip(0),
                       -preceding[:1], lower_positive))
    upper = torch.cat((-lower_positive.flip(0), preceding[:1], preceding[1:], cap.reshape(1)))
    codes = torch.arange(-qmax, qmax + 1, dtype=torch.int8)
    if not (torch.equal(_fixed_codes(lower, scale, qmax), codes)
            and torch.equal(_fixed_codes(upper, scale, qmax), codes)):
        raise TaskError("FP32 endpoint ownership check failed")
    left = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
    right = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    if bool((((left >= -cap) & (_fixed_codes(left, scale, qmax) == codes)).any()
             | ((right <= cap) & (_fixed_codes(right, scale, qmax) == codes)).any())):
        raise TaskError("FP32 cell endpoint is not maximal")
    return lower, upper


def sample_matrix(q: torch.Tensor, scale: torch.Tensor, *, bits: int, seed: int,
                  chunk_rows: int = SAMPLER_CHUNK_ROWS) -> tuple[torch.Tensor, dict[str, Any]]:
    qmax = qmax_for_bits(bits)
    if type(seed) is not int or seed < 0 or chunk_rows != SAMPLER_CHUNK_ROWS:
        raise ValueError("invalid seed or sampler chunk_rows differs from the inspected default")
    if q.ndim != 2 or q.device.type != "cpu" or q.dtype != torch.int8:
        raise ValueError("q must be a CPU rank-2 int8 tensor")
    if bool((q < -qmax).any()) or bool((q > qmax).any()):
        raise ValueError("q is outside the selected logical bit range")
    if bits != 8:
        from .signed_bits_randomized import sample_matrix as established_sample
        if not bool(q.abs().max()):
            if not torch.equal(scale, torch.ones((), dtype=torch.float32)):
                raise ValueError("zero tensor must use the codec's unit scale")
            result = torch.zeros(q.shape, dtype=torch.float32)
            return result, _sample_audit(result, q, scale, bits, seed, None)
        result, audit = established_sample(q, scale, bits=bits, seed=seed, chunk_rows=chunk_rows)
        audit["codec_metadata"] = codec_metadata(bits)
        return result, audit
    if not bool(q.abs().max()):
        if not torch.equal(scale, torch.ones((), dtype=torch.float32)):
            raise ValueError("zero tensor must use the codec's unit scale")
        return torch.zeros(q.shape, dtype=torch.float32), _sample_audit(
            torch.zeros(q.shape, dtype=torch.float32), q, scale, bits, seed, None)
    # qmax is 127 for signed 8-bit values; use the established INT8 cell
    # sampler instead of maintaining a second implementation of its endpoints.
    from .signed_int8_randomized import sample_matrix as established_sample
    result, audit = established_sample(q, scale, seed=seed, chunk_rows=chunk_rows)
    audit["bits"] = bits
    audit["codec_metadata"] = codec_metadata(bits)
    audit["zero_tensor_convention"] = "q=0,s=1 -> deterministic all-zero candidate"
    return result, audit


def _sample_audit(result: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                  bits: int, seed: int, anchor: dict[str, Any] | None) -> dict[str, Any]:
    qmax = qmax_for_bits(bits)
    mismatch = _fixed_codes(result, scale, qmax) != q
    if bool(mismatch.any()) or not bool(torch.isfinite(result).all()):
        raise TaskError("sample failed exact fixed-scale code consistency")
    if not bool(q.abs().max()) and not torch.equal(result, torch.zeros_like(result)):
        raise TaskError("zero tensor sample must be deterministic zero")
    array = result.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str(array.dtype).encode("ascii") + str(array.shape).encode("ascii")
                            + memoryview(array).cast("B")).hexdigest()
    return {
        "bits": bits, "seed": seed, "chunk_rows": SAMPLER_CHUNK_ROWS,
        "generator": "torch.Generator(cpu)",
        "sampler": "uniform_fp64_interpolation_cast_fp32_inclusive_registered_cells",
        "codec_metadata": codec_metadata(bits), "anchor": anchor,
        "scale": float(scale), "scale_reproduced": True,
        "zero_tensor_convention": "q=0,s=1 -> deterministic all-zero candidate",
        "code_consistency": {"all_equal": True, "mismatch_count": 0,
                              "numel": q.numel(), "mismatch_fraction": 0.0},
        "tensor_sha256": digest,
    }


def candidate_seed(batch_index: int, candidate_index: int) -> int:
    if not 0 <= batch_index < N_BATCHES or not 0 <= candidate_index < N_CANDIDATES:
        raise ValueError("candidate seed indices are outside the frozen plan")
    return 1234 + 1000003 * (100 * batch_index + candidate_index)


def _score(row: dict[str, Any], field: str) -> float:
    score = row.get(field)
    if isinstance(score, dict):
        score = score.get("score")
    value = float(score)
    if not np.isfinite(value):
        raise TaskError(f"nonfinite public score: {row.get('candidate_index', row.get('member_index'))}")
    return value


def rank_full(rows: list[dict[str, Any]], expected_ids: list[int], field: str) -> list[dict[str, Any]]:
    ids = [int(row["candidate_index"]) for row in rows]
    if sorted(ids) != sorted(expected_ids) or len(ids) != len(set(ids)):
        raise TaskError("candidate result set is incomplete or duplicated")
    for row in rows:
        _score(row, field)
    return sorted(rows, key=lambda row: (_score(row, field), int(row["candidate_index"])))


class StreamingTop3:
    """Exact bounded top-3 with stable ascending-index tie breaks."""
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any], field: str = "screen_score") -> tuple[list[int], list[int]]:
        index = int(row["candidate_index"])
        if any(int(item["candidate_index"]) == index for item in self.rows):
            raise TaskError("duplicate candidate passed to streaming top-3")
        _score(row, field)
        before = [int(item["candidate_index"]) for item in self.rows]
        self.rows.append(dict(row))
        self.rows.sort(key=lambda item: (_score(item, field), int(item["candidate_index"])))
        self.rows = self.rows[:TOP_K]
        after = [int(item["candidate_index"]) for item in self.rows]
        return after, sorted(set(before) - set(after))


def source_sha256_map() -> dict[str, str]:
    from .upstream_naive import source_hashes as upstream_hashes
    from .signed_bits_observation import digest
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    local = (
        project / "qacpa/same80_bits_r100_core.py",
        project / "qacpa/run_same80_bits_r100.py",
        project / "qacpa/report_same80_bits_r100.py",
        project / "qacpa/signed_bits_observation.py",
        project / "qacpa/signed_bits_randomized.py",
        project / "qacpa/signed_int8_randomized.py",
        project / "qacpa/signed_int8_screen3_fia.py",
        project / "qacpa/staged_fia.py",
        project / "qacpa/upstream_naive.py",
        project / "qacpa/run_m8_fedavg_naive.py",
        project / "qacpa/fia.py",
        project / "qacpa/alignment.py",
        project / "qacpa/seed.py",
        project / "qacpa/deterministic_hybrid_setup.py",
        project / "scripts/run_same80_bits_r100.py",
        project / "configs/same80_bits87654_R100_S5k_K3_F25k.json",
        project / "docs/SAME80_BITS_R100.md",
    )
    result = {path.resolve().as_posix(): digest(path) for path in local}
    result.update({Path(path).resolve().as_posix(): value for path, value in upstream_hashes().items()})
    result.update({
        "runtime::torch": str(torch.__version__),
        "runtime::numpy": str(np.__version__),
    })
    return result


def verify_hash_map(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if name.startswith("runtime::"):
            actual = str(torch.__version__) if name == "runtime::torch" else str(np.__version__)
        else:
            path = Path(name)
            if not path.is_file() or path.is_symlink():
                raise TaskError(f"registered dependency missing or not a regular file: {path}")
            actual = file_sha256(path)
        if actual != expected:
            raise TaskError(f"registered hash changed: {name}")


def code_identity(cfg_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": TASK_ID, "plan_version": PLAN_VERSION,
            "config_sha256": file_sha256(cfg_path), "code_sha256": source_sha256_map(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "bits": list(BITS), "batch_indices": list(range(N_BATCHES)),
            "candidate_ids": list(range(N_CANDIDATES)), "stage1": STAGE1, "fia": FIA,
            "sampler_chunk_rows": SAMPLER_CHUNK_ROWS,
            "seed_formula": "1234+1000003*(100*batch_index+candidate_index)",
            "config_values": cfg}


def verify_identity(saved: dict[str, Any], expected: dict[str, Any]) -> None:
    if saved != expected:
        raise TaskError("task/config/input/code identity mismatch; refusing reuse")
    verify_hash_map(expected["code_sha256"])


def save_marker(path: Path, identity: dict[str, Any], output_names: list[str], **values: Any) -> dict[str, Any]:
    path = Path(path)
    hashes = {}
    for name in output_names:
        output = path.parent / name
        if not output.is_file() or output.is_symlink():
            raise TaskError(f"marker output is missing or not a regular file: {output}")
        hashes[name] = file_sha256(output)
    record = {"identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              "output_hashes": hashes, **values}
    write_json(path, record)
    return record


def verify_marker(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    record = read_json(path)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if record.get("identity_sha256") != digest:
        raise TaskError(f"stage marker identity changed: {path}")
    outputs = record.get("output_hashes")
    if not isinstance(outputs, dict) or not outputs:
        raise TaskError(f"stage marker has no authenticated outputs: {path}")
    for name, expected in outputs.items():
        output = path.parent / name
        if not output.is_file() or output.is_symlink() or file_sha256(output) != expected:
            raise TaskError(f"stage output hash mismatch: {output}")
    return record


def process_start_time_utc(pid: int) -> str:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        from ctypes import wintypes
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong),
                                             ctypes.POINTER(ctypes.c_ulonglong),
                                             ctypes.POINTER(ctypes.c_ulonglong),
                                             ctypes.POINTER(ctypes.c_ulonglong))
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            creation = ctypes.c_ulonglong()
            exit_time = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                            ctypes.byref(kernel), ctypes.byref(user)):
                raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
            unix = (creation.value - 116444736000000000) / 10_000_000
            return datetime.fromtimestamp(unix, timezone.utc).isoformat()
        finally:
            kernel32.CloseHandle(handle)
    stat_tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    boot_time = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                     if line.startswith("btime "))
    seconds = boot_time + int(stat_tail[19]) / hz
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


@contextmanager
def execution_lock(root: Path) -> Iterator[dict[str, Any]]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".execution.lock"
    owner_path = root / ".execution_owner.json"
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        handle.close()
        raise TaskError(f"another worker holds the task lock: {root}") from exc
    if owner_path.is_file():
        try:
            previous = read_json(owner_path)
            previous_start = process_start_time_utc(int(previous["pid"]))
            expected_start = datetime.fromisoformat(str(previous["process_start_time_utc"]))
            actual_start = datetime.fromisoformat(previous_start)
            if abs((actual_start - expected_start).total_seconds()) < 2:
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                raise TaskError(f"owner record identifies a live process without a matching held lock: {previous}")
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            pass
    owner = {"pid": os.getpid(), "process_start_time_utc": process_start_time_utc(os.getpid()),
             "acquired_at_utc": utc_now(), "root": str(root)}
    write_json(owner_path, owner)
    try:
        yield owner
    finally:
        try:
            owner_path.unlink()
        except FileNotFoundError:
            pass
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def assert_safe_task_path(path: Path, root: Path) -> Path:
    root = Path(root).resolve()
    path = Path(os.path.abspath(Path(path)))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise TaskError(f"refusing path outside task root: {path}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise TaskError("refusing symlink in task cleanup")
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise TaskError(f"refusing path outside task root: {resolved}")
    return resolved


def remove_task_disposable(path: Path, root: Path, *, allowed_names: set[str]) -> None:
    resolved = assert_safe_task_path(path, root)
    if resolved.name not in allowed_names:
        raise TaskError(f"cleanup target is not an identified disposable: {resolved.name}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def ensure_free_space(path: Path, *, reserve_bytes: int = MIN_FREE_BYTES) -> int:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < reserve_bytes:
        raise TaskError(f"disk reserve exhausted: free={free}, reserve={reserve_bytes}")
    return free


def expected_budget() -> dict[str, int]:
    cells = len(BITS) * N_BATCHES
    random_cpa = cells * N_CANDIDATES
    naive_cpa = cells
    random_fia = cells * (N_CANDIDATES * PREFIX_STEPS + TOP_K * (TOTAL_STEPS - PREFIX_STEPS))
    naive_fia = cells * TOTAL_STEPS
    return {
        "cells": cells, "random_candidates": random_cpa, "naive_trajectories": naive_cpa,
        "cpa_trajectories": random_cpa + naive_cpa,
        "cpa_steps": (random_cpa + naive_cpa) * STAGE1["n_iter"],
        "random_fia_steps": random_fia, "naive_fia_steps": naive_fia,
        "fia_steps": random_fia + naive_fia, "random_finals": cells * TOP_K,
        "naive_finals": cells,
    }


def verify_gate_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    expected = expected_budget()
    keys = {(int(row["bits"]), int(row["batch_index"])) for row in cells}
    wanted = {(bits, batch) for bits in BITS for batch in range(N_BATCHES)}
    if keys != wanted or len(cells) != expected["cells"]:
        raise TaskError("public gate requires exactly all 25 bit/batch cells")
    candidates = sum(int(row.get("screen_candidates", 0)) for row in cells)
    finals = sum(int(row.get("selected_finals", 0)) for row in cells)
    naive = sum(int(row.get("naive_trajectories", 0)) for row in cells)
    if (candidates, finals, naive) != (expected["random_candidates"], expected["random_finals"], expected["naive_trajectories"]):
        raise TaskError("public gate has missing or duplicate trajectory completions")
    if any(row.get("screen_selection_frozen") is not True or row.get("final_selection_frozen") is not True
           or row.get("private_data_used") is not False for row in cells):
        raise TaskError("selection freeze or public/private boundary gate failed")
    return expected
