"""Isolated signed INT4 row-wise codec and public randomized sampler.

The wire format deliberately stores logical INT4 codes in native int8 bytes so
the transport accounting cannot be confused with packed four-bit bandwidth.
Every FC weight output row owns one FP32 scale; each bias vector owns one
scale.  This module has no private-data or model interface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import torch

from .signed_bits_observation import (
    ARCHIVED_SENDER_VALIDATION, CHECKPOINT_SHA, FC_SHAPES, EVALUATION_TRUTH,
    RAW_UPDATES, ARCHIVED_SENDER_VALIDATION_SHA, EVALUATION_TRUTH_SHA,
    RAW_UPDATES_SHA, digest as file_digest,
)

BITS = 4
QMAX = 7
MAGIC = b"SBRW4V1\0"
HEADER = struct.Struct("<8sIIIII")
CODEC = "signed_symmetric_int4_rowwise_nearest_even_cpu_fp32_v1"
WEIGHT_NAMES = tuple(name for name in FC_SHAPES if name.endswith(".weight"))
FC_NAMES = tuple(sorted(FC_SHAPES))
SCALE_COUNT = sum(FC_SHAPES[name][0] for name in WEIGHT_NAMES) + sum(1 for name in FC_SHAPES if name.endswith(".bias"))


class RowwiseCodecError(ValueError):
    pass


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    array = value.numpy()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode("ascii")); h.update(str(array.shape).encode("ascii"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _tensor(value: Any, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype != torch.float32 or value.layout != torch.strided or
            tuple(value.shape) != shape or value.numel() == 0 or
            not bool(torch.isfinite(value).all())):
        raise RowwiseCodecError(f"{name} must be finite CPU FP32 {shape}")
    return value.detach().contiguous()


def encode_tensor(value: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    value = _tensor(value, name, FC_SHAPES[name])
    is_weight = name.endswith(".weight")
    rows = value.shape[0] if is_weight else 1
    matrix = value.reshape(rows, -1)
    maximum = matrix.abs().amax(dim=1)
    scales = torch.where(maximum > 0, maximum / QMAX, torch.ones_like(maximum)).to(torch.float32)
    if bool(((maximum > 0) & (scales == 0)).any()):
        raise RowwiseCodecError(f"scale underflow in {name}")
    q = torch.round(matrix / scales[:, None]).clamp(-QMAX, QMAX).to(torch.int8).reshape(value.shape)
    zero = maximum == 0
    if bool((q.reshape(rows, -1)[zero] != 0).any()):
        raise RowwiseCodecError(f"zero row code mismatch in {name}")
    return q.contiguous(), scales.contiguous(), {
        "shape": list(value.shape), "rows": rows, "columns": matrix.shape[1],
        "scale_count": rows, "grouping": "axis0_output_rows" if is_weight else "one_group_for_bias_vector",
        "zero_rows": int(zero.sum()), "scale_dtype": "float32", "code_dtype": "int8",
        "logical_code_bits": 4, "rounding": "nearest_even", "scale_operation": "float32(max_abs_row/7)",
    }


def decode_tensor(q: torch.Tensor, scales: torch.Tensor, name: str) -> torch.Tensor:
    q = q.detach().cpu().contiguous(); scales = scales.detach().cpu().contiguous()
    expected = FC_SHAPES[name]; rows = expected[0] if name.endswith(".weight") else 1
    if q.dtype != torch.int8 or tuple(q.shape) != expected or scales.dtype != torch.float32 or tuple(scales.shape) != (rows,):
        raise RowwiseCodecError(f"invalid row-wise packet tensor schema for {name}")
    if bool((q < -QMAX).any()) or bool((q > QMAX).any()) or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise RowwiseCodecError(f"invalid row-wise code/scale for {name}")
    return (q.reshape(rows, -1).float() * scales[:, None]).reshape(expected).contiguous()


def pack(q: torch.Tensor, scales: torch.Tensor, name: str) -> bytes:
    q = q.detach().cpu().contiguous(); scales = scales.detach().cpu().contiguous()
    rows, cols = (q.shape[0], q[0].numel()) if name.endswith(".weight") else (1, q.numel())
    qflat = q.reshape(-1)
    if q.dtype != torch.int8 or bool((qflat < -QMAX).any()) or bool((qflat > QMAX).any()):
        raise RowwiseCodecError("codes outside signed INT4 range")
    if scales.dtype != torch.float32 or tuple(scales.shape) != (rows,):
        raise RowwiseCodecError("wrong row-scale shape")
    return HEADER.pack(MAGIC, qflat.numel(), rows, cols, scales.numel(), 0) + qflat.numpy().tobytes() + scales.numpy().tobytes()


def unpack(packet: bytes, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    expected = FC_SHAPES[name]; rows = expected[0] if name.endswith(".weight") else 1; cols = int(torch.tensor(expected).prod().item() // rows)
    if len(packet) < HEADER.size:
        raise RowwiseCodecError("truncated row-wise packet")
    magic, count, got_rows, got_cols, scale_count, reserved = HEADER.unpack_from(packet)
    if (magic != MAGIC or reserved != 0 or count != int(torch.tensor(expected).prod().item()) or got_rows != rows or
            got_cols != cols or scale_count != rows or len(packet) != HEADER.size + count + 4 * rows):
        raise RowwiseCodecError(f"row-wise packet header mismatch for {name}")
    q = torch.frombuffer(bytearray(packet[HEADER.size:HEADER.size + count]), dtype=torch.int8).clone().reshape(expected)
    scales = torch.frombuffer(bytearray(packet[HEADER.size + count:]), dtype=torch.float32).clone()
    decode_tensor(q, scales, name)
    return q, scales


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def write_bundle(folder: Path, updates: dict[str, torch.Tensor]) -> dict[str, Any]:
    folder = Path(folder).resolve()
    if folder.exists(): raise FileExistsError(folder)
    if set(updates) != set(FC_SHAPES): raise RowwiseCodecError("all six FC tensors are required")
    folder.mkdir(parents=True)
    layers: dict[str, Any] = {}
    for index, name in enumerate(FC_NAMES):
        q, scales, info = encode_tensor(updates[name], name)
        packet = pack(q, scales, name); path = folder / f"fc_{index:02d}.rw4"; path.write_bytes(packet)
        q2, s2 = unpack(packet, name)
        if not torch.equal(q, q2) or not torch.equal(scales, s2): raise RowwiseCodecError(f"wire roundtrip failed {name}")
        layers[name] = {**info, "file": path.name, "sha256": file_digest(path), "numel": q.numel(),
                        "code_bytes": q.numel(), "scale_bytes": scales.numel() * 4, "header_bytes": HEADER.size,
                        "file_bytes": len(packet), "q_dtype": "int8", "physical_code_storage": "int8"}
    comm = {"logical_code_bits": 4, "code_bytes": sum(x["code_bytes"] for x in layers.values()),
            "physical_code_storage_bytes": sum(x["code_bytes"] for x in layers.values()),
            "scale_bytes": sum(x["scale_bytes"] for x in layers.values()),
            "header_bytes": sum(x["header_bytes"] for x in layers.values()), "manifest_bytes": 0,
            "packet_file_bytes": sum(x["file_bytes"] for x in layers.values()), "total_file_bytes": 0}
    manifest = {"format": CODEC, "bits": 4, "qmin": -7, "qmax": 7, "checkpoint_sha256": CHECKPOINT_SHA,
                "model": "vgg16", "delta_convention": "theta_after10_minus_theta_start", "layers": layers,
                "communication": comm, "row_scale_count": SCALE_COUNT, "row_scale_bytes": SCALE_COUNT * 4,
                "scale_grouping": "weights_axis0_rows_bias_vector_one_group", "rounding": "nearest_even",
                "zero_convention": "q=0,s=1", "device": "cpu", "source": "same_archived_raw_update"}
    for _ in range(16):
        size = len(_json(manifest)); total = size + comm["packet_file_bytes"]
        if comm["manifest_bytes"] == size and comm["total_file_bytes"] == total: break
        comm["manifest_bytes"], comm["total_file_bytes"] = size, total
    (folder / "manifest.json").write_bytes(_json(manifest))
    return manifest


def read_bundle(folder: Path) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    folder = Path(folder).resolve(); manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CODEC or manifest.get("bits") != 4 or set(manifest.get("layers", {})) != set(FC_SHAPES):
        raise RowwiseCodecError("row-wise manifest identity/schema mismatch")
    if manifest.get("row_scale_count") != SCALE_COUNT or manifest.get("row_scale_bytes") != SCALE_COUNT * 4:
        raise RowwiseCodecError("row-wise scale accounting mismatch")
    entries = {}
    for name in FC_NAMES:
        desc = manifest["layers"][name]; path = folder / desc["file"]
        if path.resolve().parent != folder or file_digest(path) != desc["sha256"]:
            raise RowwiseCodecError(f"row-wise packet hash/path mismatch {name}")
        q, scales = unpack(path.read_bytes(), name)
        if tensor_digest(decode_tensor(q, scales, name)) != desc.get("decoded_sha256", tensor_digest(decode_tensor(q, scales, name))):
            raise RowwiseCodecError(f"decoded row-wise packet mismatch {name}")
        entries[name] = (q, scales)
    expected = sum(x["file_bytes"] for x in manifest["layers"].values()) + manifest_path.stat().st_size
    if manifest["communication"].get("total_file_bytes") != expected: raise RowwiseCodecError("byte accounting mismatch")
    return manifest, entries


def _finite_cap(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(torch.float32).cpu(); lo, hi = 0, 0x7F7FFFFF + 1
    while lo < hi:
        mid = (lo + hi) // 2
        value = torch.tensor(mid, dtype=torch.int32).view(torch.float32)
        if bool(value / QMAX <= scale): lo = mid + 1
        else: hi = mid
    cap = torch.tensor(lo - 1, dtype=torch.int32).view(torch.float32)
    if not torch.equal(cap / QMAX, scale): raise RowwiseCodecError("scale has no finite FP32 cap")
    return cap


def scalar_cell_bounds(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact scalar reference bounds, preserving nearest-even ownership."""
    scale = scale.to(torch.float32).cpu(); cap = _finite_cap(scale)
    vals = torch.arange(-QMAX, QMAX + 1, dtype=torch.int8); lower=[]; upper=[]
    finite = torch.arange(0, 0x7F7FFFFF + 1, dtype=torch.int64)
    # The reference uses binary search in float bit space for each positive code.
    for code in range(1, QMAX + 1):
        lo, hi = 0, 0x7F7FFFFF + 1
        while lo < hi:
            mid = (lo + hi) // 2; v = torch.tensor(mid, dtype=torch.int32).view(torch.float32)
            if int(torch.round(v / scale).clamp(-QMAX, QMAX)) < code: lo = mid + 1
            else: hi = mid
        lower.append(torch.tensor(lo, dtype=torch.int32).view(torch.float32))
    pos_lo = torch.stack(lower); pos_pre = torch.nextafter(pos_lo, torch.zeros_like(pos_lo))
    lower = torch.cat((-torch.cat((pos_pre[1:], cap.reshape(1))).flip(0), -pos_pre[:1], pos_lo))
    upper = torch.cat((-pos_lo.flip(0), pos_pre[:1], pos_pre[1:], cap.reshape(1)))
    if not torch.equal(torch.round(lower / scale).clamp(-QMAX,QMAX).to(torch.int8), vals): raise RowwiseCodecError("lower ownership")
    if not torch.equal(torch.round(upper / scale).clamp(-QMAX,QMAX).to(torch.int8), vals): raise RowwiseCodecError("upper ownership")
    return lower, upper


def batched_cell_bounds(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scales.dtype != torch.float32 or scales.ndim != 1 or bool((scales <= 0).any()): raise RowwiseCodecError("bad scale batch")
    pairs = [scalar_cell_bounds(s) for s in scales]
    return torch.stack([x[0] for x in pairs]), torch.stack([x[1] for x in pairs])


def sample_matrix(q: torch.Tensor, scales: torch.Tensor, *, seed: int, chunk_rows: int = 32) -> tuple[torch.Tensor, dict[str, Any]]:
    if q.dtype != torch.int8 or q.ndim != 2 or scales.dtype != torch.float32 or tuple(scales.shape) != (q.shape[0],):
        raise RowwiseCodecError("row-wise sampler expects q[M,D] and scales[M]")
    if bool((q < -QMAX).any()) or bool((q > QMAX).any()): raise RowwiseCodecError("q range")
    lower, upper = batched_cell_bounds(scales); generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.empty(q.shape, dtype=torch.float32)
    for start in range(0, q.shape[0], chunk_rows):
        stop = min(start + chunk_rows, q.shape[0]); ids = q[start:stop].long() + QMAX
        noise = torch.rand((stop-start, q.shape[1]), dtype=torch.float64, generator=generator)
        result[start:stop] = (lower[start:stop][ids].double() + noise * (upper[start:stop][ids].double()-lower[start:stop][ids])).float()
    anchors=[]
    for row in range(q.shape[0]):
        if bool((q[row] == 0).all()):
            result[row].zero_(); continue
        locations = torch.nonzero(q[row].abs() == QMAX, as_tuple=False)
        if locations.numel() == 0: raise RowwiseCodecError(f"nonzero row {row} lacks extremal code")
        col = int(locations[0]); cap = _finite_cap(scales[row]); result[row, col] = torch.copysign(cap, q[row, col].float())
        anchors.append({"row": row, "column": col, "q": int(q[row,col]), "cap": float(cap)})
    fixed = torch.round(result / scales[:,None]).clamp(-QMAX,QMAX).to(torch.int8)
    if not torch.equal(fixed, q) or not bool(torch.isfinite(result).all()): raise RowwiseCodecError("sample code mismatch")
    for row in range(q.shape[0]):
        if bool((q[row] == 0).all()):
            if bool((result[row] != 0).any()): raise RowwiseCodecError("zero row changed")
        elif not torch.equal(result[row].abs().max() / QMAX, scales[row]): raise RowwiseCodecError("row scale mismatch")
    return result.contiguous(), {"seed": seed, "chunk_rows": chunk_rows, "generator": "torch.Generator(cpu)",
        "sampler": "uniform_fp64_interpolation_cast_fp32_per_row_cells", "anchor_count": len(anchors), "anchors": anchors,
        "row_scale_count": int(scales.numel()), "code_consistency": True, "tensor_sha256": tensor_digest(result)}


def collect_rowwise(output: Path, private_output: Path) -> dict[str, Any]:
    output, private_output = Path(output).resolve(), Path(private_output).resolve()
    if file_digest(RAW_UPDATES) != RAW_UPDATES_SHA or file_digest(ARCHIVED_SENDER_VALIDATION) != ARCHIVED_SENDER_VALIDATION_SHA:
        raise RowwiseCodecError("archived sender inputs changed")
    if file_digest(EVALUATION_TRUTH) != EVALUATION_TRUTH_SHA: raise RowwiseCodecError("private truth changed before sender")
    bundle = torch.load(RAW_UPDATES, map_location="cpu", weights_only=False)
    if set(bundle) != {"updates", "delta_convention", "checkpoint_sha256"} or bundle["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RowwiseCodecError("raw update provenance mismatch")
    manifest = write_bundle(output, bundle["updates"]); read_bundle(output)
    report = {"status":"COMPLETED", "stage":"SENDER", "codec":CODEC, "bits":4,
              "raw_updates_sha256":RAW_UPDATES_SHA, "archived_sender_validation_sha256":ARCHIVED_SENDER_VALIDATION_SHA,
              "evaluation_truth_sha256":EVALUATION_TRUTH_SHA, "public_manifest_sha256":file_digest(output/"manifest.json"),
              "true_updates_consumed":True, "labels_consumed":False, "images_consumed":False,
              "raw_updates_saved_in_public":False, "row_scale_bytes":SCALE_COUNT*4, "scale_count":SCALE_COUNT,
              "physical_code_storage":"int8", "public_output":output.as_posix(), "private_provenance_output":private_output.as_posix(),
              "communication":manifest["communication"]}
    private_output.mkdir(parents=True, exist_ok=False); (private_output/"sender_validation.json").write_bytes(_json(report)); return report
