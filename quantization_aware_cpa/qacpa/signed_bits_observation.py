"""Bit-parameterized public signed observation codec for the 7/6/5/4 sweep.

Codes are stored in int8 tensors for a stable native transport.  The manifest
records logical bit width separately from physical packet bytes; int8 storage
is never reported as packed b-bit bandwidth.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch


SUPPORTED_BITS = (7, 6, 5, 4)
FC_SHAPES = {
    "classifier.0.weight": (4096, 25088),
    "classifier.0.bias": (4096,),
    "classifier.3.weight": (4096, 4096),
    "classifier.3.bias": (4096,),
    "classifier.6.weight": (1000, 4096),
    "classifier.6.bias": (1000,),
}
CHECKPOINT_SHA = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
RAW_UPDATES_SHA = "ef9aefc66ab9fcfd0b023fdfc064dbadc27567bd3ef3508bc14ebda9af28ba4c"
ARCHIVED_SENDER_VALIDATION_SHA = "6e78ec5ff2376b01c23a3606fa18b3d9e60d1209f55504211aaf811919d2e244"
EVALUATION_TRUTH_SHA = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"
RAW_UPDATES = Path(__file__).resolve().parents[1] / "results_private/signed_int4_baseline_B16_T10/observation/raw_updates.pt"
ARCHIVED_SENDER_VALIDATION = RAW_UPDATES.parent / "sender_validation.json"
EVALUATION_TRUTH = RAW_UPDATES.parent / "evaluation_truth.pt"
HEADER = struct.Struct("<8sQf")
_MAX_INT8 = 127


def qmax_for_bits(bits: int) -> int:
    if type(bits) is not int or bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    return 2 ** (bits - 1) - 1


def wire_id(bits: int) -> str:
    qmax_for_bits(bits)
    return f"signed_symmetric_bits{bits}_qplus{qmax_for_bits(bits)}_nearest_even_cpu_fp32_v1"


def _magic(bits: int) -> bytes:
    return f"SB{bits:02d}V1".encode("ascii").ljust(8, b"\0")


def codec_metadata(bits: int) -> dict[str, Any]:
    qmax = qmax_for_bits(bits)
    return {
        "codec": wire_id(bits), "bits": bits, "qmin": -qmax, "qmax": qmax,
        "scale_scope": "per_parameter_tensor_absmax_scalar", "input_dtype": "float32",
        "scale_dtype": "float32", "code_dtype": "int8", "physical_code_storage": "int8",
        "logical_code_bits": bits, "device": "cpu", "rounding": "nearest_even",
        "zero_tensor": "q=0,s=1", "scale_operation": f"float32(max_abs/{qmax})",
        "code_operation": f"round(float32(G/s)).clamp(-{qmax},{qmax}).int8",
        "edge_policy": "largest_finite_fp32_absmax_rounding_to_observed_scale",
    }


def context(bits: int) -> dict[str, Any]:
    return {
        "format": wire_id(bits), "codec_metadata": codec_metadata(bits), "checkpoint_sha256": CHECKPOINT_SHA,
        "model": "vgg16", "image_shape": [16, 3, 224, 224], "normalization": "imagenet",
        "training": {"scope": "archived_raw_update_only", "new_local_training": False},
        "observation_source": "single_client_single_local_update_archived_raw_update",
        "batch_id": "batch00", "global_round_id": None, "client_id": None, "id_reason": "NOT_RECORDED",
        "delta_convention": "theta_after10_minus_theta_start", "labels": "private",
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tensor(value: Any, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() == 0
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a nonempty finite CPU FP32 tensor")
    return value.detach().contiguous()


def _scale(value: torch.Tensor) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
            or value.ndim != 0 or not bool(torch.isfinite(value)) or not bool(value > 0)):
        raise ValueError("scale must be a finite positive CPU FP32 scalar")
    return value


def encode(value: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = _tensor(value, "G")
    qmax = qmax_for_bits(bits)
    maximum = value.abs().max()
    scale = maximum / qmax if bool(maximum > 0) else torch.ones((), dtype=torch.float32)
    _scale(scale)
    if bool(maximum > 0) and bool(scale == 0):
        raise ValueError("nonzero tensor scale underflowed to zero")
    q = torch.round(value / scale).clamp(-qmax, qmax).to(torch.int8)
    if bool((q < -qmax).any()) or bool((q > qmax).any()) or bool((q == -128).any()):
        raise ValueError("reserved or out-of-range code")
    return q.contiguous(), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = qmax_for_bits(bits)
    if (not isinstance(q, torch.Tensor) or q.device.type != "cpu" or q.dtype != torch.int8
            or q.numel() == 0 or bool((q < -qmax).any()) or bool((q > qmax).any())):
        raise ValueError("q must be a nonempty CPU int8 tensor within the registered range")
    _scale(scale)
    return q.to(torch.float32) * scale


def pack(q: torch.Tensor, scale: torch.Tensor, bits: int) -> bytes:
    qmax_for_bits(bits)
    if (not isinstance(q, torch.Tensor) or q.device.type != "cpu" or q.dtype != torch.int8
            or q.numel() == 0 or bool((q < -qmax_for_bits(bits)).any())
            or bool((q > qmax_for_bits(bits)).any()) or bool((q == -128).any())):
        raise ValueError("invalid signed bit-sweep code tensor")
    _scale(scale)
    return HEADER.pack(_magic(bits), q.numel(), float(scale)) + q.contiguous().numpy().tobytes()


def unpack(packet: bytes, shape: tuple[int, ...], bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax_for_bits(bits)
    if len(packet) < HEADER.size:
        raise ValueError("truncated bit-sweep packet")
    magic, count, raw_scale = HEADER.unpack_from(packet)
    expected = int(np.prod(shape))
    if magic != _magic(bits) or count != expected or not count or len(packet) != HEADER.size + expected:
        raise ValueError("wrong bit-sweep wire ID, shape or packet length")
    q = torch.from_numpy(np.frombuffer(packet, dtype=np.int8, offset=HEADER.size, count=expected).copy()).reshape(shape)
    scale = torch.tensor(raw_scale, dtype=torch.float32)
    dequantize(q, scale, bits)
    return q, scale


def _text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_bundle(folder: Path, updates: dict[str, torch.Tensor], bits: int) -> dict[str, Any]:
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(folder)
    qmax_for_bits(bits)
    if set(updates) != set(FC_SHAPES):
        raise ValueError("all six FC tensors are required")
    folder.mkdir(parents=True, exist_ok=False)
    layers: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(sorted(FC_SHAPES)):
        value = _tensor(updates[name], name)
        if tuple(value.shape) != FC_SHAPES[name]:
            raise ValueError(f"wrong shape for {name}")
        q, scale = encode(value, bits)
        packet = pack(q, scale, bits)
        q2, scale2 = unpack(packet, tuple(value.shape), bits)
        if not torch.equal(q, q2) or not torch.equal(scale, scale2):
            raise RuntimeError(f"wire roundtrip failed for {name}")
        path = folder / f"fc_{index:02d}.sb{bits}"
        path.write_bytes(packet)
        layers[name] = {
            "shape": list(value.shape), "file": path.name, "sha256": digest(path), "numel": value.numel(),
            "code_bytes": value.numel(), "physical_code_storage": "int8", "logical_code_bits": bits,
            "scale_bytes": 4, "header_bytes_excluding_scale": HEADER.size - 4, "file_bytes": len(packet),
            "q_dtype": "int8", "s_dtype": "float32", "s_shape": [],
        }
    communication = {
        "logical_bits": bits, "code_bytes": sum(x["code_bytes"] for x in layers.values()),
        "logical_code_bits": bits * sum(x["numel"] for x in layers.values()),
        "physical_code_storage_bytes": sum(x["code_bytes"] for x in layers.values()),
        "scale_bytes": 4 * len(layers), "header_bytes_excluding_scale": (HEADER.size - 4) * len(layers),
        "manifest_bytes": 0, "packet_file_bytes": sum(x["file_bytes"] for x in layers.values()), "total_file_bytes": 0,
    }
    manifest = {**deepcopy(context(bits)), "layers": layers, "communication": communication}
    for _ in range(16):
        size = len(_text(manifest).encode("utf-8"))
        if communication["manifest_bytes"] == size:
            break
        communication["manifest_bytes"] = size
        communication["total_file_bytes"] = size + communication["packet_file_bytes"]
    else:
        raise RuntimeError("manifest accounting did not converge")
    (folder / "manifest.json").write_text(_text(manifest), encoding="utf-8", newline="\n")
    return manifest


def read_bundle(folder: Path, bits: int) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != set(context(bits)) | {"layers", "communication"} or manifest["format"] != wire_id(bits):
        raise ValueError("unexpected bit-sweep manifest schema")
    if manifest["codec_metadata"] != codec_metadata(bits) or manifest["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("bit-sweep codec/provenance mismatch")
    if set(manifest["layers"]) != set(FC_SHAPES):
        raise ValueError("bit-sweep manifest must contain all six FC tensors")
    entries: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, name in enumerate(sorted(FC_SHAPES)):
        layer = manifest["layers"][name]
        expected_shape = list(FC_SHAPES[name])
        expected_keys = {"shape", "file", "sha256", "numel", "code_bytes", "physical_code_storage",
                         "logical_code_bits", "scale_bytes", "header_bytes_excluding_scale", "file_bytes",
                         "q_dtype", "s_dtype", "s_shape"}
        if set(layer) != expected_keys or layer["file"] != f"fc_{index:02d}.sb{bits}" or layer["shape"] != expected_shape:
            raise ValueError(f"invalid bit-sweep layer descriptor: {name}")
        path = folder / layer["file"]
        if path.is_symlink() or path.resolve().parent != folder or digest(path) != layer["sha256"]:
            raise ValueError(f"bit-sweep packet integrity/path failure: {name}")
        entries[name] = unpack(path.read_bytes(), FC_SHAPES[name], bits)
    communication = manifest["communication"]
    expected = {
        "logical_bits": bits, "code_bytes": sum(x["code_bytes"] for x in manifest["layers"].values()),
        "logical_code_bits": bits * sum(x["numel"] for x in manifest["layers"].values()),
        "physical_code_storage_bytes": sum(x["code_bytes"] for x in manifest["layers"].values()),
        "scale_bytes": 4 * len(entries), "header_bytes_excluding_scale": (HEADER.size - 4) * len(entries),
        "manifest_bytes": manifest_path.stat().st_size,
        "packet_file_bytes": sum(x["file_bytes"] for x in manifest["layers"].values()),
    }
    expected["total_file_bytes"] = expected["manifest_bytes"] + expected["packet_file_bytes"]
    if communication != expected:
        raise ValueError("bit-sweep communication accounting mismatch")
    return manifest, entries


def collect(bits: int, raw_updates: Path = RAW_UPDATES,
            archived_sender_validation: Path = ARCHIVED_SENDER_VALIDATION,
            evaluation_truth: Path = EVALUATION_TRUTH,
            output: Path | None = None, private_output: Path | None = None) -> dict[str, Any]:
    qmax_for_bits(bits)
    raw_updates, archived_sender_validation, evaluation_truth = map(Path, (raw_updates, archived_sender_validation, evaluation_truth))
    output, private_output = Path(output), Path(private_output)
    if digest(raw_updates) != RAW_UPDATES_SHA or digest(archived_sender_validation) != ARCHIVED_SENDER_VALIDATION_SHA:
        raise ValueError("archived sender inputs failed registered hashes")
    sender_record = json.loads(archived_sender_validation.read_text(encoding="utf-8"))
    if sender_record.get("status") != "COMPLETED" or sender_record.get("raw_updates_sha256") != RAW_UPDATES_SHA:
        raise ValueError("archived sender validation is not canonical")
    if digest(evaluation_truth) != EVALUATION_TRUTH_SHA:
        raise ValueError("evaluation truth hash changed before sender phase")
    if output.exists() or private_output.exists() or output.resolve() == private_output.resolve():
        raise FileExistsError("bit-sweep sender outputs must be new and isolated")
    bundle = torch.load(raw_updates, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or set(bundle) != {"updates", "delta_convention", "checkpoint_sha256"}:
        raise ValueError("raw update bundle schema changed")
    if bundle["checkpoint_sha256"] != CHECKPOINT_SHA or bundle["delta_convention"] != "theta_after10_minus_theta_start":
        raise ValueError("raw update provenance mismatch")
    updates = bundle["updates"]
    if not isinstance(updates, dict) or set(updates) != set(FC_SHAPES):
        raise ValueError("raw update bundle must contain exactly six FC tensors")
    updates = {name: _tensor(updates[name], name) for name in FC_SHAPES}
    for name, shape in FC_SHAPES.items():
        if tuple(updates[name].shape) != shape:
            raise ValueError(f"raw update shape mismatch: {name}")
    manifest = write_bundle(output, updates, bits)
    read_bundle(output, bits)
    private_output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "COMPLETED", "stage": "SENDER", "bits": bits, "codec": codec_metadata(bits),
        "raw_updates_sha256": digest(raw_updates), "archived_sender_validation_sha256": digest(archived_sender_validation),
        "evaluation_truth_sha256": digest(evaluation_truth), "public_manifest_sha256": digest(output / "manifest.json"),
        "checkpoint_sha256": CHECKPOINT_SHA, "true_updates_consumed": True, "labels_consumed": False,
        "images_consumed": False, "raw_updates_saved_in_public": False, "raw_features_saved": False,
        "public_output": output.resolve().as_posix(), "private_provenance_output": private_output.resolve().as_posix(),
        "layers": manifest["layers"], "logical_bits": bits, "physical_code_storage": "int8",
    }
    (private_output / "sender_validation.json").write_text(_text(report), encoding="utf-8", newline="\n")
    return report
