"""Isolated signed symmetric INT8 observation codec.

The sender side reads the archived raw update and emits only q/s packets.  The
public reader has no API for raw updates, labels, images, or evaluation truth.
This module intentionally does not import the historical INT4 codec.
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


WIRE_ID = "signed_symmetric_int8_qplus127_nearest_even_cpu_fp32_v1"
MAGIC = b"SGN8V1\0\0"
HEADER = struct.Struct("<8sQf")
QMIN, QMAX = -127, 127
CHECKPOINT_SHA = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
RAW_UPDATES_SHA = "ef9aefc66ab9fcfd0b023fdfc064dbadc27567bd3ef3508bc14ebda9af28ba4c"
ARCHIVED_SENDER_VALIDATION_SHA = "6e78ec5ff2376b01c23a3606fa18b3d9e60d1209f55504211aaf811919d2e244"
EVALUATION_TRUTH_SHA = "724bbacbe316cc28dac086db9cfaf6507f4c1af31144fd610cc480985aab72e2"
FC_SHAPES = {
    "classifier.0.weight": (4096, 25088),
    "classifier.0.bias": (4096,),
    "classifier.3.weight": (4096, 4096),
    "classifier.3.bias": (4096,),
    "classifier.6.weight": (1000, 4096),
    "classifier.6.bias": (1000,),
}
CODEC_METADATA = {
    "codec": WIRE_ID,
    "bits": 8,
    "qmin": QMIN,
    "qmax": QMAX,
    "scale_scope": "per_parameter_tensor_absmax_scalar",
    "input_dtype": "float32",
    "scale_dtype": "float32",
    "code_dtype": "int8",
    "device": "cpu",
    "rounding": "nearest_even",
    "zero_tensor": "q=0,s=1",
    "scale_operation": "float32(max_abs/127)",
    "code_operation": "round(float32(G/s)).clamp(-127,127).int8",
    "edge_policy": "largest_finite_fp32_absmax_rounding_to_observed_scale",
}
CONTEXT = {
    "format": WIRE_ID,
    "codec_metadata": CODEC_METADATA,
    "checkpoint_sha256": CHECKPOINT_SHA,
    "model": "vgg16",
    "image_shape": [16, 3, 224, 224],
    "normalization": "imagenet",
    "training": {"scope": "archived_raw_update_only", "new_local_training": False},
    "observation_source": "single_client_single_local_update_archived_raw_update",
    "batch_id": "batch00",
    "global_round_id": None,
    "client_id": None,
    "id_reason": "NOT_RECORDED",
    "delta_convention": "theta_after10_minus_theta_start",
    "labels": "private",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type != "cpu" or value.dtype != torch.float32
            or value.numel() == 0 or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a nonempty finite CPU FP32 tensor")
    return value.detach().contiguous()


def _require_scale(s: torch.Tensor) -> torch.Tensor:
    if (not isinstance(s, torch.Tensor) or s.device.type != "cpu"
            or s.dtype != torch.float32 or s.ndim != 0 or not bool(torch.isfinite(s))
            or not bool(s > 0)):
        raise ValueError("scale must be a finite positive CPU FP32 scalar")
    return s


def encode(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode with the registered FP32 max_abs/127 and nearest-even rule."""
    G = _require_tensor(G, "G")
    max_abs = G.abs().max()
    s = (max_abs / 127) if bool(max_abs > 0) else torch.ones((), dtype=torch.float32)
    _require_scale(s)
    if bool(max_abs > 0) and bool(s == 0):
        raise ValueError("nonzero tensor scale underflowed to zero")
    q = torch.round(G / s).clamp(QMIN, QMAX).to(torch.int8)
    if bool((q == -128).any()) or not bool(torch.isfinite(q.float()).all()):
        raise ValueError("reserved INT8 code -128 or nonfinite code")
    return q, s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if (not isinstance(q, torch.Tensor) or q.device.type != "cpu"
            or q.dtype != torch.int8 or q.numel() == 0
            or bool((q < QMIN).any()) or bool((q > QMAX).any())):
        raise ValueError("q must be nonempty CPU int8 in [-127,127]")
    _require_scale(s)
    return q.to(torch.float32) * s


def pack(q: torch.Tensor, s: torch.Tensor) -> bytes:
    if (not isinstance(q, torch.Tensor) or q.device.type != "cpu"
            or q.dtype != torch.int8 or q.numel() == 0):
        raise ValueError("codes must be nonempty CPU int8")
    if bool((q < QMIN).any()) or bool((q > QMAX).any()) or bool((q == -128).any()):
        raise ValueError("invalid signed INT8 code")
    _require_scale(s)
    return HEADER.pack(MAGIC, q.numel(), float(s)) + q.contiguous().numpy().tobytes()


def unpack(packet: bytes, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(packet) < HEADER.size:
        raise ValueError("truncated signed INT8 packet")
    magic, count, scale = HEADER.unpack_from(packet)
    expected = int(np.prod(shape))
    if magic != MAGIC or count != expected or not count:
        raise ValueError("wrong INT8 wire ID or shape")
    if len(packet) != HEADER.size + expected:
        raise ValueError("wrong INT8 packet length")
    array = np.frombuffer(packet, dtype=np.int8, offset=HEADER.size, count=expected)
    q = torch.from_numpy(array.copy()).reshape(shape)
    s = torch.tensor(scale, dtype=torch.float32)
    dequantize(q, s)
    return q, s


def _text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_bundle(folder: Path, updates: dict[str, torch.Tensor]) -> dict[str, Any]:
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(folder)
    if set(updates) != set(FC_SHAPES):
        raise ValueError("all six FC tensors are required")
    folder.mkdir(parents=True, exist_ok=False)
    layers: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(sorted(FC_SHAPES)):
        G = _require_tensor(updates[name], name)
        if tuple(G.shape) != FC_SHAPES[name]:
            raise ValueError(f"wrong shape for {name}")
        q, s = encode(G)
        packet = pack(q, s)
        q2, s2 = unpack(packet, tuple(G.shape))
        if not torch.equal(q, q2) or not torch.equal(s, s2):
            raise RuntimeError(f"wire roundtrip failed for {name}")
        filename = f"fc_{index:02d}.sgn8"
        path = folder / filename
        path.write_bytes(packet)
        layers[name] = {
            "shape": list(G.shape), "file": filename,
            "sha256": digest(path), "numel": G.numel(),
            "code_bytes": G.numel(), "scale_bytes": 4,
            "header_bytes_excluding_scale": HEADER.size - 4,
            "file_bytes": len(packet), "q_dtype": "int8", "s_dtype": "float32", "s_shape": [],
        }
    manifest = {**deepcopy(CONTEXT), "layers": layers, "communication": {
        "code_bytes": sum(x["code_bytes"] for x in layers.values()),
        "scale_bytes": 4 * len(layers),
        "header_bytes_excluding_scale": (HEADER.size - 4) * len(layers),
        "manifest_bytes": 0,
        "packet_file_bytes": sum(x["file_bytes"] for x in layers.values()),
        "total_file_bytes": 0,
    }}
    for _ in range(16):
        size = len(_text(manifest).encode("utf-8"))
        counts = manifest["communication"]
        if counts["manifest_bytes"] == size:
            break
        counts["manifest_bytes"] = size
        counts["total_file_bytes"] = size + counts["packet_file_bytes"]
    else:
        raise RuntimeError("manifest accounting did not converge")
    (folder / "manifest.json").write_text(_text(manifest), encoding="utf-8", newline="\n")
    return manifest


def read_bundle(folder: Path) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != set(CONTEXT) | {"layers", "communication"}:
        raise ValueError("unexpected INT8 public manifest fields")
    for key, expected in CONTEXT.items():
        if manifest[key] != expected:
            raise ValueError(f"public INT8 metadata mismatch: {key}")
    if set(manifest["layers"]) != set(FC_SHAPES):
        raise ValueError("public INT8 manifest must contain all six FC tensors")
    entries = {}
    for index, name in enumerate(sorted(FC_SHAPES)):
        layer = manifest["layers"][name]
        expected_keys = {"shape", "file", "sha256", "numel", "code_bytes", "scale_bytes",
                         "header_bytes_excluding_scale", "file_bytes", "q_dtype", "s_dtype", "s_shape"}
        if set(layer) != expected_keys or layer["file"] != f"fc_{index:02d}.sgn8":
            raise ValueError(f"invalid INT8 layer descriptor: {name}")
        shape = FC_SHAPES[name]
        expected = {"shape": list(shape), "numel": int(np.prod(shape)),
                    "code_bytes": int(np.prod(shape)), "scale_bytes": 4,
                    "header_bytes_excluding_scale": HEADER.size - 4,
                    "file_bytes": HEADER.size + int(np.prod(shape)),
                    "q_dtype": "int8", "s_dtype": "float32", "s_shape": []}
        if any(layer[k] != v for k, v in expected.items()):
            raise ValueError(f"invalid INT8 layer accounting: {name}")
        path = folder / layer["file"]
        if path.is_symlink() or path.resolve().parent != folder or digest(path) != layer["sha256"]:
            raise ValueError(f"INT8 packet integrity/path failure: {name}")
        entries[name] = unpack(path.read_bytes(), shape)
    counts = {"code_bytes": sum(x["code_bytes"] for x in manifest["layers"].values()),
              "scale_bytes": 4 * len(entries),
              "header_bytes_excluding_scale": (HEADER.size - 4) * len(entries),
              "manifest_bytes": manifest_path.stat().st_size,
              "packet_file_bytes": sum(x["file_bytes"] for x in manifest["layers"].values())}
    counts["total_file_bytes"] = counts["manifest_bytes"] + counts["packet_file_bytes"]
    if manifest["communication"] != counts:
        raise ValueError("INT8 communication accounting mismatch")
    return manifest, entries


def collect(raw_updates: Path, archived_sender_validation: Path, evaluation_truth: Path,
            output: Path, private_output: Path) -> dict[str, Any]:
    """Sender-only conversion of the already archived update."""
    raw_updates, archived_sender_validation = Path(raw_updates).resolve(), Path(archived_sender_validation).resolve()
    evaluation_truth, output, private_output = (Path(x).resolve() for x in (evaluation_truth, output, private_output))
    if digest(raw_updates) != RAW_UPDATES_SHA or digest(archived_sender_validation) != ARCHIVED_SENDER_VALIDATION_SHA:
        raise ValueError("archived sender inputs failed their registered hashes")
    sender_record = json.loads(archived_sender_validation.read_text(encoding="utf-8"))
    if sender_record.get("status") != "COMPLETED" or sender_record.get("raw_updates_sha256") != RAW_UPDATES_SHA:
        raise ValueError("archived sender validation is not canonical")
    if digest(evaluation_truth) != EVALUATION_TRUTH_SHA:
        raise ValueError("evaluation truth hash changed before sender phase")
    if output.exists() or private_output.exists() or output.resolve() == private_output.resolve():
        raise FileExistsError("INT8 sender outputs must be new and isolated")
    bundle = torch.load(raw_updates, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or set(bundle) != {"updates", "delta_convention", "checkpoint_sha256"}:
        raise ValueError("raw update bundle schema changed")
    if bundle["checkpoint_sha256"] != CHECKPOINT_SHA or bundle["delta_convention"] != "theta_after10_minus_theta_start":
        raise ValueError("raw update provenance mismatch")
    updates = bundle["updates"]
    if not isinstance(updates, dict) or set(updates) != set(FC_SHAPES):
        raise ValueError("raw update bundle must contain exactly six FC tensors")
    updates = {name: _require_tensor(updates[name], name) for name in FC_SHAPES}
    for name, shape in FC_SHAPES.items():
        if tuple(updates[name].shape) != shape:
            raise ValueError(f"raw update shape mismatch: {name}")
    manifest = write_bundle(output, updates)
    read_bundle(output)
    private_output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "COMPLETED", "stage": "SENDER", "format": WIRE_ID,
        "raw_updates_sha256": digest(raw_updates),
        "archived_sender_validation_sha256": digest(archived_sender_validation),
        "evaluation_truth_sha256": digest(evaluation_truth),
        "public_manifest_sha256": digest(output / "manifest.json"),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "true_updates_consumed": True, "labels_consumed": False, "images_consumed": False,
        "raw_updates_saved_in_public": False, "raw_features_saved": False,
        "public_output": output.as_posix(), "private_provenance_output": private_output.as_posix(),
        "codec": CODEC_METADATA, "layers": manifest["layers"],
    }
    (private_output / "sender_validation.json").write_text(_text(report), encoding="utf-8", newline="\n")
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sender = sub.add_parser("sender")
    sender.add_argument("--raw-updates", type=Path, required=True)
    sender.add_argument("--archived-sender-validation", type=Path, required=True)
    sender.add_argument("--evaluation-truth", type=Path, required=True)
    sender.add_argument("--output", type=Path, required=True)
    sender.add_argument("--private-output", type=Path, required=True)
    reader = sub.add_parser("read")
    reader.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sender":
        # ``command`` is the subparser selector, not part of collect()'s
        # authenticated sender contract.  Passing vars(args) used to fail
        # before the sender read its inputs.
        result = collect(
            raw_updates=args.raw_updates,
            archived_sender_validation=args.archived_sender_validation,
            evaluation_truth=args.evaluation_truth,
            output=args.output,
            private_output=args.private_output,
        )
    else:
        manifest, entries = read_bundle(args.folder)
        result = {"status": "PASSED", "manifest": manifest, "layers": sorted(entries)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
