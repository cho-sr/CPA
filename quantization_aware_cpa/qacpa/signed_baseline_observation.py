"""New signed int4 wire format; public q/s only, no sender truth or raw updates."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from quantized_update_cpa.cell_utils import CODEC_METADATA, dequantize, encode


WIRE_ID = "signed_symmetric_int4_qplus7_packed_cpu_fp32_v1"
MAGIC = b"SGN4V1\0\0"
HEADER = struct.Struct("<8sQf")
CHECKPOINT_SHA = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
TRUTH_SHA = "9430257f1142c81f12a6610bb13f6458c50c26221538019e4140246367c26eb4"
FC_SHAPES = {"classifier.0.weight": (4096, 25088), "classifier.0.bias": (4096,),
             "classifier.3.weight": (4096, 4096), "classifier.3.bias": (4096,),
             "classifier.6.weight": (1000, 4096), "classifier.6.bias": (1000,)}
TRAINING = {"scope": "conv_frozen_all_six_fc_parameters", "steps": 10, "optimizer": "sgd",
            "lr": .001, "momentum": 0., "weight_decay": 0., "foreach": False,
            "cross_entropy_reduction": "mean", "seed": 42, "dropout": 0., "model_eval": True,
            "batch_order": [list(range(16)) for _ in range(10)]}
CONTEXT = {"format": WIRE_ID, "codec_metadata": dict(CODEC_METADATA),
           "checkpoint_sha256": CHECKPOINT_SHA, "model": "vgg16", "image_shape": [16, 3, 224, 224],
           "normalization": "imagenet", "training": TRAINING,
           "observation_source": "single_client_single_local_update", "batch_id": "batch00",
           "global_round_id": None, "client_id": None, "id_reason": "NOT_RECORDED",
           "delta_convention": "theta_after10_minus_theta_start", "labels": "private"}


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack(q: torch.Tensor, s: torch.Tensor) -> bytes:
    # Reuse utility validation without allocating dequantization of a large q.
    if q.device.type != "cpu" or q.dtype != torch.int8 or not q.numel():
        raise ValueError("codes must be nonempty CPU int8")
    if bool(((q < -7) | (q > 7)).any()):
        raise ValueError("invalid signed code")
    dequantize(q.flatten()[:1], s)
    mapped = (q.flatten().numpy() + 7).astype(np.uint8)
    packed = mapped[0::2].copy()
    packed[:len(mapped)//2] |= mapped[1::2] << 4
    return HEADER.pack(MAGIC, q.numel(), float(s)) + packed.tobytes()


def unpack(packet: bytes, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(packet) < HEADER.size:
        raise ValueError("truncated signed packet")
    magic, count, scale = HEADER.unpack_from(packet)
    if magic != MAGIC or count != int(np.prod(shape)) or not count:
        raise ValueError("wrong wire ID or shape")
    if len(packet) != HEADER.size + (count + 1)//2:
        raise ValueError("wrong packet length")
    payload = np.frombuffer(packet, dtype=np.uint8, offset=HEADER.size)
    low, high = payload & 15, payload >> 4
    if np.any(low == 15) or np.any(high[:count//2] == 15):
        raise ValueError("reserved nibble 15 is not a signed code")
    if count % 2 and high[-1] != 0:
        raise ValueError("nonzero padding nibble")
    mapped = np.empty(count, dtype=np.int8)
    mapped[0::2], mapped[1::2] = low.astype(np.int8)-7, high[:count//2].astype(np.int8)-7
    q, s = torch.from_numpy(mapped).reshape(shape), torch.tensor(scale, dtype=torch.float32)
    dequantize(q.flatten()[:1], s)
    return q, s


def _text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_bundle(folder: Path, updates: dict[str, torch.Tensor]) -> dict:
    """Sender-only export; only strict public fields are serialized."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    if set(updates) != set(FC_SHAPES):
        raise ValueError("all six FC tensors required")
    folder.mkdir(parents=True)
    layers = {}
    for index, name in enumerate(sorted(FC_SHAPES)):
        G = updates[name]
        if tuple(G.shape) != FC_SHAPES[name]:
            raise ValueError("wrong FC shape")
        q, s = encode(G)
        packet = pack(q, s)
        q2, s2 = unpack(packet, tuple(G.shape))
        if not torch.equal(q, q2) or not torch.equal(s, s2):
            raise RuntimeError("signed wire roundtrip failed")
        filename = f"fc_{index:02d}.sgn4"
        (folder / filename).write_bytes(packet)
        layers[name] = {"shape": list(G.shape), "file": filename, "sha256": digest(folder/filename),
                        "numel": G.numel(), "code_bytes": (G.numel()+1)//2,
                        "scale_bytes": 4, "header_bytes_excluding_scale": HEADER.size-4,
                        "file_bytes": len(packet), "q_dtype": "int8", "s_dtype": "float32", "s_shape": []}
    manifest = {**deepcopy(CONTEXT), "layers": layers, "communication": {
        "code_bytes": sum(v["code_bytes"] for v in layers.values()), "scale_bytes": 4*len(layers),
        "header_bytes_excluding_scale": (HEADER.size-4)*len(layers), "manifest_bytes": 0,
        "packet_file_bytes": sum(v["file_bytes"] for v in layers.values()), "total_file_bytes": 0}}
    counts = manifest["communication"]
    for _ in range(12):
        size = len(_text(manifest).encode("utf-8"))
        if size == counts["manifest_bytes"]:
            break
        counts["manifest_bytes"] = size
        counts["total_file_bytes"] = size + counts["packet_file_bytes"]
    else:
        raise RuntimeError("manifest accounting did not converge")
    (folder/"manifest.json").write_text(_text(manifest), encoding="utf-8", newline="\n")
    return manifest


def read_bundle(folder: Path) -> tuple[dict, dict]:
    """Strict public manifest/packet reader. No private file API or payload accepted."""
    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(value) != set(CONTEXT) | {"layers", "communication"}:
        raise ValueError("unexpected public manifest fields")
    for key, expected in CONTEXT.items():
        if _text(value[key]) != _text(expected):
            raise ValueError(f"public metadata mismatch: {key}")
    if set(value["layers"]) != set(FC_SHAPES):
        raise ValueError("expected six FC parameters")
    entries = {}
    for index, name in enumerate(sorted(FC_SHAPES)):
        layer = value["layers"][name]
        expected_keys = {"shape", "file", "sha256", "numel", "code_bytes", "scale_bytes",
                         "header_bytes_excluding_scale", "file_bytes", "q_dtype", "s_dtype", "s_shape"}
        if set(layer) != expected_keys or layer["file"] != f"fc_{index:02d}.sgn4":
            raise ValueError("invalid public layer descriptor")
        shape, count = FC_SHAPES[name], int(np.prod(FC_SHAPES[name]))
        expected = {"shape": list(shape), "numel": count, "code_bytes": (count+1)//2, "scale_bytes": 4,
                    "header_bytes_excluding_scale": HEADER.size-4, "file_bytes": HEADER.size+(count+1)//2,
                    "q_dtype": "int8", "s_dtype": "float32", "s_shape": []}
        if any(layer[k] != v for k, v in expected.items()):
            raise ValueError("invalid public shape/dtype/byte count")
        path = folder/layer["file"]
        if path.is_symlink() or path.resolve().parent != folder or digest(path) != layer["sha256"]:
            raise ValueError("public packet integrity/path failure")
        entries[name] = unpack(path.read_bytes(), shape)
    counts = {"code_bytes": sum(v["code_bytes"] for v in value["layers"].values()),
              "scale_bytes": 4*len(entries), "header_bytes_excluding_scale": (HEADER.size-4)*len(entries),
              "manifest_bytes": manifest_path.stat().st_size,
              "packet_file_bytes": sum(v["file_bytes"] for v in value["layers"].values())}
    counts["total_file_bytes"] = counts["manifest_bytes"] + counts["packet_file_bytes"]
    if value["communication"] != counts:
        raise ValueError("communication accounting mismatch")
    return value, entries
