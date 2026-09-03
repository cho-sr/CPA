from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .quantizer import QuantizedObservation, assert_cell_membership, quantize
from .seed import torch_generator
from .victim import VictimBatch


def quant_cache_path(
    cache_dir: str | Path,
    *,
    seed: int,
    B: int,
    batch_index: int,
    bits: int,
    rounding: str,
) -> Path:
    name = f"quant_seed{seed}_B{B}_batch{batch_index}_bits{bits}_{rounding}.pt"
    return Path(cache_dir) / name


def load_or_quantize_observation(
    victim: VictimBatch,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> QuantizedObservation:
    quant_cfg = config["quantization"]
    bits = int(quant_cfg["bits"])
    rounding = str(quant_cfg["rounding"])
    path = quant_cache_path(
        config["paths"]["cache_dir"],
        seed=int(config["seed"]),
        B=int(config["data"]["batch_size"]),
        batch_index=int(victim.batch_index),
        bits=bits,
        rounding=rounding,
    )
    if path.is_file() and not force:
        payload = torch.load(path, map_location="cpu")
        return QuantizedObservation(
            Y_hat=payload["Y_hat"],
            delta=payload["delta"],
            bits=int(payload["bits"]),
            rounding=str(payload["rounding"]),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    generator_seed = int(config["seed"]) + 1_000_003 * int(victim.batch_index)
    qobs = quantize(
        victim.Y,
        bits=bits,
        rounding=rounding,
        generator=torch_generator(generator_seed, device=victim.Y.device),
    )
    assert_cell_membership(victim.Y, qobs.Y_hat, qobs.delta)
    torch.save(
        {
            "schema_version": 1,
            "Y_hat": qobs.Y_hat.cpu(),
            "delta": qobs.delta.cpu(),
            "bits": bits,
            "rounding": rounding,
            "seed": int(config["seed"]),
            "batch_index": int(victim.batch_index),
        },
        path,
    )
    return qobs
