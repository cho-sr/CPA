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
    scale_granularity: str = "per_row",
    bucket_size: int | None = None,
    require_distinct_labels: bool = True,
) -> Path:
    bucket_suffix = "" if scale_granularity != "per_bucket" else f"_bucket{int(bucket_size or 256)}"
    label_suffix = "" if require_distinct_labels else "_labelsmixed"
    name = (
        f"quant_seed{seed}_B{B}_batch{batch_index}{label_suffix}_bits{bits}_{rounding}"
        f"_scale{scale_granularity}{bucket_suffix}.pt"
    )
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
    scale_granularity = str(quant_cfg.get("scale_granularity", "per_row"))
    bucket_size = int(quant_cfg.get("bucket_size", 256))
    require_distinct = bool(config.get("data", {}).get("require_distinct_labels", True))
    path = quant_cache_path(
        config["paths"]["cache_dir"],
        seed=int(config["seed"]),
        B=int(config["data"]["batch_size"]),
        batch_index=int(victim.batch_index),
        bits=bits,
        rounding=rounding,
        scale_granularity=scale_granularity,
        bucket_size=bucket_size,
        require_distinct_labels=require_distinct,
    )
    if path.is_file() and not force:
        payload = torch.load(path, map_location="cpu")
        return QuantizedObservation(
            Y_hat=payload["Y_hat"],
            delta=payload["delta"],
            bits=int(payload["bits"]),
            rounding=str(payload["rounding"]),
            scale_granularity=str(payload.get("scale_granularity", scale_granularity)),
            bucket_size=payload.get("bucket_size", bucket_size),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    generator_seed = int(config["seed"]) + 1_000_003 * int(victim.batch_index)
    qobs = quantize(
        victim.Y,
        bits=bits,
        rounding=rounding,
        generator=torch_generator(generator_seed, device=victim.Y.device),
        scale_granularity=scale_granularity,
        bucket_size=bucket_size,
    )
    assert_cell_membership(victim.Y, qobs.Y_hat, qobs.delta)
    torch.save(
        {
            "schema_version": 1,
            "Y_hat": qobs.Y_hat.cpu(),
            "delta": qobs.delta.cpu(),
            "bits": bits,
            "rounding": rounding,
            "scale_granularity": scale_granularity,
            "bucket_size": bucket_size,
            "require_distinct_labels": require_distinct,
            "seed": int(config["seed"]),
            "batch_index": int(victim.batch_index),
        },
        path,
    )
    return qobs
