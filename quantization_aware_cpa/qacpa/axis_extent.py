from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .diagnostics import _sample_columns, estimate_coordinate_extents
from .run import append_jsonl, git_sha
from .seed import seed_all
from .victim import load_or_collect_victim_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log coordinate-wise feasible extents.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    diag_cfg = config["diagnostics"]
    batch_indices = parse_batch_indices(args.batch_indices, int(args.n_batches))

    for batch_index in batch_indices:
        start = time.perf_counter()
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=bool(args.force_cache))
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))
        columns = _sample_columns(
            victim.S_true.shape[1],
            int(diag_cfg.get("gamma_columns", 512)),
            seed + 10_000 * batch_index,
        )
        extents = estimate_coordinate_extents(
            A=victim.A_true,
            S_reference=victim.S_true,
            Y_hat=qobs.Y_hat,
            delta=qobs.delta,
            columns=columns,
            method=str(diag_cfg.get("lp_method", "highs")),
            feasibility_tol=float(diag_cfg.get("lp_feasibility_tol", 1.0e-7)),
        )
        row = make_row(
            config=config,
            batch_index=batch_index,
            victim=victim,
            extents=extents,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} extent_vs_inv_norm pearson={pearson:.4f} spearman={spearman:.4f} "
            "feasible={feasible:.3f} extent_times_norm={products}".format(
                batch=batch_index,
                pearson=float(extents["pearson_extent_vs_inv_a_norm"]),
                spearman=float(extents["spearman_extent_vs_inv_a_norm"]),
                feasible=float(extents["feasible_column_fraction"]),
                products="[" + ", ".join(f"{float(x):.3g}" for x in extents["extent_times_a_norm"]) + "]",
            ),
            flush=True,
        )


def make_row(
    *,
    config: dict[str, Any],
    batch_index: int,
    victim: Any,
    extents: dict[str, Any],
    wall_time_sec: float,
) -> dict[str, Any]:
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = "v3_axis_extent"
    return {
        "schema_version": 1,
        "run_tag": "v3_axis_extent",
        "milestone": "coordinate_extent",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "pearson_extent_vs_inv_a_norm": extents["pearson_extent_vs_inv_a_norm"],
            "spearman_extent_vs_inv_a_norm": extents["spearman_extent_vs_inv_a_norm"],
            "feasible_column_fraction": extents["feasible_column_fraction"],
        },
        "diagnostics": {
            **extents,
            "label_per_source": labels,
            "sample_loss_per_source": losses,
        },
        "wall_time_sec": float(wall_time_sec),
    }


def parse_batch_indices(raw: str | None, n_batches: int) -> list[int]:
    if raw is None:
        return list(range(int(n_batches)))
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--batch-indices must contain at least one integer")
    return values


if __name__ == "__main__":
    main()
