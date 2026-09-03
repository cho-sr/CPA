from __future__ import annotations

import argparse
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number, row_normalized
from .quantizer import consistency_rate, mean_hinge
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage2 import oracle_pinv_initialization, solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-2 A_oracle baseline only.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--run-tag", default="v3_A_oracle")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    device = resolve_device(str(config.get("device", "auto")))
    stage2_cfg = config.get("stage2", {})

    for batch_index in range(int(args.n_batches)):
        start = time.perf_counter()
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=bool(args.force_cache))
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))
        init_start = time.perf_counter()
        S_init = oracle_pinv_initialization(victim.A_true, qobs.Y_hat)
        init_wall = time.perf_counter() - init_start
        result = solve_stage2_consistency(
            A=victim.A_true,
            Y_hat=qobs.Y_hat,
            delta=qobs.delta,
            S_init=S_init,
            max_iter=int(stage2_cfg.get("max_iter", 250)),
            chunk_size=int(stage2_cfg.get("chunk_size", 4096)),
            device=device,
            stop_tolerance=float(stage2_cfg.get("stop_tolerance", 1.0e-10)),
            check_every=int(stage2_cfg.get("check_every", 25)),
        )
        row = make_oracle_row(
            config=config,
            run_tag=str(args.run_tag),
            batch_index=batch_index,
            victim=victim,
            qobs=qobs,
            S_init=S_init,
            result=result,
            init_wall_time_sec=init_wall,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} A_oracle recovered={rec}/{B} source_cos={cosines} "
            "consistency={cons:.6f} residual={res:.3g} init_consistency={init_cons:.6f} "
            "iters_max={iters}".format(
                batch=batch_index,
                rec=row["metrics"]["n_sources_recovered"],
                B=row["metrics"]["B_requested"],
                cosines="[" + ", ".join(f"{value:.4f}" for value in row["metrics"]["source_cos_per_source"]) + "]",
                cons=row["metrics"]["consistency_rate"],
                res=row["metrics"]["residual_violation"],
                init_cons=row["diagnostics"]["initial_consistency_rate"],
                iters=max(row["stage2"]["chunk_iterations"]),
            ),
            flush=True,
        )


def make_oracle_row(
    *,
    config: dict[str, Any],
    run_tag: str,
    batch_index: int,
    victim: Any,
    qobs: Any,
    S_init: torch.Tensor,
    result: Any,
    init_wall_time_sec: float,
    wall_time_sec: float,
) -> dict[str, Any]:
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    a_true_norms = [float(value) for value in victim.A_true.norm(dim=0).detach().cpu().tolist()]
    source_cos = direct_source_cosines(victim.S_true, result.S_star)
    init_source_cos = direct_source_cosines(victim.S_true, S_init)
    n_recovered = int(sum(value > 0.9 for value in source_cos))
    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = run_tag
    stage2_dict = asdict(result)
    stage2_dict.pop("S_star", None)
    stage2_dict.pop("per_column_hinge_mean", None)
    stage2_dict.pop("per_column_squared_hinge_mean", None)

    hinge_mean = result.per_column_hinge_mean.detach().cpu()
    hinge_sq_mean = result.per_column_squared_hinge_mean.detach().cpu()
    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "milestone": "stage2_A_oracle",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "method": "A_oracle",
            "B_requested": int(victim.S_true.shape[0]),
            "B_eff": int(victim.S_true.shape[0]),
            "source_cos_per_source": source_cos,
            "source_cos_initialization": init_source_cos,
            "source_cos": float(sum(source_cos) / max(1, len(source_cos))),
            "n_sources_recovered": n_recovered,
            "recovered_over_B": float(n_recovered / max(1, int(victim.S_true.shape[0]))),
            "consistency_rate": consistency_rate(victim.A_true, result.S_star, qobs.Y_hat, qobs.delta),
            "residual_violation": mean_hinge(victim.A_true, result.S_star, qobs.Y_hat, qobs.delta),
            "initial_consistency_rate": consistency_rate(victim.A_true, S_init, qobs.Y_hat, qobs.delta),
            "initial_residual_violation": mean_hinge(victim.A_true, S_init, qobs.Y_hat, qobs.delta),
        },
        "diagnostics": {
            "cond_A_true": condition_number(victim.A_true),
            "a_true_norm_per_source": a_true_norms,
            "sample_loss_per_source": losses,
            "label_per_source": labels,
            "per_column_hinge_mean_mean": float(hinge_mean.mean().cpu()),
            "per_column_hinge_mean_max": float(hinge_mean.max().cpu()),
            "per_column_squared_hinge_mean_mean": float(hinge_sq_mean.mean().cpu()),
            "per_column_squared_hinge_mean_max": float(hinge_sq_mean.max().cpu()),
            "initial_consistency_rate": result.initial_consistency_rate,
            "final_consistency_rate": result.final_consistency_rate,
            "initial_residual_violation": result.initial_residual_violation,
            "final_residual_violation": result.final_residual_violation,
            "chunk_iteration_min": min(result.chunk_iterations),
            "chunk_iteration_max": max(result.chunk_iterations),
            "chunk_iteration_mean": statistics.fmean(result.chunk_iterations),
            "init_wall_time_sec": float(init_wall_time_sec),
        },
        "stage2": stage2_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def direct_source_cosines(S_true: torch.Tensor, S_est: torch.Tensor) -> list[float]:
    similarity = row_normalized(S_true.detach().cpu().float()) * row_normalized(S_est.detach().cpu().float())
    return [float(value) for value in similarity.sum(dim=1).abs().cpu().tolist()]


if __name__ == "__main__":
    main()
