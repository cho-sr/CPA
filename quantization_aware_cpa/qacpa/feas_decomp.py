from __future__ import annotations

import argparse
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import torch

from .alignment import align_estimate_to_truth
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .diagnostics import _sample_columns, estimate_column_feasibility
from .linalg import condition_number
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .victim import load_or_collect_victim_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose LP feasibility for true/full, true/kept, and A_hat.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    n_batches = int(args.n_batches if args.n_batches is not None else config["data"].get("n_batches", 1))
    diag_cfg = config["diagnostics"]

    for batch_index in range(n_batches):
        start = time.perf_counter()
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=bool(args.force_cache))
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))
        stage1_cfg = config["stage1"]
        stage1 = estimate_mixing_and_sources(
            qobs.Y_hat,
            qobs.delta,
            n_components=int(stage1_cfg["n_components"]),
            rounding=qobs.rounding,
            debias=bool(stage1_cfg.get("debias", True)),
            seed=seed + batch_index,
            max_iter=int(stage1_cfg.get("fastica_max_iter", 1000)),
            tol=float(stage1_cfg.get("fastica_tol", 1.0e-4)),
            fun=str(stage1_cfg.get("fastica_fun", "logcosh")),
            c_mp=float(stage1_cfg.get("c_mp", 1.5)),
            mp_truncation=bool(stage1_cfg.get("mp_truncation", True)),
            fastica_algorithm=str(stage1_cfg.get("fastica_algorithm", "parallel")),
        )
        _A_aligned, _S_aligned, alignment = align_estimate_to_truth(stage1.A_hat, stage1.S_hat, victim.S_true)
        matched = torch.tensor(alignment.true_indices, dtype=torch.long)
        A_true_kept = victim.A_true.index_select(1, matched)
        columns = _sample_columns(
            victim.S_true.shape[1],
            int(diag_cfg.get("gamma_columns", 512)),
            seed + 10_000 * batch_index,
        )
        common_kwargs = {
            "Y_hat": qobs.Y_hat,
            "delta": qobs.delta,
            "columns": columns,
            "method": str(diag_cfg.get("lp_method", "highs")),
            "feasibility_tol": float(diag_cfg.get("lp_feasibility_tol", 1.0e-7)),
        }
        full = estimate_column_feasibility(A=victim.A_true, **common_kwargs)
        kept = estimate_column_feasibility(A=A_true_kept, **common_kwargs)
        ahat = estimate_column_feasibility(A=stage1.A_hat, **common_kwargs)
        if float(full["feasible_column_fraction"]) < 0.999:
            raise RuntimeError(
                "A_true full feasibility should be 1.0; got "
                f"{float(full['feasible_column_fraction']):.6f} for batch {batch_index}"
            )

        row = make_decomp_row(
            config=config,
            batch_index=batch_index,
            victim=victim,
            stage1=stage1,
            alignment=alignment,
            full=full,
            kept=kept,
            ahat=ahat,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} B_eff={beff}/{B} feasibility full={full:.3f} "
            "true_kept={kept:.3f} A_hat={ahat:.3f} gap={gap:.3f}".format(
                batch=batch_index,
                beff=stage1.B_eff,
                B=stage1.B_requested,
                full=float(full["feasible_column_fraction"]),
                kept=float(kept["feasible_column_fraction"]),
                ahat=float(ahat["feasible_column_fraction"]),
                gap=float(kept["feasible_column_fraction"]) - float(ahat["feasible_column_fraction"]),
            ),
            flush=True,
        )


def make_decomp_row(
    *,
    config: dict[str, Any],
    batch_index: int,
    victim: Any,
    stage1: Any,
    alignment: Any,
    full: dict[str, float | int],
    kept: dict[str, float | int],
    ahat: dict[str, float | int],
    wall_time_sec: float,
) -> dict[str, Any]:
    stage1_dict = asdict(stage1)
    stage1_dict.pop("A_hat", None)
    stage1_dict.pop("S_hat", None)
    stage1_dict.pop("Y_recon", None)
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = "v3_feas_decomp"

    return {
        "schema_version": 1,
        "run_tag": "v3_feas_decomp",
        "milestone": "feasibility_decomposition",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "B_requested": int(stage1.B_requested),
            "B_eff": int(stage1.B_eff),
            "A_true_full_feasible_column_fraction": float(full["feasible_column_fraction"]),
            "A_true_kept_feasible_column_fraction": float(kept["feasible_column_fraction"]),
            "A_hat_truncated_feasible_column_fraction": float(ahat["feasible_column_fraction"]),
            "A_true_kept_minus_A_hat_truncated": (
                float(kept["feasible_column_fraction"]) - float(ahat["feasible_column_fraction"])
            ),
            "A_true_full_minus_A_true_kept": (
                float(full["feasible_column_fraction"]) - float(kept["feasible_column_fraction"])
            ),
        },
        "diagnostics": {
            "matched_true_indices": list(alignment.true_indices),
            "unmatched_true_indices": list(alignment.unmatched_true_indices),
            "cond_A_true": condition_number(victim.A_true),
            "cond_A_true_matched": condition_number(
                victim.A_true.index_select(1, torch.tensor(alignment.true_indices, dtype=torch.long))
            ),
            "cond_A_hat": condition_number(stage1.A_hat),
            "full": full,
            "true_kept": kept,
            "A_hat_truncated": ahat,
        },
        "stage1": stage1_dict,
        "wall_time_sec": float(wall_time_sec),
    }


if __name__ == "__main__":
    main()
