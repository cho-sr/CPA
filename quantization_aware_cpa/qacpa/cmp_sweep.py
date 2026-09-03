from __future__ import annotations

import argparse
from dataclasses import asdict
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .alignment import align_estimate_to_truth
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .victim import load_or_collect_victim_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep c_mp for Stage-1 MP truncation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--c-mp-values", default="1.0,1.5,3,6,12")
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    c_mp_values = [float(value.strip()) for value in args.c_mp_values.split(",") if value.strip()]
    n_batches = int(args.n_batches if args.n_batches is not None else config["data"].get("n_batches", 1))

    for c_mp in c_mp_values:
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
                c_mp=float(c_mp),
                mp_truncation=True,
                fastica_algorithm=str(stage1_cfg.get("fastica_algorithm", "parallel")),
            )
            A_aligned, _S_aligned, alignment = align_estimate_to_truth(
                stage1.A_hat,
                stage1.S_hat,
                victim.S_true,
            )
            row = make_sweep_row(
                config=config,
                batch_index=batch_index,
                c_mp=float(c_mp),
                victim=victim,
                stage1=stage1,
                alignment=alignment,
                A_aligned=A_aligned,
                wall_time_sec=time.perf_counter() - start,
            )
            append_jsonl(Path(config["paths"]["results_jsonl"]), row)
            print(
                "c_mp={c_mp:g} batch={batch} B_eff={beff}/{B} min_lam_ratio={lam:.4g} "
                "worst_cos={cos:.4f} spearman_loss_cos={rho:.4f} recovered={rec}/{beff}".format(
                    c_mp=c_mp,
                    batch=batch_index,
                    beff=row["metrics"]["B_eff"],
                    B=row["metrics"]["B_requested"],
                    lam=row["metrics"]["min_retained_lam_over_mp_edge"],
                    cos=row["metrics"]["worst_source_cosine"],
                    rho=row["metrics"]["spearman_loss_vs_cosine"],
                    rec=row["metrics"]["n_sources_recovered"],
                ),
                flush=True,
            )


def make_sweep_row(
    *,
    config: dict[str, Any],
    batch_index: int,
    c_mp: float,
    victim: Any,
    stage1: Any,
    alignment: Any,
    A_aligned: torch.Tensor,
    wall_time_sec: float,
) -> dict[str, Any]:
    stage1_dict = asdict(stage1)
    stage1_dict.pop("A_hat", None)
    stage1_dict.pop("S_hat", None)
    stage1_dict.pop("Y_recon", None)

    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    a_true_norms = [float(value) for value in victim.A_true.norm(dim=0).detach().cpu().tolist()]
    source_cos_per_truth = [0.0] * int(victim.S_true.shape[0])
    for true_idx, cosine in zip(alignment.true_indices, alignment.per_source_abs_cosine):
        source_cos_per_truth[int(true_idx)] = float(cosine)

    spearman = spearman_rank_correlation(losses, source_cos_per_truth)
    loss_ranks = rankdata(losses)
    cosine_ranks = rankdata(source_cos_per_truth)
    worst_pos = int(np.argmin(np.asarray(alignment.per_source_abs_cosine, dtype=np.float64)))
    worst_true_idx = int(alignment.true_indices[worst_pos])
    min_ratio_pos = int(np.argmin(np.asarray(stage1.lam_over_mp_edge_kept, dtype=np.float64)))
    min_ratio_component = int(stage1.kept_component_indices[min_ratio_pos])
    n_recovered = int(sum(value > 0.9 for value in alignment.per_source_abs_cosine))

    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = "v3_cmp_sweep"
    config_for_row.setdefault("stage1", {})["c_mp"] = float(c_mp)
    source_records = [
        {
            "source_index": idx,
            "label": labels[idx],
            "loss": losses[idx],
            "a_true_norm": a_true_norms[idx],
            "source_cosine": source_cos_per_truth[idx],
            "loss_rank": float(loss_ranks[idx]),
            "cosine_rank": float(cosine_ranks[idx]),
            "matched": idx in set(alignment.true_indices),
        }
        for idx in range(len(labels))
    ]

    return {
        "schema_version": 1,
        "run_tag": "v3_cmp_sweep",
        "milestone": "c_mp_sweep",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "c_mp": float(c_mp),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "B_requested": int(stage1.B_requested),
            "B_eff": int(stage1.B_eff),
            "source_cos_per_source": [float(x) for x in alignment.per_source_abs_cosine],
            "source_cos_per_truth": source_cos_per_truth,
            "n_sources_recovered": n_recovered,
            "recovered_over_beff": float(n_recovered / max(1, int(stage1.B_eff))),
            "recovered_over_B": float(n_recovered / max(1, int(stage1.B_requested))),
            "min_retained_lam_over_mp_edge": float(stage1.lam_over_mp_edge_kept[min_ratio_pos]),
            "min_retained_component_index": min_ratio_component,
            "worst_source_cosine": float(alignment.per_source_abs_cosine[worst_pos]),
            "worst_source_true_index": worst_true_idx,
            "worst_source_label": labels[worst_true_idx],
            "worst_source_loss": losses[worst_true_idx],
            "worst_source_a_true_norm": a_true_norms[worst_true_idx],
            "min_lam_ratio_worst_cosine_pair": {
                "min_retained_lam_over_mp_edge": float(stage1.lam_over_mp_edge_kept[min_ratio_pos]),
                "min_retained_component_index": min_ratio_component,
                "worst_source_cosine": float(alignment.per_source_abs_cosine[worst_pos]),
                "worst_source_true_index": worst_true_idx,
                "worst_source_label": labels[worst_true_idx],
                "worst_source_loss": losses[worst_true_idx],
                "worst_source_a_true_norm": a_true_norms[worst_true_idx],
            },
            "spearman_loss_vs_cosine": spearman,
            "loss_rank": [float(x) for x in loss_ranks],
            "cosine_rank": [float(x) for x in cosine_ranks],
        },
        "diagnostics": {
            "cond_A_hat": condition_number(stage1.A_hat),
            "cond_A_true": condition_number(victim.A_true),
            "cond_A_true_matched": condition_number(
                victim.A_true.index_select(1, torch.tensor(alignment.true_indices, dtype=torch.long))
            ),
            "a_hat_norm_per_source": [float(x) for x in A_aligned.norm(dim=0).detach().cpu().tolist()],
            "a_true_norm_per_source": a_true_norms,
            "matched_true_indices": list(alignment.true_indices),
            "unmatched_true_indices": list(alignment.unmatched_true_indices),
            "source_records": source_records,
        },
        "stage1": stage1_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    rx = np.asarray(rankdata(x), dtype=np.float64)
    ry = np.asarray(rankdata(y), dtype=np.float64)
    if rx.size < 2 or float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def rankdata(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < arr.shape[0]:
        j = i + 1
        while j < arr.shape[0] and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return [float(x) for x in ranks]


if __name__ == "__main__":
    main()
