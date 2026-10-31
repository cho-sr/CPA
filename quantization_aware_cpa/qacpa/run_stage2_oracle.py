from __future__ import annotations

import argparse
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .alignment import match_sources_by_abs_cosine
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number, row_normalized
from .quantizer import consistency_rate, mean_hinge
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .stage2 import oracle_pinv_initialization, solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-2 A_oracle baseline only.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--run-tag", default="v3_A_oracle")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-stage1-mp-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    device = resolve_device(str(config.get("device", "auto")))
    stage1_cfg = config["stage1"]
    stage2_cfg = config.get("stage2", {})
    batch_indices = parse_batch_indices(args.batch_indices, int(args.n_batches))

    for batch_index in batch_indices:
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
            lam_R=float(stage2_cfg.get("lam_R", 0.0)),
            mu=float(stage2_cfg.get("mu", 1.0e3)),
            eps=float(stage2_cfg.get("eps", 1.0e-3)),
            stop_tolerance=float(stage2_cfg.get("stop_tolerance", 1.0e-10)),
            check_every=int(stage2_cfg.get("check_every", 25)),
            phase1_max_iter=int(stage2_cfg.get("phase1_max_iter", stage2_cfg.get("max_iter", 250))),
            phase2_max_iter=int(stage2_cfg.get("phase2_max_iter", stage2_cfg.get("max_iter", 250))),
            phase1_acceleration=bool(stage2_cfg.get("phase1_acceleration", True)),
        )
        stage1 = None
        if not bool(args.skip_stage1_mp_metrics):
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
                c_mp=float(stage1_cfg.get("c_mp", 3.0)),
                mp_truncation=bool(stage1_cfg.get("mp_truncation", True)),
                fastica_algorithm=str(stage1_cfg.get("fastica_algorithm", "parallel")),
                ica_method=str(stage1_cfg.get("ica_method", "cpa")),
                cpa_ica_max_iter=int(stage1_cfg.get("cpa_ica_max_iter", 25000)),
                cpa_ica_lr=float(stage1_cfg.get("cpa_ica_lr", 1.0e-3)),
                cpa_ica_optimizer=str(stage1_cfg.get("cpa_ica_optimizer", "adam")),
                cpa_ica_decor=float(stage1_cfg.get("cpa_ica_decor", 5.3)),
                cpa_ica_T=float(stage1_cfg.get("cpa_ica_T", 7.7)),
                cpa_ica_ne=float(stage1_cfg.get("cpa_ica_ne", 1.0)),
                cpa_ica_nv=float(stage1_cfg.get("cpa_ica_nv", 0.13)),
                cpa_ica_l1=float(stage1_cfg.get("cpa_ica_l1", 5.0)),
                cpa_ica_tv=float(stage1_cfg.get("cpa_ica_tv", 0.1)),
                cpa_ica_log_every=int(stage1_cfg.get("cpa_ica_log_every", 0)),
            )
        row = make_oracle_row(
            config=config,
            run_tag=str(args.run_tag),
            batch_index=batch_index,
            victim=victim,
            qobs=qobs,
            S_init=S_init,
            result=result,
            stage1=stage1,
            init_wall_time_sec=init_wall,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} A_oracle recovered={rec}/{B} source_cos={cosines} "
            "phase1_consistency={p1_cons:.6f} phase2_consistency={p2_cons:.6f} "
            "H={h0:.3g}->{h1:.3g}->{h2:.3g} lam_R={lam_R:.3g} "
            "subthr09={subthr09} subthr07={subthr07} subthr05={subthr05} "
            "max_cos_delta={delta:.3g} iters_max={iters}".format(
                batch=batch_index,
                rec=row["metrics"]["n_sources_recovered"],
                B=row["metrics"]["B_requested"],
                cosines="[" + ", ".join(f"{value:.4f}" for value in row["metrics"]["source_cos_per_source"]) + "]",
                p1_cons=row["stage2"]["phase1_consistency_rate"],
                p2_cons=row["stage2"]["final_consistency_rate"],
                h0=row["diagnostics"]["per_column_H_initial_mean"],
                h1=row["stage2"]["phase1_H_mean"],
                h2=row["stage2"]["final_H_mean"],
                lam_R=row["stage2"]["lam_R"],
                subthr09=row["metrics"]["sub_threshold_recovered_cos_gt_0_9"],
                subthr07=row["metrics"]["sub_threshold_recovered_cos_gt_0_7"],
                subthr05=row["metrics"]["sub_threshold_recovered_cos_gt_0_5"],
                delta=row["metrics"]["max_abs_source_cos_improvement"],
                iters=max(row["stage2"]["phase2_chunk_iterations"]),
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
    stage1: Any | None,
    init_wall_time_sec: float,
    wall_time_sec: float,
) -> dict[str, Any]:
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    a_true_norms = [float(value) for value in victim.A_true.norm(dim=0).detach().cpu().tolist()]
    source_cos = direct_source_cosines(victim.S_true, result.S_star)
    init_source_cos = direct_source_cosines(victim.S_true, S_init)
    phase1_source_cos = direct_source_cosines(victim.S_true, result.S_phase1)
    source_cos_improvement = [float(after - before) for before, after in zip(init_source_cos, source_cos)]
    recovered_indices = [idx for idx, value in enumerate(source_cos) if value > 0.9]
    failed_indices = [idx for idx, value in enumerate(source_cos) if value <= 0.9]
    n_recovered = int(sum(value > 0.9 for value in source_cos))
    mp_summary = mp_detected_source_summary(stage1, victim.S_true, source_cos)
    subthreshold_indices = [int(idx) for idx in mp_summary["mp_subthreshold_true_indices"]]
    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = run_tag
    stage2_dict = asdict(result)
    for key, value in list(stage2_dict.items()):
        if isinstance(value, torch.Tensor):
            stage2_dict.pop(key, None)

    hinge_mean = result.per_column_hinge_mean.detach().cpu()
    hinge_sq_mean = result.per_column_squared_hinge_mean.detach().cpu()
    stage1_dict = None
    if stage1 is not None:
        stage1_dict = asdict(stage1)
        stage1_dict.pop("A_hat", None)
        stage1_dict.pop("S_hat", None)
        stage1_dict.pop("Y_recon", None)
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
            "B_eff": int(mp_summary["B_eff"]),
            "source_cos_per_source": source_cos,
            "source_cos_initialization": init_source_cos,
            "source_cos_phase1": phase1_source_cos,
            "source_cos_pairs": [
                {
                    "source_index": idx,
                    "initial": init_source_cos[idx],
                    "phase1": phase1_source_cos[idx],
                    "final": source_cos[idx],
                    "delta": source_cos_improvement[idx],
                }
                for idx in range(len(source_cos))
            ],
            "source_cos_improvement": source_cos_improvement,
            "max_abs_source_cos_improvement": max((abs(value) for value in source_cos_improvement), default=0.0),
            "source_cos": float(sum(source_cos) / max(1, len(source_cos))),
            "n_sources_recovered": n_recovered,
            "n_sources_recovered_over_beff": int(mp_summary["n_sources_recovered_over_beff"]),
            "recovered_over_beff": float(mp_summary["recovered_over_beff"]),
            "recovered_over_B": float(n_recovered / max(1, int(victim.S_true.shape[0]))),
            "sub_threshold_recovered": int(mp_summary["sub_threshold_recovered"]),
            "sub_threshold_recovered_cos_gt_0_9": int(mp_summary["sub_threshold_recovered_cos_gt_0_9"]),
            "sub_threshold_recovered_cos_gt_0_7": int(mp_summary["sub_threshold_recovered_cos_gt_0_7"]),
            "sub_threshold_recovered_cos_gt_0_5": int(mp_summary["sub_threshold_recovered_cos_gt_0_5"]),
            "sub_threshold_recovered_by_threshold": dict(mp_summary["sub_threshold_recovered_by_threshold"]),
            "sub_threshold_cosines": [source_cos[int(idx)] for idx in subthreshold_indices],
            "consistency_rate": consistency_rate(victim.A_true, result.S_star, qobs.Y_hat, qobs.delta),
            "residual_violation": mean_hinge(victim.A_true, result.S_star, qobs.Y_hat, qobs.delta),
            "initial_consistency_rate": consistency_rate(victim.A_true, S_init, qobs.Y_hat, qobs.delta),
            "initial_residual_violation": mean_hinge(victim.A_true, S_init, qobs.Y_hat, qobs.delta),
        },
        "diagnostics": {
            "cond_A_true": condition_number(victim.A_true),
            "cond_A_true_matched": condition_number_for_columns(victim.A_true, recovered_indices),
            "cond_A_true_recovered": condition_number_for_columns(victim.A_true, recovered_indices),
            "cond_A_true_failed_removed": condition_number_for_columns(victim.A_true, recovered_indices),
            "recovered_source_indices": recovered_indices,
            "failed_source_indices": failed_indices,
            "failed_source_records": [
                {
                    "source_index": idx,
                    "label": labels[idx],
                    "loss": losses[idx],
                    "a_true_norm": a_true_norms[idx],
                    "source_cosine": source_cos[idx],
                }
                for idx in failed_indices
            ],
            "a_true_norm_per_source": a_true_norms,
            "sample_loss_per_source": losses,
            "label_per_source": labels,
            "mp_detected_true_indices": list(mp_summary["mp_detected_true_indices"]),
            "mp_subthreshold_true_indices": list(mp_summary["mp_subthreshold_true_indices"]),
            "mp_lam_over_edge_all": list(mp_summary["mp_lam_over_edge_all"]),
            "mp_c_mp": mp_summary["mp_c_mp"],
            "mp_edge": mp_summary["mp_edge"],
            "stage1_B_eff_for_mp_metrics": int(mp_summary["B_eff"]),
            "per_column_H_initial_mean": float(result.per_column_H_initial.detach().cpu().mean()),
            "per_column_H_initial_max": float(result.per_column_H_initial.detach().cpu().max()),
            "per_column_H_phase1_mean": float(result.per_column_H_phase1.detach().cpu().mean()),
            "per_column_H_phase1_max": float(result.per_column_H_phase1.detach().cpu().max()),
            "per_column_H_final_mean": float(result.per_column_H_final.detach().cpu().mean()),
            "per_column_H_final_max": float(result.per_column_H_final.detach().cpu().max()),
            "per_column_l1_raw_phase1_mean": float(result.per_column_l1_raw_phase1.detach().cpu().mean()),
            "per_column_l1_raw_final_mean": float(result.per_column_l1_raw_final.detach().cpu().mean()),
            "per_column_l1_normalized_phase1_mean": float(result.per_column_l1_normalized_phase1.detach().cpu().mean()),
            "per_column_l1_normalized_final_mean": float(result.per_column_l1_normalized_final.detach().cpu().mean()),
            "phase2_excess_mean": result.phase2_excess_mean,
            "phase2_excess_max": result.phase2_excess_max,
            "input_guard_reject_fraction": result.input_guard_reject_fraction,
            "per_column_hinge_mean_mean": float(hinge_mean.mean().cpu()),
            "per_column_hinge_mean_max": float(hinge_mean.max().cpu()),
            "per_column_squared_hinge_mean_mean": float(hinge_sq_mean.mean().cpu()),
            "per_column_squared_hinge_mean_max": float(hinge_sq_mean.max().cpu()),
            "initial_consistency_rate": result.initial_consistency_rate,
            "phase1_consistency_rate": result.phase1_consistency_rate,
            "final_consistency_rate": result.final_consistency_rate,
            "initial_residual_violation": result.initial_residual_violation,
            "phase1_residual_violation": result.phase1_residual_violation,
            "final_residual_violation": result.final_residual_violation,
            "initial_l1_mean": result.initial_l1_mean,
            "phase1_l1_mean": result.phase1_l1_mean,
            "final_l1_mean": result.final_l1_mean,
            "initial_l1_normalized_mean": result.initial_l1_normalized_mean,
            "phase1_l1_normalized_mean": result.phase1_l1_normalized_mean,
            "final_l1_normalized_mean": result.final_l1_normalized_mean,
            "initial_objective": result.initial_objective,
            "phase1_objective": result.phase1_objective,
            "final_objective": result.final_objective,
            "phase1_chunk_iteration_min": min(result.phase1_chunk_iterations),
            "phase1_chunk_iteration_max": max(result.phase1_chunk_iterations),
            "phase1_chunk_iteration_mean": statistics.fmean(result.phase1_chunk_iterations),
            "phase2_chunk_iteration_min": min(result.phase2_chunk_iterations),
            "phase2_chunk_iteration_max": max(result.phase2_chunk_iterations),
            "phase2_chunk_iteration_mean": statistics.fmean(result.phase2_chunk_iterations),
            "init_wall_time_sec": float(init_wall_time_sec),
            "per_column_phase_logs": {
                "H_initial": [float(x) for x in result.per_column_H_initial.detach().cpu().tolist()],
                "H_phase1": [float(x) for x in result.per_column_H_phase1.detach().cpu().tolist()],
                "H_final": [float(x) for x in result.per_column_H_final.detach().cpu().tolist()],
                "l1_raw_phase1": [float(x) for x in result.per_column_l1_raw_phase1.detach().cpu().tolist()],
                "l1_raw_final": [float(x) for x in result.per_column_l1_raw_final.detach().cpu().tolist()],
                "l1_normalized_phase1": [
                    float(x) for x in result.per_column_l1_normalized_phase1.detach().cpu().tolist()
                ],
                "l1_normalized_final": [
                    float(x) for x in result.per_column_l1_normalized_final.detach().cpu().tolist()
                ],
            },
        },
        "stage1_for_mp_metrics": stage1_dict,
        "stage2": stage2_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def direct_source_cosines(S_true: torch.Tensor, S_est: torch.Tensor) -> list[float]:
    similarity = row_normalized(S_true.detach().cpu().float()) * row_normalized(S_est.detach().cpu().float())
    return [float(value) for value in similarity.sum(dim=1).abs().cpu().tolist()]


def parse_batch_indices(raw: str | None, n_batches: int) -> list[int]:
    if raw is None:
        return list(range(int(n_batches)))
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--batch-indices must contain at least one integer")
    return values


def condition_number_for_columns(A: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return float("nan")
    return condition_number(A.index_select(1, torch.tensor(indices, dtype=torch.long)))


def mp_detected_source_summary(stage1: Any | None, S_true: torch.Tensor, source_cos: list[float]) -> dict[str, Any]:
    B = int(S_true.shape[0])
    if stage1 is None:
        detected = list(range(B))
        subthreshold: list[int] = []
        lam_over_edge_all: list[float] = []
        c_mp: float | None = None
        mp_edge: float | None = None
        B_eff = B
    else:
        alignment = match_sources_by_abs_cosine(S_true.detach().cpu().float(), stage1.S_hat.detach().cpu().float())
        detected = [int(idx) for idx in alignment.true_indices]
        subthreshold = [int(idx) for idx in alignment.unmatched_true_indices]
        lam_over_edge_all = [float(x) for x in stage1.lam_over_mp_edge_all]
        c_mp = float(stage1.c_mp)
        mp_edge = float(stage1.mp_edge)
        B_eff = int(stage1.B_eff)

    detected_recovered = int(sum(source_cos[idx] > 0.9 for idx in detected))
    threshold_counts = {
        "0.5": int(sum(source_cos[idx] > 0.5 for idx in subthreshold)),
        "0.7": int(sum(source_cos[idx] > 0.7 for idx in subthreshold)),
        "0.9": int(sum(source_cos[idx] > 0.9 for idx in subthreshold)),
    }
    return {
        "B_eff": B_eff,
        "mp_detected_true_indices": detected,
        "mp_subthreshold_true_indices": subthreshold,
        "mp_lam_over_edge_all": lam_over_edge_all,
        "mp_c_mp": c_mp,
        "mp_edge": mp_edge,
        "n_sources_recovered_over_beff": detected_recovered,
        "recovered_over_beff": float(detected_recovered / max(1, B_eff)),
        "sub_threshold_recovered": threshold_counts["0.9"],
        "sub_threshold_recovered_cos_gt_0_9": threshold_counts["0.9"],
        "sub_threshold_recovered_cos_gt_0_7": threshold_counts["0.7"],
        "sub_threshold_recovered_cos_gt_0_5": threshold_counts["0.5"],
        "sub_threshold_recovered_by_threshold": threshold_counts,
    }


if __name__ == "__main__":
    main()
