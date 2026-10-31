from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .alignment import align_estimate_to_truth, match_sources_by_abs_cosine
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number
from .quantizer import consistency_rate, mean_hinge
from .refine_a import solve_stage2_with_A_refinement
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .stage2 import solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 with estimated A_hat.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--run-tag", default="v4_A_cpaica")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--refine-A", dest="refine_A", action="store_true")
    parser.add_argument("--refine-a-steps", type=int, default=5)
    parser.add_argument("--refine-a-step-scale", type=float, default=1.0)
    parser.add_argument("--refine-a-chunk-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    device = resolve_device(str(config.get("device", "auto")))
    stage1_cfg = config["stage1"]
    stage2_cfg = config.get("stage2", {})

    for batch_index in parse_batch_indices(args.batch_indices, int(args.n_batches)):
        start = time.perf_counter()
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=bool(args.force_cache))
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))
        stage1 = estimate_stage1_from_config(
            qobs.Y_hat,
            qobs.delta,
            stage1_cfg=stage1_cfg,
            rounding=qobs.rounding,
            seed=seed + batch_index,
        )
        refine_info = None
        A_effective = stage1.A_hat
        if bool(args.refine_A):
            refined = solve_stage2_with_A_refinement(
                A=stage1.A_hat,
                Y_hat=qobs.Y_hat,
                delta=qobs.delta,
                S_init=stage1.S_hat,
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
                n_a_steps=int(args.refine_a_steps),
                a_step_size_scale=float(args.refine_a_step_scale),
                a_chunk_size=args.refine_a_chunk_size,
            )
            result = refined.stage2_result
            A_effective = refined.A_star
            refine_info = {
                "enabled": True,
                "n_a_steps_requested": refined.n_a_steps_requested,
                "n_a_steps_accepted": refined.n_a_steps_accepted,
                "a_step_size_scale": refined.a_step_size_scale,
                "a_chunk_size": refined.a_chunk_size,
                "a_step_logs": refined.a_step_logs,
                "wall_time_sec": refined.wall_time_sec,
            }
        else:
            result = solve_stage2_consistency(
                A=stage1.A_hat,
                Y_hat=qobs.Y_hat,
                delta=qobs.delta,
                S_init=stage1.S_hat,
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
            refine_info = {"enabled": False}
        A_aligned, S_aligned, alignment = align_estimate_to_truth(A_effective, result.S_star, victim.S_true)
        row = make_A_row(
            config=config,
            run_tag=str(args.run_tag),
            batch_index=batch_index,
            victim=victim,
            qobs=qobs,
            stage1=stage1,
            result=result,
            alignment=alignment,
            A_aligned=A_aligned,
            S_aligned=S_aligned,
            A_effective=A_effective,
            refine_info=refine_info,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} A recovered={rec}/{B_eff} ({recB}/{B}) source_cos={cosines} "
            "phase1_consistency={p1_cons:.6f} phase2_consistency={p2_cons:.6f} "
            "H={h0:.3g}->{h1:.3g}->{h2:.3g} lam_R={lam_R:.3g} "
            "subthr09={sub09} subthr07={sub07} subthr05={sub05}".format(
                batch=batch_index,
                rec=row["metrics"]["n_sources_recovered_over_beff"],
                B_eff=row["metrics"]["B_eff"],
                recB=row["metrics"]["n_sources_recovered"],
                B=row["metrics"]["B_requested"],
                cosines="[" + ", ".join(f"{value:.4f}" for value in row["metrics"]["source_cos_per_source"]) + "]",
                p1_cons=row["stage2"]["phase1_consistency_rate"],
                p2_cons=row["stage2"]["final_consistency_rate"],
                h0=row["diagnostics"]["per_column_H_initial_mean"],
                h1=row["stage2"]["phase1_H_mean"],
                h2=row["stage2"]["final_H_mean"],
                lam_R=row["stage2"]["lam_R"],
                sub09=row["metrics"]["sub_threshold_recovered_cos_gt_0_9"],
                sub07=row["metrics"]["sub_threshold_recovered_cos_gt_0_7"],
                sub05=row["metrics"]["sub_threshold_recovered_cos_gt_0_5"],
            ),
            flush=True,
        )


def estimate_stage1_from_config(
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    stage1_cfg: dict[str, Any],
    rounding: str,
    seed: int,
) -> Any:
    return estimate_mixing_and_sources(
        Y_hat,
        delta,
        n_components=int(stage1_cfg["n_components"]),
        rounding=rounding,
        debias=bool(stage1_cfg.get("debias", True)),
        seed=int(seed),
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


def make_A_row(
    *,
    config: dict[str, Any],
    run_tag: str,
    batch_index: int,
    victim: Any,
    qobs: Any,
    stage1: Any,
    result: Any,
    alignment: Any,
    A_aligned: torch.Tensor,
    S_aligned: torch.Tensor,
    wall_time_sec: float,
    A_effective: torch.Tensor | None = None,
    refine_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    a_true_norms = [float(value) for value in victim.A_true.norm(dim=0).detach().cpu().tolist()]

    initial_alignment = match_sources_by_abs_cosine(
        victim.S_true.detach().cpu().float(),
        stage1.S_hat.detach().cpu().float(),
    )
    source_cos_per_truth = cosine_per_truth(victim.S_true.shape[0], alignment)
    init_source_cos_per_truth = cosine_per_truth(victim.S_true.shape[0], initial_alignment)
    mp_detected_true_indices = [int(idx) for idx in initial_alignment.true_indices]
    mp_subthreshold_true_indices = [int(idx) for idx in initial_alignment.unmatched_true_indices]
    threshold_counts = {
        "0.5": int(sum(source_cos_per_truth[int(idx)] > 0.5 for idx in mp_subthreshold_true_indices)),
        "0.7": int(sum(source_cos_per_truth[int(idx)] > 0.7 for idx in mp_subthreshold_true_indices)),
        "0.9": int(sum(source_cos_per_truth[int(idx)] > 0.9 for idx in mp_subthreshold_true_indices)),
    }
    n_recovered = int(sum(value > 0.9 for value in alignment.per_source_abs_cosine))
    n_recovered_detected = int(sum(source_cos_per_truth[int(idx)] > 0.9 for idx in mp_detected_true_indices))
    recovered_indices = [idx for idx, value in enumerate(source_cos_per_truth) if value > 0.9]
    failed_indices = [idx for idx, value in enumerate(source_cos_per_truth) if value <= 0.9]
    source_records = [
        {
            "source_index": idx,
            "label": labels[idx],
            "loss": losses[idx],
            "a_true_norm": a_true_norms[idx],
            "source_cosine_initialization": init_source_cos_per_truth[idx],
            "source_cosine_final": source_cos_per_truth[idx],
            "source_cosine_delta": source_cos_per_truth[idx] - init_source_cos_per_truth[idx],
            "mp_detected": idx in set(mp_detected_true_indices),
            "recovered": source_cos_per_truth[idx] > 0.9,
        }
        for idx in range(len(labels))
    ]

    stage1_dict = asdict(stage1)
    stage1_dict.pop("A_hat", None)
    stage1_dict.pop("S_hat", None)
    stage1_dict.pop("Y_recon", None)
    stage2_dict = asdict(result)
    for key, value in list(stage2_dict.items()):
        if isinstance(value, torch.Tensor):
            stage2_dict.pop(key, None)

    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = run_tag
    A_for_diagnostics = stage1.A_hat if A_effective is None else A_effective.detach().cpu().float()

    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "milestone": "stage2_A",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "method": "A",
            "B_requested": int(stage1.B_requested),
            "B_eff": int(stage1.B_eff),
            "source_cos_per_source": [float(x) for x in alignment.per_source_abs_cosine],
            "source_cos_per_truth": source_cos_per_truth,
            "source_cos_initialization_per_source": [float(x) for x in initial_alignment.per_source_abs_cosine],
            "source_cos_initialization_per_truth": init_source_cos_per_truth,
            "source_cos_improvement_per_truth": [
                float(after - before) for before, after in zip(init_source_cos_per_truth, source_cos_per_truth)
            ],
            "source_cos": float(sum(alignment.per_source_abs_cosine) / max(1, len(alignment.per_source_abs_cosine))),
            "n_sources_recovered": n_recovered,
            "n_sources_recovered_over_beff": n_recovered_detected,
            "recovered_over_beff": float(n_recovered_detected / max(1, int(stage1.B_eff))),
            "recovered_over_B": float(n_recovered / max(1, int(stage1.B_requested))),
            "sub_threshold_recovered": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_9": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_7": threshold_counts["0.7"],
            "sub_threshold_recovered_cos_gt_0_5": threshold_counts["0.5"],
            "sub_threshold_recovered_by_threshold": threshold_counts,
            "sub_threshold_cosines": [source_cos_per_truth[int(idx)] for idx in mp_subthreshold_true_indices],
            "consistency_rate": consistency_rate(A_aligned, S_aligned, qobs.Y_hat, qobs.delta),
            "residual_violation": mean_hinge(A_aligned, S_aligned, qobs.Y_hat, qobs.delta),
            "initial_consistency_rate": result.initial_consistency_rate,
            "initial_residual_violation": result.initial_residual_violation,
        },
        "diagnostics": {
            "cond_A_hat": condition_number(A_for_diagnostics),
            "cond_A_hat_initial": condition_number(stage1.A_hat),
            "cond_A_true": condition_number(victim.A_true),
            "cond_A_true_matched": condition_number(
                victim.A_true.index_select(1, torch.tensor(alignment.true_indices, dtype=torch.long))
            ),
            "matched_true_indices": list(alignment.true_indices),
            "unmatched_true_indices": list(alignment.unmatched_true_indices),
            "mp_detected_true_indices": mp_detected_true_indices,
            "mp_subthreshold_true_indices": mp_subthreshold_true_indices,
            "mp_lam_over_edge_all": list(stage1.lam_over_mp_edge_all),
            "mp_c_mp": stage1.c_mp,
            "mp_edge": stage1.mp_edge,
            "recovered_source_indices": recovered_indices,
            "failed_source_indices": failed_indices,
            "a_hat_norm_per_source": [float(x) for x in A_aligned.norm(dim=0).detach().cpu().tolist()],
            "a_true_norm_per_source": a_true_norms,
            "sample_loss_per_source": losses,
            "label_per_source": labels,
            "source_records": source_records,
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
        },
        "refine_A": refine_info or {"enabled": False},
        "stage1": stage1_dict,
        "stage2": stage2_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def cosine_per_truth(B: int, alignment: Any) -> list[float]:
    values = [0.0] * int(B)
    for true_idx, cosine in zip(alignment.true_indices, alignment.per_source_abs_cosine):
        values[int(true_idx)] = float(cosine)
    return values


def parse_batch_indices(raw: str | None, n_batches: int) -> list[int]:
    if raw is None:
        return list(range(int(n_batches)))
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--batch-indices must contain at least one integer")
    return values


if __name__ == "__main__":
    main()
