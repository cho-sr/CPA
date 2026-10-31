from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .alignment import align_estimate_to_truth, match_sources_by_abs_cosine
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number, relative_l2
from .quantizer import consistency_rate, mean_hinge
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .stage2 import solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual rank-split Stage 2.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--run-tag", default="v5_rank_split_residual")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=Path("results/v5_rank_split_residual.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    device = resolve_device(str(config.get("device", "auto")))
    stage1_cfg = config["stage1"]
    stage2_cfg = config.get("stage2", {})
    rank_cfg = config.get("rank_split", {})
    summary_rows: list[dict[str, Any]] = []

    for batch_index in parse_batch_indices(args.batch_indices, int(args.n_batches)):
        start = time.perf_counter()
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=bool(args.force_cache))
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))

        stage1_keep = estimate_stage1_from_config(
            qobs.Y_hat,
            qobs.delta,
            stage1_cfg=stage1_cfg,
            rounding=qobs.rounding,
            seed=seed + batch_index,
            device=device,
        )
        rank_system = build_rank_split_system(
            Y_hat=qobs.Y_hat,
            delta=qobs.delta,
            rounding=qobs.rounding,
            stage1_keep=stage1_keep,
            stage1_cfg=stage1_cfg,
            rank_cfg=rank_cfg,
            seed=seed + 100_000 + batch_index,
            device=device,
        )

        result = solve_stage2_consistency(
            A=rank_system["A_full"],
            Y_hat=qobs.Y_hat,
            delta=qobs.delta,
            S_init=rank_system["S_init"],
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
            row_update_scale=rank_system["row_update_scale"],
        )

        _, S_aligned, alignment = align_estimate_to_truth(
            rank_system["A_full"],
            result.S_star,
            victim.S_true,
        )
        keep_alignment = match_sources_by_abs_cosine(
            victim.S_true.detach().cpu().float(),
            stage1_keep.S_hat.detach().cpu().float(),
        )
        extra_alignment = None
        if rank_system["stage1_extra"] is not None:
            extra_alignment = match_sources_by_abs_cosine(
                victim.S_true.detach().cpu().float(),
                rank_system["stage1_extra"].S_hat.detach().cpu().float(),
            )
        row = make_rank_split_row(
            config=config,
            run_tag=str(args.run_tag),
            batch_index=batch_index,
            victim=victim,
            qobs=qobs,
            stage1_keep=stage1_keep,
            stage1_extra=rank_system["stage1_extra"],
            result=result,
            alignment=alignment,
            keep_alignment=keep_alignment,
            extra_alignment=extra_alignment,
            A_full=rank_system["A_full"],
            S_init=rank_system["S_init"],
            S_aligned=S_aligned,
            residual=rank_system["residual"],
            residual_recon=rank_system["residual_recon"],
            row_update_scale=rank_system["row_update_scale"],
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        summary_rows.append(flatten_summary_row(row))
        print(
            "batch={batch} rank_split residual recovered={rec}/{B} beff={beff} "
            "subthr09={s09} subthr07={s07} subthr05={s05} "
            "H={h0:.3g}->{h1:.3g}->{h2:.3g} init_rel={init_rel:.4g} final_rel={final_rel:.4g} "
            "cos={cos}".format(
                batch=batch_index,
                rec=row["metrics"]["n_sources_recovered"],
                B=row["metrics"]["B_requested"],
                beff=row["metrics"]["B_eff"],
                s09=row["metrics"]["sub_threshold_recovered_cos_gt_0_9"],
                s07=row["metrics"]["sub_threshold_recovered_cos_gt_0_7"],
                s05=row["metrics"]["sub_threshold_recovered_cos_gt_0_5"],
                h0=row["diagnostics"]["per_column_H_initial_mean"],
                h1=row["stage2"]["phase1_H_mean"],
                h2=row["stage2"]["final_H_mean"],
                init_rel=row["diagnostics"]["rank_split_init_reconstruction_rel_l2"],
                final_rel=row["diagnostics"]["rank_split_final_reconstruction_rel_l2"],
                cos="[" + ", ".join(f"{value:.4f}" for value in row["metrics"]["source_cos_per_truth"]) + "]",
            ),
            flush=True,
        )

    if summary_rows:
        write_summary_csv(Path(args.summary_csv), summary_rows)


def build_rank_split_system(
    *,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    rounding: str,
    stage1_keep: Any,
    stage1_cfg: dict[str, Any],
    rank_cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    B_requested = int(stage1_keep.B_requested)
    B_eff = int(stage1_keep.B_eff)
    n_extra = max(0, B_requested - B_eff)
    keep_update_scale = float(rank_cfg.get("keep_update_scale", 0.05))
    extra_update_scale = float(rank_cfg.get("extra_update_scale", 1.0))

    A_keep = stage1_keep.A_hat.detach().cpu().float()
    S_keep = stage1_keep.S_hat.detach().cpu().float()
    residual = Y_hat.detach().cpu().float() - A_keep.matmul(S_keep)
    stage1_extra = None
    residual_recon = torch.zeros_like(residual)

    if n_extra > 0:
        stage1_extra = estimate_mixing_and_sources(
            residual,
            delta.detach().cpu().float(),
            n_components=n_extra,
            rounding=rounding,
            debias=bool(rank_cfg.get("extra_debias", False)),
            seed=int(seed),
            max_iter=int(stage1_cfg.get("fastica_max_iter", 1000)),
            tol=float(stage1_cfg.get("fastica_tol", 1.0e-4)),
            fun=str(stage1_cfg.get("fastica_fun", "logcosh")),
            c_mp=float(rank_cfg.get("extra_c_mp", 0.0)),
            mp_truncation=False,
            fastica_algorithm=str(stage1_cfg.get("fastica_algorithm", "parallel")),
            ica_method=str(rank_cfg.get("extra_ica_method", stage1_cfg.get("ica_method", "cpa"))),
            cpa_ica_max_iter=int(rank_cfg.get("extra_cpa_ica_max_iter", 5000)),
            cpa_ica_lr=float(rank_cfg.get("extra_cpa_ica_lr", stage1_cfg.get("cpa_ica_lr", 1.0e-3))),
            cpa_ica_optimizer=str(rank_cfg.get("extra_cpa_ica_optimizer", stage1_cfg.get("cpa_ica_optimizer", "adam"))),
            cpa_ica_decor=float(rank_cfg.get("extra_cpa_ica_decor", stage1_cfg.get("cpa_ica_decor", 5.3))),
            cpa_ica_T=float(rank_cfg.get("extra_cpa_ica_T", stage1_cfg.get("cpa_ica_T", 7.7))),
            cpa_ica_ne=float(rank_cfg.get("extra_cpa_ica_ne", stage1_cfg.get("cpa_ica_ne", 1.0))),
            cpa_ica_nv=float(rank_cfg.get("extra_cpa_ica_nv", stage1_cfg.get("cpa_ica_nv", 0.13))),
            cpa_ica_l1=float(rank_cfg.get("extra_cpa_ica_l1", stage1_cfg.get("cpa_ica_l1", 5.0))),
            cpa_ica_tv=float(rank_cfg.get("extra_cpa_ica_tv", stage1_cfg.get("cpa_ica_tv", 0.1))),
            cpa_ica_log_every=int(rank_cfg.get("extra_cpa_ica_log_every", 0)),
            device=device,
        )
        A_full = torch.cat([A_keep, stage1_extra.A_hat.detach().cpu().float()], dim=1).contiguous()
        S_init = torch.cat([S_keep, stage1_extra.S_hat.detach().cpu().float()], dim=0).contiguous()
        residual_recon = stage1_extra.A_hat.detach().cpu().float().matmul(
            stage1_extra.S_hat.detach().cpu().float()
        )
        row_update_scale = torch.cat(
            [
                torch.full((B_eff,), keep_update_scale, dtype=torch.float32),
                torch.full((n_extra,), extra_update_scale, dtype=torch.float32),
            ],
            dim=0,
        )
    else:
        A_full = A_keep.contiguous()
        S_init = S_keep.contiguous()
        row_update_scale = torch.full((B_eff,), keep_update_scale, dtype=torch.float32)

    return {
        "A_full": A_full,
        "S_init": S_init,
        "residual": residual,
        "residual_recon": residual_recon,
        "stage1_extra": stage1_extra,
        "row_update_scale": row_update_scale,
    }


def estimate_stage1_from_config(
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    stage1_cfg: dict[str, Any],
    rounding: str,
    seed: int,
    device: torch.device,
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
        device=device,
    )


def make_rank_split_row(
    *,
    config: dict[str, Any],
    run_tag: str,
    batch_index: int,
    victim: Any,
    qobs: Any,
    stage1_keep: Any,
    stage1_extra: Any | None,
    result: Any,
    alignment: Any,
    keep_alignment: Any,
    extra_alignment: Any | None,
    A_full: torch.Tensor,
    S_init: torch.Tensor,
    S_aligned: torch.Tensor,
    residual: torch.Tensor,
    residual_recon: torch.Tensor,
    row_update_scale: torch.Tensor,
    wall_time_sec: float,
) -> dict[str, Any]:
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    source_cos_per_truth = cosine_per_truth(victim.S_true.shape[0], alignment)
    init_alignment = match_sources_by_abs_cosine(
        victim.S_true.detach().cpu().float(),
        S_init.detach().cpu().float(),
    )
    init_source_cos_per_truth = cosine_per_truth(victim.S_true.shape[0], init_alignment)
    mp_detected_true_indices = [int(idx) for idx in keep_alignment.true_indices]
    mp_subthreshold_true_indices = [idx for idx in range(victim.S_true.shape[0]) if idx not in set(mp_detected_true_indices)]
    threshold_counts = threshold_summary(source_cos_per_truth, mp_subthreshold_true_indices)
    n_recovered = int(sum(value > 0.9 for value in source_cos_per_truth))
    n_recovered_detected = int(sum(source_cos_per_truth[idx] > 0.9 for idx in mp_detected_true_indices))
    recovered_indices = [idx for idx, value in enumerate(source_cos_per_truth) if value > 0.9]
    failed_indices = [idx for idx, value in enumerate(source_cos_per_truth) if value <= 0.9]

    stage1_keep_dict = asdict(stage1_keep)
    for key in ("A_hat", "S_hat", "Y_recon"):
        stage1_keep_dict.pop(key, None)
    stage1_extra_dict = None
    if stage1_extra is not None:
        stage1_extra_dict = asdict(stage1_extra)
        for key in ("A_hat", "S_hat", "Y_recon"):
            stage1_extra_dict.pop(key, None)
    stage2_dict = asdict(result)
    for key, value in list(stage2_dict.items()):
        if isinstance(value, torch.Tensor):
            stage2_dict.pop(key, None)

    A_cpu = A_full.detach().cpu().float()
    S_init_cpu = S_init.detach().cpu().float()
    S_final_cpu = result.S_star.detach().cpu().float()
    Y_cpu = qobs.Y_hat.detach().cpu().float()
    source_records = [
        {
            "source_index": idx,
            "label": labels[idx],
            "loss": losses[idx],
            "a_true_norm": float(victim.A_true[:, idx].norm().detach().cpu()),
            "source_cosine_initialization": init_source_cos_per_truth[idx],
            "source_cosine_final": source_cos_per_truth[idx],
            "source_cosine_delta": source_cos_per_truth[idx] - init_source_cos_per_truth[idx],
            "mp_detected": idx in set(mp_detected_true_indices),
            "recovered": source_cos_per_truth[idx] > 0.9,
        }
        for idx in range(len(labels))
    ]

    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = run_tag
    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "milestone": "rank_split_residual",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "method": "rank_split_residual",
            "B_requested": int(stage1_keep.B_requested),
            "B_eff": int(stage1_keep.B_eff),
            "source_cos_per_source": [float(x) for x in alignment.per_source_abs_cosine],
            "source_cos_per_truth": source_cos_per_truth,
            "source_cos_initialization_per_source": [float(x) for x in init_alignment.per_source_abs_cosine],
            "source_cos_initialization_per_truth": init_source_cos_per_truth,
            "source_cos_improvement_per_truth": [
                float(after - before) for before, after in zip(init_source_cos_per_truth, source_cos_per_truth)
            ],
            "source_cos": float(sum(source_cos_per_truth) / max(1, len(source_cos_per_truth))),
            "n_sources_recovered": n_recovered,
            "n_sources_recovered_over_beff": n_recovered_detected,
            "recovered_over_beff": float(n_recovered_detected / max(1, int(stage1_keep.B_eff))),
            "recovered_over_B": float(n_recovered / max(1, int(stage1_keep.B_requested))),
            "sub_threshold_recovered": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_9": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_7": threshold_counts["0.7"],
            "sub_threshold_recovered_cos_gt_0_5": threshold_counts["0.5"],
            "sub_threshold_recovered_by_threshold": threshold_counts,
            "sub_threshold_cosines": [source_cos_per_truth[int(idx)] for idx in mp_subthreshold_true_indices],
            "consistency_rate": consistency_rate(A_cpu, result.S_star, qobs.Y_hat, qobs.delta),
            "residual_violation": mean_hinge(A_cpu, result.S_star, qobs.Y_hat, qobs.delta),
            "initial_consistency_rate": result.initial_consistency_rate,
            "initial_residual_violation": result.initial_residual_violation,
        },
        "diagnostics": {
            "cond_A_rank_split": condition_number(A_cpu),
            "cond_A_hat_keep": condition_number(stage1_keep.A_hat),
            "cond_A_extra": condition_number(stage1_extra.A_hat) if stage1_extra is not None else None,
            "cond_A_true": condition_number(victim.A_true),
            "cond_A_true_matched": condition_number_for_columns(victim.A_true, recovered_indices),
            "matched_true_indices": list(alignment.true_indices),
            "unmatched_true_indices": list(alignment.unmatched_true_indices),
            "mp_detected_true_indices": mp_detected_true_indices,
            "mp_subthreshold_true_indices": mp_subthreshold_true_indices,
            "extra_estimated_true_indices": list(extra_alignment.true_indices) if extra_alignment is not None else [],
            "extra_estimated_cosines": list(extra_alignment.per_source_abs_cosine) if extra_alignment is not None else [],
            "mp_lam_over_edge_all": list(stage1_keep.lam_over_mp_edge_all),
            "mp_c_mp": stage1_keep.c_mp,
            "mp_edge": stage1_keep.mp_edge,
            "recovered_source_indices": recovered_indices,
            "failed_source_indices": failed_indices,
            "a_hat_norm_per_source": [float(x) for x in A_cpu.norm(dim=0).detach().cpu().tolist()],
            "a_true_norm_per_source": [float(x) for x in victim.A_true.norm(dim=0).detach().cpu().tolist()],
            "sample_loss_per_source": losses,
            "label_per_source": labels,
            "source_records": source_records,
            "rank_split_residual_rel_l2": relative_l2(residual, Y_cpu),
            "rank_split_residual_recon_rel_l2": relative_l2(residual_recon, residual) if residual.numel() else 0.0,
            "rank_split_init_reconstruction_rel_l2": relative_l2(A_cpu.matmul(S_init_cpu), Y_cpu),
            "rank_split_final_reconstruction_rel_l2": relative_l2(A_cpu.matmul(S_final_cpu), Y_cpu),
            "row_update_scale_per_component": [float(x) for x in row_update_scale.detach().cpu().tolist()],
            "keep_update_scale": float(row_update_scale[: int(stage1_keep.B_eff)].median().detach().cpu()),
            "extra_update_scale": (
                float(row_update_scale[int(stage1_keep.B_eff) :].median().detach().cpu())
                if row_update_scale.numel() > int(stage1_keep.B_eff)
                else None
            ),
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
        "stage1": stage1_keep_dict,
        "stage1_extra": stage1_extra_dict,
        "stage2": stage2_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def flatten_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    diag = row["diagnostics"]
    return {
        "run_tag": row["run_tag"],
        "batch_index": row["batch_index"],
        "B_requested": metrics["B_requested"],
        "B_eff": metrics["B_eff"],
        "recovered_over_B": metrics["recovered_over_B"],
        "recovered_over_beff": metrics["recovered_over_beff"],
        "n_sources_recovered": metrics["n_sources_recovered"],
        "sub_threshold_recovered_cos_gt_0_9": metrics["sub_threshold_recovered_cos_gt_0_9"],
        "sub_threshold_recovered_cos_gt_0_7": metrics["sub_threshold_recovered_cos_gt_0_7"],
        "sub_threshold_recovered_cos_gt_0_5": metrics["sub_threshold_recovered_cos_gt_0_5"],
        "source_cos_per_truth": format_float_list(metrics["source_cos_per_truth"]),
        "sub_threshold_cosines": format_float_list(metrics["sub_threshold_cosines"]),
        "extra_estimated_true_indices": " ".join(str(x) for x in diag["extra_estimated_true_indices"]),
        "extra_estimated_cosines": format_float_list(diag["extra_estimated_cosines"]),
        "H_initial_mean": diag["per_column_H_initial_mean"],
        "H_phase1_mean": diag["per_column_H_phase1_mean"],
        "H_final_mean": diag["per_column_H_final_mean"],
        "initial_consistency_rate": diag["initial_consistency_rate"],
        "phase1_consistency_rate": diag["phase1_consistency_rate"],
        "final_consistency_rate": diag["final_consistency_rate"],
        "rank_split_residual_rel_l2": diag["rank_split_residual_rel_l2"],
        "rank_split_residual_recon_rel_l2": diag["rank_split_residual_recon_rel_l2"],
        "rank_split_init_reconstruction_rel_l2": diag["rank_split_init_reconstruction_rel_l2"],
        "rank_split_final_reconstruction_rel_l2": diag["rank_split_final_reconstruction_rel_l2"],
        "wall_time_sec": row["wall_time_sec"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cosine_per_truth(B: int, alignment: Any) -> list[float]:
    values = [0.0] * int(B)
    for true_idx, cosine in zip(alignment.true_indices, alignment.per_source_abs_cosine):
        values[int(true_idx)] = float(cosine)
    return values


def threshold_summary(source_cos: list[float], indices: list[int]) -> dict[str, int]:
    return {
        "0.5": int(sum(source_cos[idx] > 0.5 for idx in indices)),
        "0.7": int(sum(source_cos[idx] > 0.7 for idx in indices)),
        "0.9": int(sum(source_cos[idx] > 0.9 for idx in indices)),
    }


def condition_number_for_columns(A: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return float("nan")
    order = torch.tensor(indices, dtype=torch.long, device=A.device)
    return condition_number(A.index_select(1, order))


def format_float_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def parse_batch_indices(raw: str | None, n_batches: int) -> list[int]:
    if raw is None:
        return list(range(int(n_batches)))
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--batch-indices must contain at least one integer")
    return values


if __name__ == "__main__":
    main()
