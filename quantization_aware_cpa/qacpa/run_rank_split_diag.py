from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .alignment import align_estimate_to_truth
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number, row_normalized
from .quantizer import consistency_rate, mean_hinge
from .run import append_jsonl, git_sha
from .stage1 import estimate_mixing_and_sources
from .stage2 import solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage-2 rank loss with oracle discarded columns.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--run-tag", default="v5_diag_rank_split")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
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
            seed=int(config["seed"]) + batch_index,
        )
        A_aug, S_init, detected, subthreshold, init_cos = build_augmented_system(
            victim=victim,
            stage1=stage1,
            Y_hat=qobs.Y_hat,
        )
        result = solve_stage2_consistency(
            A=A_aug,
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
        final_cos = direct_source_cosines(victim.S_true, result.S_star)
        row = make_row(
            config=config,
            run_tag=str(args.run_tag),
            batch_index=batch_index,
            victim=victim,
            qobs=qobs,
            stage1=stage1,
            result=result,
            A_aug=A_aug,
            S_init=S_init,
            detected=detected,
            subthreshold=subthreshold,
            init_cos=init_cos,
            final_cos=final_cos,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} rank_split recovered={rec}/{B} subthr09={s09} subthr07={s07} "
            "subthr05={s05} H={h0:.3g}->{h1:.3g}->{h2:.3g} cos={cos}".format(
                batch=batch_index,
                rec=row["metrics"]["n_sources_recovered"],
                B=row["metrics"]["B_requested"],
                s09=row["metrics"]["sub_threshold_recovered_cos_gt_0_9"],
                s07=row["metrics"]["sub_threshold_recovered_cos_gt_0_7"],
                s05=row["metrics"]["sub_threshold_recovered_cos_gt_0_5"],
                h0=row["diagnostics"]["per_column_H_initial_mean"],
                h1=row["stage2"]["phase1_H_mean"],
                h2=row["stage2"]["final_H_mean"],
                cos="[" + ", ".join(f"{value:.4f}" for value in final_cos) + "]",
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


def build_augmented_system(
    *,
    victim: Any,
    stage1: Any,
    Y_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[float]]:
    A_keep, S_keep, alignment = align_estimate_to_truth(stage1.A_hat, stage1.S_hat, victim.S_true)
    B = int(victim.S_true.shape[0])
    keep_by_true = {
        int(true_idx): (A_keep[:, local_idx], S_keep[local_idx])
        for local_idx, true_idx in enumerate(alignment.true_indices)
    }
    detected = [int(idx) for idx in alignment.true_indices]
    subthreshold = [idx for idx in range(B) if idx not in set(detected)]

    A_cols: list[torch.Tensor] = []
    for idx in range(B):
        if idx in keep_by_true:
            A_cols.append(keep_by_true[idx][0])
        else:
            A_cols.append(victim.A_true[:, idx].detach().cpu().float())
    A_aug = torch.stack(A_cols, dim=1).float().contiguous()
    S_pinv = torch.linalg.pinv(A_aug).matmul(Y_hat.detach().cpu().float()).clamp_min(0.0)
    S_init = S_pinv.clone()
    for idx, (_, s_row) in keep_by_true.items():
        S_init[idx] = s_row.detach().cpu().float().clamp_min(0.0)
    init_cos = direct_source_cosines(victim.S_true, S_init)
    return A_aug, S_init.contiguous(), detected, subthreshold, init_cos


def make_row(
    *,
    config: dict[str, Any],
    run_tag: str,
    batch_index: int,
    victim: Any,
    qobs: Any,
    stage1: Any,
    result: Any,
    A_aug: torch.Tensor,
    S_init: torch.Tensor,
    detected: list[int],
    subthreshold: list[int],
    init_cos: list[float],
    final_cos: list[float],
    wall_time_sec: float,
) -> dict[str, Any]:
    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = run_tag
    stage1_dict = asdict(stage1)
    stage1_dict.pop("A_hat", None)
    stage1_dict.pop("S_hat", None)
    stage1_dict.pop("Y_recon", None)
    stage2_dict = asdict(result)
    for key, value in list(stage2_dict.items()):
        if isinstance(value, torch.Tensor):
            stage2_dict.pop(key, None)
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    threshold_counts = threshold_summary(final_cos, subthreshold)
    recovered = int(sum(value > 0.9 for value in final_cos))
    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "milestone": "rank_split_diag",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "method": "rank_split_diag_oracle_extra_columns",
            "B_requested": int(victim.S_true.shape[0]),
            "B_eff": int(stage1.B_eff),
            "source_cos_initialization_per_truth": init_cos,
            "source_cos_per_truth": final_cos,
            "source_cos": float(sum(final_cos) / max(1, len(final_cos))),
            "n_sources_recovered": recovered,
            "recovered_over_B": float(recovered / max(1, int(victim.S_true.shape[0]))),
            "n_sources_recovered_over_beff": int(sum(final_cos[idx] > 0.9 for idx in detected)),
            "recovered_over_beff": float(sum(final_cos[idx] > 0.9 for idx in detected) / max(1, len(detected))),
            "sub_threshold_recovered": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_9": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_7": threshold_counts["0.7"],
            "sub_threshold_recovered_cos_gt_0_5": threshold_counts["0.5"],
            "sub_threshold_recovered_by_threshold": threshold_counts,
            "sub_threshold_cosines": [float(final_cos[idx]) for idx in subthreshold],
            "consistency_rate": consistency_rate(A_aug, result.S_star, qobs.Y_hat, qobs.delta),
            "residual_violation": mean_hinge(A_aug, result.S_star, qobs.Y_hat, qobs.delta),
        },
        "diagnostics": {
            "cond_A_aug": condition_number(A_aug),
            "cond_A_hat_keep": condition_number(stage1.A_hat),
            "cond_A_true": condition_number(victim.A_true),
            "detected_true_indices": detected,
            "mp_subthreshold_true_indices": subthreshold,
            "a_aug_norm_per_source": [float(x) for x in A_aug.norm(dim=0).detach().cpu().tolist()],
            "a_true_norm_per_source": [float(x) for x in victim.A_true.norm(dim=0).detach().cpu().tolist()],
            "sample_loss_per_source": losses,
            "label_per_source": labels,
            "per_column_H_initial_mean": float(result.per_column_H_initial.detach().cpu().mean()),
            "per_column_H_initial_max": float(result.per_column_H_initial.detach().cpu().max()),
            "per_column_H_phase1_mean": float(result.per_column_H_phase1.detach().cpu().mean()),
            "per_column_H_phase1_max": float(result.per_column_H_phase1.detach().cpu().max()),
            "per_column_H_final_mean": float(result.per_column_H_final.detach().cpu().mean()),
            "per_column_H_final_max": float(result.per_column_H_final.detach().cpu().max()),
            "input_guard_reject_fraction": result.input_guard_reject_fraction,
            "phase1_chunk_iteration_mean": statistics.fmean(result.phase1_chunk_iterations),
            "phase2_chunk_iteration_mean": statistics.fmean(result.phase2_chunk_iterations),
        },
        "stage1": stage1_dict,
        "stage2": stage2_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def direct_source_cosines(S_true: torch.Tensor, S_est: torch.Tensor) -> list[float]:
    similarity = row_normalized(S_true.detach().cpu().float()) * row_normalized(S_est.detach().cpu().float())
    return [float(value) for value in similarity.sum(dim=1).abs().cpu().tolist()]


def threshold_summary(source_cos: list[float], indices: list[int]) -> dict[str, int]:
    return {
        "0.5": int(sum(source_cos[idx] > 0.5 for idx in indices)),
        "0.7": int(sum(source_cos[idx] > 0.7 for idx in indices)),
        "0.9": int(sum(source_cos[idx] > 0.9 for idx in indices)),
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
