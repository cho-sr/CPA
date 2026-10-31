from __future__ import annotations

import argparse
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import torch

from .alignment import align_estimate_to_truth, match_sources_by_abs_cosine
from .bplus import solve_bplus_l1
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .linalg import condition_number
from .quantizer import consistency_rate, mean_hinge
from .run import append_jsonl, git_sha
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .victim import load_or_collect_victim_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bplus baseline with the same L1 prior budget.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--run-tag", default="v3_bplus_l1")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    stage1_cfg = config["stage1"]
    stage2_cfg = config.get("stage2", {})
    bplus_cfg = config.get("bplus", {})

    for batch_index in parse_batch_indices(args.batch_indices, int(args.n_batches)):
        start = time.perf_counter()
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=bool(args.force_cache))
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))
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
        mp_reference_stage1 = stage1
        if not bool(stage1_cfg.get("mp_truncation", True)):
            mp_reference_stage1 = estimate_mixing_and_sources(
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
                mp_truncation=True,
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
        result = solve_bplus_l1(
            A_hat=stage1.A_hat,
            S_hat=stage1.S_hat,
            delta=qobs.delta,
            lam_R=float(stage2_cfg.get("lam_R", 0.0)),
            radius_scale=float(bplus_cfg.get("radius_scale", 1.0)),
            max_iter=int(stage2_cfg.get("max_iter", 250)),
            chunk_size=int(stage2_cfg.get("chunk_size", 4096)),
            step_size=float(bplus_cfg.get("step_size", 1.0)),
        )
        A_aligned, S_aligned, alignment = align_estimate_to_truth(stage1.A_hat, result.S_star, victim.S_true)
        row = make_bplus_row(
            config=config,
            run_tag=str(args.run_tag),
            batch_index=batch_index,
            victim=victim,
            qobs=qobs,
            stage1=stage1,
            mp_reference_stage1=mp_reference_stage1,
            result=result,
            alignment=alignment,
            A_aligned=A_aligned,
            S_aligned=S_aligned,
            wall_time_sec=time.perf_counter() - start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} bplus recovered={rec}/{B_eff} ({recB}/{B}) source_cos={cosines} "
                "lam_R={lam_R:.3g} subthr09={subthr09} subthr07={subthr07} "
                "subthr05={subthr05} lower_hit={hit:.3f}".format(
                batch=batch_index,
                rec=row["metrics"]["n_sources_recovered"],
                B_eff=row["metrics"]["B_eff"],
                recB=row["metrics"]["n_sources_recovered"],
                B=row["metrics"]["B_requested"],
                cosines="[" + ", ".join(f"{value:.4f}" for value in row["metrics"]["source_cos_per_source"]) + "]",
                lam_R=row["bplus"]["lam_R"],
                subthr09=row["metrics"]["sub_threshold_recovered_cos_gt_0_9"],
                subthr07=row["metrics"]["sub_threshold_recovered_cos_gt_0_7"],
                subthr05=row["metrics"]["sub_threshold_recovered_cos_gt_0_5"],
                hit=row["bplus"]["lower_bound_hit_fraction"],
            ),
            flush=True,
        )


def make_bplus_row(
    *,
    config: dict[str, Any],
    run_tag: str,
    batch_index: int,
    victim: Any,
    qobs: Any,
    stage1: Any,
    mp_reference_stage1: Any,
    result: Any,
    alignment: Any,
    A_aligned: torch.Tensor,
    S_aligned: torch.Tensor,
    wall_time_sec: float,
) -> dict[str, Any]:
    labels = [int(label) for label in victim.labels.tolist()]
    losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    source_cos_per_truth = [0.0] * int(victim.S_true.shape[0])
    for true_idx, cosine in zip(alignment.true_indices, alignment.per_source_abs_cosine):
        source_cos_per_truth[int(true_idx)] = float(cosine)
    n_recovered = int(sum(value > 0.9 for value in alignment.per_source_abs_cosine))
    mp_reference_alignment = match_sources_by_abs_cosine(
        victim.S_true.detach().cpu().float(),
        mp_reference_stage1.S_hat.detach().cpu().float(),
    )
    mp_subthreshold_true_indices = [int(idx) for idx in mp_reference_alignment.unmatched_true_indices]
    threshold_counts = {
        "0.5": int(sum(source_cos_per_truth[int(idx)] > 0.5 for idx in mp_subthreshold_true_indices)),
        "0.7": int(sum(source_cos_per_truth[int(idx)] > 0.7 for idx in mp_subthreshold_true_indices)),
        "0.9": int(sum(source_cos_per_truth[int(idx)] > 0.9 for idx in mp_subthreshold_true_indices)),
    }
    config_for_row = deepcopy(config)
    config_for_row["run_tag"] = run_tag

    stage1_dict = asdict(stage1)
    stage1_dict.pop("A_hat", None)
    stage1_dict.pop("S_hat", None)
    stage1_dict.pop("Y_recon", None)
    mp_reference_stage1_dict = asdict(mp_reference_stage1)
    mp_reference_stage1_dict.pop("A_hat", None)
    mp_reference_stage1_dict.pop("S_hat", None)
    mp_reference_stage1_dict.pop("Y_recon", None)
    bplus_dict = asdict(result)
    bplus_dict.pop("S_star", None)

    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "milestone": "bplus_l1",
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": losses,
        "config": config_for_row,
        "metrics": {
            "method": "bplus",
            "B_requested": int(stage1.B_requested),
            "B_eff": int(stage1.B_eff),
            "source_cos_per_source": [float(x) for x in alignment.per_source_abs_cosine],
            "source_cos_per_truth": source_cos_per_truth,
            "source_cos": float(sum(alignment.per_source_abs_cosine) / max(1, len(alignment.per_source_abs_cosine))),
            "n_sources_recovered": n_recovered,
            "n_sources_recovered_over_beff": n_recovered,
            "recovered_over_beff": float(n_recovered / max(1, int(stage1.B_eff))),
            "recovered_over_B": float(n_recovered / max(1, int(stage1.B_requested))),
            "sub_threshold_recovered": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_9": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_7": threshold_counts["0.7"],
            "sub_threshold_recovered_cos_gt_0_5": threshold_counts["0.5"],
            "sub_threshold_recovered_by_threshold": threshold_counts,
            "sub_threshold_cosines": [
                source_cos_per_truth[int(idx)] for idx in mp_subthreshold_true_indices
            ],
            "consistency_rate": consistency_rate(A_aligned, S_aligned, qobs.Y_hat, qobs.delta),
            "residual_violation": mean_hinge(A_aligned, S_aligned, qobs.Y_hat, qobs.delta),
        },
        "diagnostics": {
            "cond_A_hat": condition_number(stage1.A_hat),
            "cond_A_true": condition_number(victim.A_true),
            "cond_A_true_matched": condition_number(
                victim.A_true.index_select(1, torch.tensor(alignment.true_indices, dtype=torch.long))
            ),
            "matched_true_indices": list(alignment.true_indices),
            "unmatched_true_indices": list(alignment.unmatched_true_indices),
            "mp_reference_B_eff": int(mp_reference_stage1.B_eff),
            "mp_reference_matched_true_indices": list(mp_reference_alignment.true_indices),
            "mp_subthreshold_true_indices": mp_subthreshold_true_indices,
            "mp_lam_over_edge_all": list(stage1.lam_over_mp_edge_all),
            "mp_reference_lam_over_edge_all": list(mp_reference_stage1.lam_over_mp_edge_all),
            "sub_threshold_recovered": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_9": threshold_counts["0.9"],
            "sub_threshold_recovered_cos_gt_0_7": threshold_counts["0.7"],
            "sub_threshold_recovered_cos_gt_0_5": threshold_counts["0.5"],
            "a_hat_norm_per_source": [float(x) for x in A_aligned.norm(dim=0).detach().cpu().tolist()],
            "a_true_norm_per_source": [float(x) for x in victim.A_true.norm(dim=0).detach().cpu().tolist()],
            "sample_loss_per_source": losses,
            "label_per_source": labels,
        },
        "stage1": stage1_dict,
        "stage1_for_mp_metrics": mp_reference_stage1_dict,
        "bplus": bplus_dict,
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
