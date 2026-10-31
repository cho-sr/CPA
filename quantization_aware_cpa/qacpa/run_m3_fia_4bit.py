from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
import torch

from .alignment import align_estimate_to_truth
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .fia import denormalize_imagenet, evaluate_images, run_direct_fia
from .linalg import relative_l2
from .run import append_jsonl, git_sha
from .stage1 import estimate_mixing_and_sources
from .stage2 import oracle_pinv_initialization, solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


CORE_FIGURE_METHODS = ("fp32", "naive", "A_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 3 upstream-style FIA for 4-bit sources.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="v5_m3_fia_4bit")
    parser.add_argument("--methods", default="fp32,naive,A,A_oracle")
    parser.add_argument("--fp32-mode", choices=("truth", "stage1"), default="truth")
    parser.add_argument("--batch-indices", default="0,1")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--n-iter-fi", type=int, default=25000)
    parser.add_argument("--n-log-fi", type=int, default=5000)
    parser.add_argument("--lr-fi", type=float, default=1.0e-1)
    parser.add_argument("--tv", type=float, default=1.0e-1)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--opt-fi", default="adam")
    parser.add_argument("--sch-fi", default="cosine")
    parser.add_argument("--seed-fi", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v5_m3_fia_4bit"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-fia", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    methods = parse_methods(args.methods)
    batch_indices = parse_batch_indices(args.batch_indices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.get("device", "auto")))

    start_all = time.perf_counter()
    target_sets = collect_target_sets(
        config=config,
        methods=methods,
        batch_indices=batch_indices,
        max_images=int(args.max_images),
        device=device,
        force_cache=bool(args.force_cache),
        fp32_mode=str(args.fp32_mode),
    )
    summary: dict[str, Any] = {}
    rec_by_method: dict[str, torch.Tensor] = {}
    meta_by_method: dict[str, list[dict[str, Any]]] = {}

    for method in methods:
        targets = target_sets[method]
        if targets["z"].numel() == 0:
            continue
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        torch.save(targets, method_dir / "targets.pt")

        rec_path = method_dir / "rec.pt"
        log_path = method_dir / "fia_log.json"
        if rec_path.is_file() and not bool(args.force_fia):
            rec = torch.load(rec_path, map_location="cpu")
            metrics = evaluate_images(x_true=targets["x"], rec=rec, device=device)
            fia_log: list[dict[str, float]] = []
            wall_time = 0.0
        else:
            result = run_direct_fia(
                target_z=targets["z"],
                x_true=targets["x"],
                device=device,
                n_iter=int(args.n_iter_fi),
                log_every=int(args.n_log_fi),
                lr=float(args.lr_fi),
                tv=float(args.tv),
                fi=float(args.fi),
                optimizer=str(args.opt_fi),
                scheduler=str(args.sch_fi),
                seed=int(args.seed_fi),
                progress_label=method,
            )
            rec = result.rec
            metrics = result.metrics
            fia_log = result.log
            wall_time = float(result.wall_time_sec)
            torch.save(rec, rec_path)
            log_path.write_text(json.dumps(fia_log, indent=2), encoding="utf-8")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metrics_rows = write_method_outputs(
            method_dir=method_dir,
            method=method,
            rec=rec,
            x_true=targets["x"],
            metadata=targets["metadata"],
            psnr=metrics.psnr,
            ssim=metrics.ssim,
            lpips=metrics.lpips,
        )
        rho, pvalue = safe_spearman(
            [float(row["per_sample_loss"]) for row in metrics_rows],
            [float(row["lpips"]) for row in metrics_rows],
        )
        summary[method] = {
            "n_images": len(metrics_rows),
            "psnr_mean": float(np.mean(metrics.psnr)),
            "ssim_mean": float(np.mean(metrics.ssim)),
            "lpips_mean": float(np.mean(metrics.lpips)),
            "spearman_loss_lpips": rho,
            "spearman_loss_lpips_pvalue": pvalue,
            "fia_wall_time_sec": wall_time,
            "target_source_cosine_mean": float(np.mean([row["target_source_cosine"] for row in metrics_rows])),
            "target_source_cosine_gt_0_5": int(sum(float(row["target_source_cosine"]) > 0.5 for row in metrics_rows)),
            "target_source_cosine_gt_0_7": int(sum(float(row["target_source_cosine"]) > 0.7 for row in metrics_rows)),
            "target_source_cosine_gt_0_9": int(sum(float(row["target_source_cosine"]) > 0.9 for row in metrics_rows)),
            "final_log": fia_log[-1] if fia_log else None,
        }
        rec_by_method[method] = rec
        meta_by_method[method] = list(targets["metadata"])
        print(
            "{method}: n={n} lpips={lpips:.4f} psnr={psnr:.3f} ssim={ssim:.3f} "
            "spearman={rho:.3f} wall={wall:.1f}s".format(
                method=method,
                n=len(metrics_rows),
                lpips=summary[method]["lpips_mean"],
                psnr=summary[method]["psnr_mean"],
                ssim=summary[method]["ssim_mean"],
                rho=summary[method]["spearman_loss_lpips"],
                wall=wall_time,
            ),
            flush=True,
        )

    if all(method in rec_by_method for method in CORE_FIGURE_METHODS):
        make_four_column_figures(
            output_dir=output_dir,
            target_sets=target_sets,
            rec_by_method=rec_by_method,
            meta_by_method=meta_by_method,
        )

    row = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": 3,
        "git_sha": git_sha(),
        "config": {
            **deepcopy(config),
            "run_tag": str(args.run_tag),
            "m3_fia": {
                "methods": methods,
                "fp32_mode": str(args.fp32_mode),
                "batch_indices": batch_indices,
                "max_images": int(args.max_images),
                "n_iter_fi": int(args.n_iter_fi),
                "n_log_fi": int(args.n_log_fi),
                "lr_fi": float(args.lr_fi),
                "tv": float(args.tv),
                "fi": float(args.fi),
                "opt_fi": str(args.opt_fi),
                "sch_fi": str(args.sch_fi),
                "seed_fi": int(args.seed_fi),
                "output_dir": str(output_dir),
            },
        },
        "metrics": summary,
        "batch_diagnostics": target_sets.get("_batch_diagnostics", []),
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(config["paths"]["results_jsonl"]), row)
    (output_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    write_rows_csv(output_dir / "batch_diagnostics.csv", target_sets.get("_batch_diagnostics", []))


def collect_target_sets(
    *,
    config: dict[str, Any],
    methods: list[str],
    batch_indices: list[int],
    max_images: int,
    device: torch.device,
    force_cache: bool,
    fp32_mode: str,
) -> dict[str, dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = {
        method: {"z_list": [], "x_list": [], "metadata": []} for method in methods
    }
    batch_diagnostics: list[dict[str, Any]] = []
    seed = int(config["seed"])
    stage1_cfg = config["stage1"]
    stage2_cfg = config.get("stage2", {})

    for batch_index in batch_indices:
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=force_cache)
        qobs = load_or_quantize_observation(victim, config, force=force_cache)
        labels = [int(label) for label in victim.labels.detach().cpu().view(-1).tolist()]
        quant_error = (qobs.Y_hat.detach().cpu().float() - victim.Y.detach().cpu().float())
        batch_diag: dict[str, Any] = {
            "batch_index": int(batch_index),
            "B": int(victim.S_true.shape[0]),
            "bits": int(qobs.bits),
            "rounding": str(qobs.rounding),
            "scale_granularity": str(qobs.scale_granularity),
            "bucket_size": qobs.bucket_size if qobs.bucket_size is not None else "",
            "labels_unique_count": int(len(set(labels))),
            "labels_duplicate_count": int(len(labels) - len(set(labels))),
            "quant_mse": float(quant_error.square().mean().item()),
            "quant_rel_l2": float(relative_l2(qobs.Y_hat.detach().cpu().float(), victim.Y.detach().cpu().float())),
            "quant_max_abs_error": float(quant_error.abs().max().item()),
            "delta_min": float(qobs.delta.detach().cpu().float().min().item()),
            "delta_max": float(qobs.delta.detach().cpu().float().max().item()),
            "delta_mean": float(qobs.delta.detach().cpu().float().mean().item()),
        }
        needed = {method for method in methods if len(accum[method]["metadata"]) < max_images}
        if not needed:
            break

        if "fp32" in needed and str(fp32_mode) == "truth":
            add_direct_targets(
                accum["fp32"],
                batch_index=batch_index,
                victim=victim,
                S=victim.S_true.detach().cpu().float().clamp_min(0.0),
                source_cosines=[1.0] * int(victim.S_true.shape[0]),
                method="fp32",
                max_images=max_images,
            )

        if "fp32" in needed and str(fp32_mode) == "stage1":
            fp32_n_components = min(
                int(stage1_cfg["n_components"]),
                effective_centered_rank(victim.Y),
            )
            stage1_fp32 = estimate_stage1(
                victim.Y,
                torch.zeros(victim.Y.shape[0], dtype=victim.Y.dtype),
                stage1_cfg=stage1_cfg,
                rounding="fp32",
                debias=False,
                mp_truncation=False,
                seed=seed + batch_index,
                n_components=fp32_n_components,
            )
            _, S_fp32, alignment_fp32 = align_estimate_to_truth(
                stage1_fp32.A_hat,
                stage1_fp32.S_hat,
                victim.S_true,
            )
            add_aligned_targets(
                accum["fp32"],
                batch_index=batch_index,
                victim=victim,
                true_indices=alignment_fp32.true_indices,
                S=S_fp32.abs(),
                source_cosines=alignment_fp32.per_source_abs_cosine,
                method="fp32",
                max_images=max_images,
            )

        S_naive_full: torch.Tensor | None = None
        if "naive" in needed or "A" in needed:
            stage1_naive = estimate_stage1(
                qobs.Y_hat,
                qobs.delta,
                stage1_cfg=stage1_cfg,
                rounding=qobs.rounding,
                debias=False,
                mp_truncation=bool(stage1_cfg.get("mp_truncation", True)),
                seed=seed + batch_index,
            )
            _, S_naive, alignment_naive = align_estimate_to_truth(
                stage1_naive.A_hat,
                stage1_naive.S_hat,
                victim.S_true,
            )
            batch_diag.update(stage1_result_diagnostics("naive_stage1", stage1_naive))
            S_naive_full, naive_filled = aligned_sources_to_truth_order(
                S_aligned=S_naive.abs(),
                true_indices=alignment_naive.true_indices,
                n_sources=int(victim.S_true.shape[0]),
                n_features=int(victim.S_true.shape[1]),
            )
            if not all(naive_filled):
                missing = [idx for idx, filled in enumerate(naive_filled) if not filled]
                raise RuntimeError(f"naive Stage-1 did not recover every source needed for A fallback: {missing}")
            if "naive" in needed:
                add_direct_targets(
                    accum["naive"],
                    batch_index=batch_index,
                    victim=victim,
                    S=S_naive_full,
                    source_cosines=direct_source_cosines(victim.S_true, S_naive_full),
                    method="naive",
                    max_images=max_images,
                )

        stage1_default = None
        if "A" in needed:
            if S_naive_full is None:
                raise RuntimeError("A fallback requires a naive full-B initialization")
            stage1_default = estimate_stage1(
                qobs.Y_hat,
                qobs.delta,
                stage1_cfg=stage1_cfg,
                rounding=qobs.rounding,
                debias=bool(stage1_cfg.get("debias", True)),
                mp_truncation=bool(stage1_cfg.get("mp_truncation", True)),
                seed=seed + batch_index,
            )
            batch_diag.update(stage1_result_diagnostics("A_stage1", stage1_default))
            result_A = solve_stage2_consistency(
                A=stage1_default.A_hat,
                Y_hat=qobs.Y_hat,
                delta=qobs.delta,
                S_init=stage1_default.S_hat,
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
            _, S_A, alignment_A = align_estimate_to_truth(
                stage1_default.A_hat,
                result_A.S_star,
                victim.S_true,
            )
            S_A_full = S_naive_full.clone()
            for local_row, true_idx in enumerate(alignment_A.true_indices):
                S_A_full[int(true_idx)] = S_A[local_row].detach().cpu().float().abs()
            add_direct_targets(
                accum["A"],
                batch_index=batch_index,
                victim=victim,
                S=S_A_full.clamp_min(0.0),
                source_cosines=direct_source_cosines(victim.S_true, S_A_full),
                method="A",
                max_images=max_images,
            )

        if "A_oracle" in needed:
            S_init = oracle_pinv_initialization(victim.A_true, qobs.Y_hat)
            result_oracle = solve_stage2_consistency(
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
            source_cos = direct_source_cosines(victim.S_true, result_oracle.S_star)
            add_direct_targets(
                accum["A_oracle"],
                batch_index=batch_index,
                victim=victim,
                S=result_oracle.S_star.clamp_min(0.0),
                source_cosines=source_cos,
                method="A_oracle",
                max_images=max_images,
            )

        batch_diagnostics.append(batch_diag)

    target_sets: dict[str, dict[str, Any]] = {}
    for method, data in accum.items():
        if data["z_list"]:
            target_sets[method] = {
                "z": torch.cat(data["z_list"], dim=0).float().contiguous(),
                "x": torch.cat(data["x_list"], dim=0).float().contiguous(),
                "metadata": data["metadata"],
            }
        else:
            target_sets[method] = {
                "z": torch.empty(0, 25088, dtype=torch.float32),
                "x": torch.empty(0, 3, 224, 224, dtype=torch.float32),
                "metadata": [],
            }
    target_sets["_batch_diagnostics"] = batch_diagnostics
    return target_sets


def stage1_result_diagnostics(prefix: str, result: Any) -> dict[str, Any]:
    ratios = list(getattr(result, "lam_over_mp_edge_all", []))
    B_eff = int(getattr(result, "B_eff"))
    return {
        f"{prefix}_B_requested": int(getattr(result, "B_requested")),
        f"{prefix}_B_eff": B_eff,
        f"{prefix}_mp_edge": float(getattr(result, "mp_edge")),
        f"{prefix}_sigma2": float(getattr(result, "sigma2")),
        f"{prefix}_c_mp": float(getattr(result, "c_mp")),
        f"{prefix}_mp_threshold": float(getattr(result, "mp_threshold")),
        f"{prefix}_kept_min_lam_over_mp_edge": float(min(ratios[:B_eff])) if ratios and B_eff > 0 else float("nan"),
        f"{prefix}_discarded_max_lam_over_mp_edge": float(max(ratios[B_eff:])) if ratios and B_eff < len(ratios) else float("nan"),
        f"{prefix}_lam_over_mp_edge_all": json.dumps(ratios),
        f"{prefix}_source_count_gt_0_9_aux": "",
    }


def estimate_stage1(
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    *,
    stage1_cfg: dict[str, Any],
    rounding: str,
    debias: bool,
    mp_truncation: bool,
    seed: int,
    n_components: int | None = None,
) -> Any:
    return estimate_mixing_and_sources(
        Y_hat,
        delta,
        n_components=int(stage1_cfg["n_components"] if n_components is None else n_components),
        rounding=rounding,
        debias=bool(debias),
        seed=int(seed),
        max_iter=int(stage1_cfg.get("fastica_max_iter", 1000)),
        tol=float(stage1_cfg.get("fastica_tol", 1.0e-4)),
        fun=str(stage1_cfg.get("fastica_fun", "logcosh")),
        c_mp=float(stage1_cfg.get("c_mp", 3.0)),
        mp_truncation=bool(mp_truncation),
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


def effective_centered_rank(Y: torch.Tensor) -> int:
    Y_cpu = Y.detach().cpu().float()
    Y_zc = Y_cpu - Y_cpu.mean(dim=1, keepdim=True)
    rank = int(torch.linalg.matrix_rank(Y_zc).item())
    if rank <= 0:
        raise RuntimeError("fp32 centered observation has zero effective rank")
    return rank


def aligned_sources_to_truth_order(
    *,
    S_aligned: torch.Tensor,
    true_indices: list[int],
    n_sources: int,
    n_features: int,
) -> tuple[torch.Tensor, list[bool]]:
    S_full = torch.zeros((int(n_sources), int(n_features)), dtype=S_aligned.detach().cpu().float().dtype)
    filled = [False] * int(n_sources)
    for local_row, true_idx in enumerate(true_indices):
        idx = int(true_idx)
        S_full[idx] = S_aligned[local_row].detach().cpu().float()
        filled[idx] = True
    return S_full, filled


def add_aligned_targets(
    data: dict[str, Any],
    *,
    batch_index: int,
    victim: Any,
    true_indices: list[int],
    S: torch.Tensor,
    source_cosines: list[float],
    method: str,
    max_images: int,
) -> None:
    for local_row, true_idx in enumerate(true_indices):
        if len(data["metadata"]) >= max_images:
            break
        idx = int(true_idx)
        data["z_list"].append(S[local_row : local_row + 1].detach().cpu())
        data["x_list"].append(victim.x[idx : idx + 1].detach().cpu())
        data["metadata"].append(sample_metadata(victim, batch_index, idx, method, float(source_cosines[local_row])))


def add_direct_targets(
    data: dict[str, Any],
    *,
    batch_index: int,
    victim: Any,
    S: torch.Tensor,
    source_cosines: list[float],
    method: str,
    max_images: int,
) -> None:
    for idx in range(int(S.shape[0])):
        if len(data["metadata"]) >= max_images:
            break
        data["z_list"].append(S[idx : idx + 1].detach().cpu())
        data["x_list"].append(victim.x[idx : idx + 1].detach().cpu())
        data["metadata"].append(sample_metadata(victim, batch_index, idx, method, float(source_cosines[idx])))


def sample_metadata(victim: Any, batch_index: int, source_index: int, method: str, source_cosine: float) -> dict[str, Any]:
    B = int(victim.S_true.shape[0])
    return {
        "global_index": int(batch_index) * B + int(source_index),
        "batch_index": int(batch_index),
        "source_index": int(source_index),
        "label": int(victim.labels[int(source_index)]),
        "per_sample_loss": float(victim.per_sample_loss[int(source_index)]),
        "image_path": str(victim.image_paths[int(source_index)]),
        "method": method,
        "target_source_cosine": float(source_cosine),
    }


def direct_source_cosines(S_true: torch.Tensor, S_est: torch.Tensor) -> list[float]:
    true_norm = S_true.detach().cpu().float() / S_true.detach().cpu().float().norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    est_norm = S_est.detach().cpu().float() / S_est.detach().cpu().float().norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    return [float(x) for x in (true_norm * est_norm).sum(dim=1).abs().tolist()]


def write_method_outputs(
    *,
    method_dir: Path,
    method: str,
    rec: torch.Tensor,
    x_true: torch.Tensor,
    metadata: list[dict[str, Any]],
    psnr: list[float],
    ssim: list[float],
    lpips: list[float],
) -> list[dict[str, Any]]:
    png_dir = method_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    truth = denormalize_imagenet(x_true.detach().cpu().float())
    rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        rec_path = png_dir / f"{method}_global{int(meta['global_index']):02d}_rec.png"
        orig_path = png_dir / f"{method}_global{int(meta['global_index']):02d}_orig.png"
        pair_path = png_dir / f"{method}_global{int(meta['global_index']):02d}_pair.png"
        save_tensor_png(rec[idx], rec_path)
        save_tensor_png(truth[idx], orig_path)
        save_pair(orig_path, rec_path, pair_path, left_label="original", right_label=method)
        row = {
            **meta,
            "psnr": float(psnr[idx]),
            "ssim": float(ssim[idx]),
            "lpips": float(lpips[idx]),
            "orig_png": str(orig_path),
            "rec_png": str(rec_path),
            "pair_png": str(pair_path),
        }
        rows.append(row)

    csv_path = method_dir / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_four_column_figures(
    *,
    output_dir: Path,
    target_sets: dict[str, dict[str, Any]],
    rec_by_method: dict[str, torch.Tensor],
    meta_by_method: dict[str, list[dict[str, Any]]],
) -> None:
    fp32_meta = meta_by_method["fp32"]
    common = []
    for meta in fp32_meta:
        gid = int(meta["global_index"])
        if all(find_meta_index(meta_by_method[method], gid) is not None for method in CORE_FIGURE_METHODS):
            common.append(gid)
    make_four_column_figure(
        output_dir / "figure_original_fp32_4bit_naive_4bit_A_oracle.png",
        common,
        target_sets,
        rec_by_method,
        meta_by_method,
    )
    loss_by_gid = {int(meta["global_index"]): float(meta["per_sample_loss"]) for meta in fp32_meta}
    make_four_column_figure(
        output_dir / "figure_original_fp32_4bit_naive_4bit_A_oracle_loss_sorted.png",
        sorted(common, key=lambda gid: loss_by_gid[gid]),
        target_sets,
        rec_by_method,
        meta_by_method,
    )


def make_four_column_figure(
    path: Path,
    global_indices: list[int],
    target_sets: dict[str, dict[str, Any]],
    rec_by_method: dict[str, torch.Tensor],
    meta_by_method: dict[str, list[dict[str, Any]]],
) -> None:
    columns = ["original", "fp32", "4-bit naive", "4-bit A_oracle"]
    methods = ["fp32", "fp32", "naive", "A_oracle"]
    tile = 224
    header = 24
    width = tile * len(columns)
    height = header + tile * len(global_indices)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((col * tile + 6, 5), label, fill=(0, 0, 0))
    truth = denormalize_imagenet(target_sets["fp32"]["x"].detach().cpu().float())
    for row_idx, gid in enumerate(global_indices):
        y = header + row_idx * tile
        fp32_idx = find_meta_index(meta_by_method["fp32"], gid)
        if fp32_idx is None:
            continue
        sheet.paste(tensor_to_pil(truth[fp32_idx]), (0, y))
        for col, method in enumerate(methods[1:], start=1):
            method_idx = find_meta_index(meta_by_method[method], gid)
            if method_idx is None:
                continue
            sheet.paste(tensor_to_pil(rec_by_method[method][method_idx]), (col * tile, y))
    sheet.save(path)


def find_meta_index(metadata: list[dict[str, Any]], global_index: int) -> int | None:
    for idx, meta in enumerate(metadata):
        if int(meta["global_index"]) == int(global_index):
            return idx
    return None


def save_tensor_png(tensor: torch.Tensor, path: Path) -> None:
    tensor_to_pil(tensor).save(path)


def save_pair(left_path: Path, right_path: Path, out_path: Path, *, left_label: str, right_label: str) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    header = 22
    pair = Image.new("RGB", (left.width + right.width, left.height + header), "white")
    pair.paste(left, (0, header))
    pair.paste(right, (left.width, header))
    draw = ImageDraw.Draw(pair)
    draw.text((4, 4), left_label, fill=(0, 0, 0))
    draw.text((left.width + 4, 4), right_label, fill=(0, 0, 0))
    pair.save(out_path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def safe_spearman(losses: list[float], lpips: list[float]) -> tuple[float, float]:
    if len(losses) < 2 or len(set(losses)) < 2 or len(set(lpips)) < 2:
        return float("nan"), float("nan")
    result = spearmanr(losses, lpips)
    return float(result.statistic), float(result.pvalue)


def parse_methods(raw: str) -> list[str]:
    allowed = {"fp32", "naive", "A", "A_oracle"}
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; allowed={sorted(allowed)}")
    return values


def parse_batch_indices(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--batch-indices must contain at least one integer")
    return values


if __name__ == "__main__":
    main()
