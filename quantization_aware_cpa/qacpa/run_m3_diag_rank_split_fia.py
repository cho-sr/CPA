from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .fia import FIAMetrics, FIAResult, evaluate_images, run_direct_fia
from .run import append_jsonl, git_sha
from .run_m3_fia_4bit import write_method_outputs, safe_spearman, sample_metadata
from .run_rank_split_diag import build_augmented_system, direct_source_cosines, estimate_stage1_from_config
from .stage2 import solve_stage2_consistency
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIA on oracle-extra rank-split recovered sources.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="v6_diag_rank_split_fia")
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-indices", default=None)
    parser.add_argument("--cos-threshold", type=float, default=0.9)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--n-iter-fi", type=int, default=25000)
    parser.add_argument("--n-log-fi", type=int, default=5000)
    parser.add_argument("--lr-fi", type=float, default=1.0e-1)
    parser.add_argument("--tv", type=float, default=1.0e-1)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--opt-fi", default="adam")
    parser.add_argument("--sch-fi", default="cosine")
    parser.add_argument("--seed-fi", type=int, default=42)
    parser.add_argument("--fia-chunk-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v6_diag_rank_split_fia"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-fia", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = "diag_rank_split"

    start_all = time.perf_counter()
    targets, source_rows = collect_rank_split_targets(
        config=config,
        batch_indices=parse_batch_indices(args.batch_indices, int(args.n_batches)),
        device=device,
        cos_threshold=float(args.cos_threshold),
        max_images=int(args.max_images),
        force_cache=bool(args.force_cache),
    )
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    torch.save(targets, method_dir / "targets.pt")
    write_source_recovery_csv(output_dir / "source_recovery.csv", source_rows)

    rec_path = method_dir / "rec.pt"
    log_path = method_dir / "fia_log.json"
    if rec_path.is_file() and not bool(args.force_fia):
        rec = torch.load(rec_path, map_location="cpu")
        metrics = evaluate_images(x_true=targets["x"], rec=rec, device=device)
        fia_log: list[dict[str, Any]] = []
        fia_wall_time = 0.0
    else:
        fia = run_fia_chunks(
            target_z=targets["z"],
            x_true=targets["x"],
            device=device,
            chunk_size=int(args.fia_chunk_size),
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
        rec = fia.rec
        metrics = fia.metrics
        fia_log = fia.log
        fia_wall_time = float(fia.wall_time_sec)
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
    summary = {
        "n_images": len(metrics_rows),
        "psnr_mean": float(np.mean(metrics.psnr)),
        "ssim_mean": float(np.mean(metrics.ssim)),
        "lpips_mean": float(np.mean(metrics.lpips)),
        "spearman_loss_lpips": rho,
        "spearman_loss_lpips_pvalue": pvalue,
        "fia_wall_time_sec": fia_wall_time,
        "target_source_cosine_mean": float(np.mean([row["target_source_cosine"] for row in metrics_rows])),
    }
    row = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": "m3_diag_rank_split_fia",
        "git_sha": git_sha(),
        "config": {
            **deepcopy(config),
            "run_tag": str(args.run_tag),
            "m3_diag_rank_split_fia": {
                "n_batches": int(args.n_batches),
                "batch_indices": parse_batch_indices(args.batch_indices, int(args.n_batches)),
                "cos_threshold": float(args.cos_threshold),
                "max_images": int(args.max_images),
                "n_iter_fi": int(args.n_iter_fi),
                "n_log_fi": int(args.n_log_fi),
                "lr_fi": float(args.lr_fi),
                "tv": float(args.tv),
                "fi": float(args.fi),
                "opt_fi": str(args.opt_fi),
                "sch_fi": str(args.sch_fi),
                "seed_fi": int(args.seed_fi),
                "fia_chunk_size": int(args.fia_chunk_size),
                "output_dir": str(output_dir),
            },
        },
        "metrics": {method: summary},
        "rank_split_source_summary": summarize_source_rows(source_rows),
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(config["paths"]["results_jsonl"]), row)
    (output_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        "{method}: n={n} lpips={lpips:.4f} psnr={psnr:.3f} ssim={ssim:.3f} "
        "spearman={rho:.3f} wall={wall:.1f}s".format(
            method=method,
            n=summary["n_images"],
            lpips=summary["lpips_mean"],
            psnr=summary["psnr_mean"],
            ssim=summary["ssim_mean"],
            rho=summary["spearman_loss_lpips"],
            wall=fia_wall_time,
        ),
        flush=True,
    )


def collect_rank_split_targets(
    *,
    config: dict[str, Any],
    batch_indices: list[int],
    device: torch.device,
    cos_threshold: float,
    max_images: int,
    force_cache: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage1_cfg = config["stage1"]
    stage2_cfg = config.get("stage2", {})
    z_list: list[torch.Tensor] = []
    x_list: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for batch_index in batch_indices:
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=force_cache)
        qobs = load_or_quantize_observation(victim, config, force=force_cache)
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
        source_cos = direct_source_cosines(victim.S_true, result.S_star)
        recovered = 0
        for source_index, cosine in enumerate(source_cos):
            keep_for_fia = bool(float(cosine) > float(cos_threshold))
            source_rows.append(
                {
                    "batch_index": int(batch_index),
                    "source_index": int(source_index),
                    "global_index": int(batch_index) * int(victim.S_true.shape[0]) + int(source_index),
                    "label": int(victim.labels[int(source_index)]),
                    "per_sample_loss": float(victim.per_sample_loss[int(source_index)]),
                    "source_cosine_initialization": float(init_cos[int(source_index)]),
                    "source_cosine_final": float(cosine),
                    "mp_detected": int(source_index) in set(detected),
                    "mp_subthreshold": int(source_index) in set(subthreshold),
                    "selected_for_fia": keep_for_fia,
                }
            )
            if keep_for_fia and (max_images <= 0 or len(metadata) < max_images):
                z_list.append(result.S_star[source_index : source_index + 1].detach().cpu().float().clamp_min(0.0))
                x_list.append(victim.x[source_index : source_index + 1].detach().cpu().float())
                metadata.append(
                    sample_metadata(
                        victim,
                        batch_index,
                        int(source_index),
                        "diag_rank_split",
                        float(cosine),
                    )
                )
                recovered += 1
        print(
            "batch={batch} diag_rank_split selected={selected}/8 cos={cos}".format(
                batch=batch_index,
                selected=recovered,
                cos="[" + ", ".join(f"{value:.4f}" for value in source_cos) + "]",
            ),
            flush=True,
        )
        if max_images > 0 and len(metadata) >= max_images:
            break

    if not z_list:
        raise RuntimeError("no recovered rank-split sources selected for FIA")
    return (
        {
            "z": torch.cat(z_list, dim=0).contiguous(),
            "x": torch.cat(x_list, dim=0).contiguous(),
            "metadata": metadata,
        },
        source_rows,
    )


def run_fia_chunks(
    *,
    target_z: torch.Tensor,
    x_true: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    n_iter: int,
    log_every: int,
    lr: float,
    tv: float,
    fi: float,
    optimizer: str,
    scheduler: str,
    seed: int,
    progress_label: str,
) -> FIAResult:
    n = int(target_z.shape[0])
    if int(chunk_size) <= 0 or int(chunk_size) >= n:
        return run_direct_fia(
            target_z=target_z,
            x_true=x_true,
            device=device,
            n_iter=n_iter,
            log_every=log_every,
            lr=lr,
            tv=tv,
            fi=fi,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=seed,
            progress_label=progress_label,
        )

    start = time.perf_counter()
    rec_chunks: list[torch.Tensor] = []
    psnr: list[float] = []
    ssim: list[float] = []
    lpips: list[float] = []
    log: list[dict[str, Any]] = []
    for chunk_idx, start_row in enumerate(range(0, n, int(chunk_size))):
        end_row = min(n, start_row + int(chunk_size))
        result = run_direct_fia(
            target_z=target_z[start_row:end_row],
            x_true=x_true[start_row:end_row],
            device=device,
            n_iter=n_iter,
            log_every=log_every,
            lr=lr,
            tv=tv,
            fi=fi,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=int(seed) + chunk_idx,
            progress_label=f"{progress_label}[{start_row}:{end_row}]",
        )
        rec_chunks.append(result.rec)
        psnr.extend(result.metrics.psnr)
        ssim.extend(result.metrics.ssim)
        lpips.extend(result.metrics.lpips)
        for entry in result.log:
            log.append({"chunk_start": start_row, "chunk_end": end_row, **entry})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return FIAResult(
        rec=torch.cat(rec_chunks, dim=0).contiguous(),
        metrics=FIAMetrics(psnr=psnr, ssim=ssim, lpips=lpips),
        log=log,
        wall_time_sec=float(time.perf_counter() - start),
    )


def write_source_recovery_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if bool(row["selected_for_fia"])]
    return {
        "n_sources": len(rows),
        "n_selected_for_fia": len(selected),
        "selected_global_indices": [int(row["global_index"]) for row in selected],
        "selected_source_cosines": [float(row["source_cosine_final"]) for row in selected],
        "subthreshold_selected": int(sum(bool(row["mp_subthreshold"]) for row in selected)),
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
