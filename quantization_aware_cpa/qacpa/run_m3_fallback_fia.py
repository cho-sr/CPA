from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .alignment import align_estimate_to_truth
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .fia import evaluate_images
from .run import append_jsonl, git_sha
from .run_m3_diag_rank_split_fia import run_fia_chunks
from .run_m3_fia_4bit import (
    aligned_sources_to_truth_order,
    direct_source_cosines,
    estimate_stage1,
    sample_metadata,
    safe_spearman,
    write_method_outputs,
)
from .victim import load_or_collect_victim_batch, resolve_device


FALLBACK_METHODS = {"naive_full", "debias_full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIA only on the M3 fallback sources.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="v10_m3_fallback_fia")
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path("results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv"),
    )
    parser.add_argument("--fallback-methods", default="naive_full,debias_full")
    parser.add_argument("--n-iter-fi", type=int, default=25000)
    parser.add_argument("--n-log-fi", type=int, default=5000)
    parser.add_argument("--lr-fi", type=float, default=1.0e-1)
    parser.add_argument("--tv", type=float, default=1.0e-1)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--opt-fi", default="adam")
    parser.add_argument("--sch-fi", default="cosine")
    parser.add_argument("--seed-fi", type=int, default=42)
    parser.add_argument("--fia-chunk-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v10_m3_fallback_fia"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-fia", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_indices = fallback_global_indices(args.aggregated_csv)
    methods = parse_fallback_methods(args.fallback_methods)
    start_all = time.perf_counter()
    all_source_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        targets, source_rows = collect_fallback_targets(
            config=config,
            global_indices=fallback_indices,
            fallback_method=method,
            force_cache=bool(args.force_cache),
        )
        all_source_rows.extend(source_rows)
        torch.save(targets, method_dir / "targets.pt")

        rec_path = method_dir / "rec.pt"
        log_path = method_dir / "fia_log.json"
        if rec_path.is_file() and not bool(args.force_fia):
            rec = torch.load(rec_path, map_location="cpu")
            metrics = evaluate_images(x_true=targets["x"], rec=rec, device=device)
            fia_log: list[dict[str, Any]] = []
            fia_wall = 0.0
        else:
            result = run_fia_chunks(
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
            rec = result.rec
            metrics = result.metrics
            fia_log = result.log
            fia_wall = float(result.wall_time_sec)
            torch.save(rec, rec_path)
            log_path.write_text(json.dumps(fia_log, indent=2), encoding="utf-8")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows = write_method_outputs(
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
            [float(row["per_sample_loss"]) for row in rows],
            [float(row["lpips"]) for row in rows],
        )
        summaries[method] = {
            "n_images": len(rows),
            "psnr_mean": float(np.mean(metrics.psnr)),
            "ssim_mean": float(np.mean(metrics.ssim)),
            "lpips_mean": float(np.mean(metrics.lpips)),
            "spearman_loss_lpips": rho,
            "spearman_loss_lpips_pvalue": pvalue,
            "target_source_cosine_mean": float(np.mean([float(row["target_source_cosine"]) for row in rows])),
            "fia_wall_time_sec": fia_wall,
        }
        print(
            "{method}: n={n} lpips={lpips:.4f} source_cos={cos:.4f} wall={wall:.1f}s".format(
                method=method,
                n=summaries[method]["n_images"],
                lpips=summaries[method]["lpips_mean"],
                cos=summaries[method]["target_source_cosine_mean"],
                wall=fia_wall,
            ),
            flush=True,
        )

    write_csv(output_dir / "fallback_source_quality.csv", all_source_rows)
    row = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": "m3_fallback_fia",
        "git_sha": git_sha(),
        "config": {
            **deepcopy(config),
            "run_tag": str(args.run_tag),
            "m3_fallback_fia": {
                "fallback_global_indices": fallback_indices,
                "fallback_methods": methods,
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
        "metrics": summaries,
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(config["paths"]["results_jsonl"]), row)
    (output_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def collect_fallback_targets(
    *,
    config: dict[str, Any],
    global_indices: list[int],
    fallback_method: str,
    force_cache: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch_size = int(config["data"]["batch_size"])
    stage1_cfg = config["stage1"]
    z_list: list[torch.Tensor] = []
    x_list: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    by_batch: dict[int, list[int]] = {}
    for gid in global_indices:
        by_batch.setdefault(int(gid) // batch_size, []).append(int(gid) % batch_size)

    for batch_index in sorted(by_batch):
        victim = load_or_collect_victim_batch(config, batch_index=batch_index, force=force_cache)
        qobs = load_or_quantize_observation(victim, config, force=force_cache)
        stage1 = estimate_stage1(
            qobs.Y_hat,
            qobs.delta,
            stage1_cfg=stage1_cfg,
            rounding=qobs.rounding,
            debias=fallback_method == "debias_full",
            mp_truncation=False,
            seed=int(config["seed"]) + batch_index,
        )
        _, S_aligned, alignment = align_estimate_to_truth(stage1.A_hat, stage1.S_hat, victim.S_true)
        S_full, filled = aligned_sources_to_truth_order(
            S_aligned=S_aligned.abs(),
            true_indices=alignment.true_indices,
            n_sources=int(victim.S_true.shape[0]),
            n_features=int(victim.S_true.shape[1]),
        )
        if not all(filled):
            missing = [idx for idx, ok in enumerate(filled) if not ok]
            raise RuntimeError(f"{fallback_method} did not fill all fallback candidates: batch={batch_index}, {missing}")
        source_cosines = direct_source_cosines(victim.S_true, S_full)
        for source_index in by_batch[batch_index]:
            global_index = int(batch_index) * batch_size + int(source_index)
            z_list.append(S_full[source_index : source_index + 1].detach().cpu().float().clamp_min(0.0))
            x_list.append(victim.x[source_index : source_index + 1].detach().cpu().float())
            metadata.append(
                sample_metadata(
                    victim,
                    batch_index,
                    int(source_index),
                    fallback_method,
                    float(source_cosines[source_index]),
                )
            )
            source_rows.append(
                {
                    "fallback_method": fallback_method,
                    "batch_index": int(batch_index),
                    "source_index": int(source_index),
                    "global_index": global_index,
                    "label": int(victim.labels[int(source_index)]),
                    "per_sample_loss": float(victim.per_sample_loss[int(source_index)]),
                    "source_cosine": float(source_cosines[source_index]),
                    "B_eff": int(stage1.B_eff),
                    "mp_truncation": bool(stage1.mp_truncation),
                    "debias": fallback_method == "debias_full",
                    "lam_over_mp_edge_all": " ".join(f"{float(x):.9g}" for x in stage1.lam_over_mp_edge_all),
                }
            )
    return (
        {
            "z": torch.cat(z_list, dim=0).contiguous(),
            "x": torch.cat(x_list, dim=0).contiguous(),
            "metadata": metadata,
        },
        source_rows,
    )


def fallback_global_indices(path: Path) -> list[int]:
    rows = read_csv(path)
    values = sorted(
        {
            int(row["global_index"])
            for row in rows
            if row["method"] == "A" and row.get("A_source") == "naive_fallback"
        }
    )
    if not values:
        raise RuntimeError(f"no A naive_fallback rows found in {path}")
    return values


def parse_fallback_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in methods if item not in FALLBACK_METHODS]
    if unknown:
        raise ValueError(f"unknown fallback methods {unknown}; allowed={sorted(FALLBACK_METHODS)}")
    return methods


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
