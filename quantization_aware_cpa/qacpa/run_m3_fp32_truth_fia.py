from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .config import apply_overrides, load_config
from .fia import evaluate_images, run_direct_fia
from .run import append_jsonl, git_sha
from .run_m3_fia_4bit import sample_metadata, safe_spearman, write_method_outputs
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-source fp32 FIA for selected M3 global indices.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="v7_m3_fp32_truth_missing")
    parser.add_argument("--global-indices", required=True, help="Comma-separated M3 global indices.")
    parser.add_argument("--n-iter-fi", type=int, default=25000)
    parser.add_argument("--n-log-fi", type=int, default=5000)
    parser.add_argument("--lr-fi", type=float, default=1.0e-1)
    parser.add_argument("--tv", type=float, default=1.0e-1)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--opt-fi", default="adam")
    parser.add_argument("--sch-fi", default="cosine")
    parser.add_argument("--seed-fi", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v7_m3_fp32_truth_missing"))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-fia", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(args.output_dir)
    method_dir = output_dir / "fp32"
    method_dir.mkdir(parents=True, exist_ok=True)
    global_indices = parse_global_indices(args.global_indices)
    start_all = time.perf_counter()

    targets = collect_fp32_truth_targets(
        config=config,
        global_indices=global_indices,
        force_cache=bool(args.force_cache),
    )
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
            progress_label="fp32_truth",
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
        method="fp32",
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
        "fia_wall_time_sec": wall_time,
        "target_source_cosine_mean": 1.0,
        "final_log": fia_log[-1] if fia_log else None,
    }
    row = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": 3,
        "git_sha": git_sha(),
        "config": {
            **deepcopy(config),
            "run_tag": str(args.run_tag),
            "m3_fp32_truth_fia": {
                "global_indices": global_indices,
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
        "metrics": {"fp32": summary},
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(config["paths"]["results_jsonl"]), row)
    (output_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        "fp32_truth: n={n} lpips={lpips:.4f} psnr={psnr:.3f} ssim={ssim:.3f} spearman={rho:.3f}".format(
            n=summary["n_images"],
            lpips=summary["lpips_mean"],
            psnr=summary["psnr_mean"],
            ssim=summary["ssim_mean"],
            rho=summary["spearman_loss_lpips"],
        ),
        flush=True,
    )


def collect_fp32_truth_targets(
    *,
    config: dict[str, Any],
    global_indices: list[int],
    force_cache: bool,
) -> dict[str, Any]:
    batch_size = int(config["data"]["batch_size"])
    z_list: list[torch.Tensor] = []
    x_list: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    cache: dict[int, Any] = {}
    for global_index in global_indices:
        batch_index = int(global_index) // batch_size
        source_index = int(global_index) % batch_size
        if batch_index not in cache:
            cache[batch_index] = load_or_collect_victim_batch(config, batch_index=batch_index, force=force_cache)
        victim = cache[batch_index]
        z_list.append(victim.S_true[source_index : source_index + 1].detach().cpu().float().clamp_min(0.0))
        x_list.append(victim.x[source_index : source_index + 1].detach().cpu().float())
        metadata.append(sample_metadata(victim, batch_index, source_index, "fp32", 1.0))
    if not z_list:
        raise ValueError("--global-indices must not be empty")
    return {
        "z": torch.cat(z_list, dim=0).contiguous(),
        "x": torch.cat(x_list, dim=0).contiguous(),
        "metadata": metadata,
    }


def parse_global_indices(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("--global-indices must contain at least one index")
    return values


if __name__ == "__main__":
    main()
