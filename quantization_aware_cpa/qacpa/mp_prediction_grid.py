from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import torch

from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .run import append_jsonl, git_sha
from .seed import seed_all
from .victim import load_or_collect_victim_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MP-truncated rank over the reduced M4 grid without ICA/FIA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="v10_mp_prediction_grid")
    parser.add_argument("--output-dir", type=Path, default=Path("results/v10_mp_prediction_grid"))
    parser.add_argument("--bits", default="2,3,4,6,8")
    parser.add_argument("--batch-sizes", default="32,128,512")
    parser.add_argument("--extra-granularity-bits", default="3,4")
    parser.add_argument("--extra-granularity-B", default="32,128")
    parser.add_argument("--scale-granularities", default="per_row,per_tensor,per_bucket")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--c-mp", type=float, default=3.0)
    parser.add_argument("--label-modes", default="distinct,mixed")
    parser.add_argument("--include-B512-label-mixed", action="store_true", default=True)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = apply_overrides(load_config(args.config), args.overrides)
    seed_all(int(base_config["seed"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)
    rows: list[dict[str, Any]] = []
    victim_cache: dict[tuple[int, int, int, bool], Any] = {}
    start_all = time.perf_counter()
    for idx, condition in enumerate(conditions):
        start = time.perf_counter()
        cfg = condition_config(base_config, condition, c_mp=float(args.c_mp), bucket_size=int(args.bucket_size))
        victim_key = (
            int(condition["seed"]),
            int(condition["B"]),
            int(args.batch_index),
            bool(condition["require_distinct_labels"]),
        )
        if victim_key not in victim_cache:
            victim_cache[victim_key] = load_or_collect_victim_batch(
                cfg,
                batch_index=int(args.batch_index),
                force=bool(args.force_cache),
            )
        victim = victim_cache[victim_key]
        qobs = load_or_quantize_observation(victim, cfg, force=bool(args.force_cache))
        row = compute_prediction_row(
            config=cfg,
            condition=condition,
            batch_index=int(args.batch_index),
            labels=victim.labels,
            Y_hat=qobs.Y_hat,
            delta=qobs.delta,
            c_mp=float(args.c_mp),
            wall_time_sec=time.perf_counter() - start,
        )
        rows.append(row)
        print(
            "[{i}/{n}] B={B} bits={bits} scale={scale} distinct={distinct} "
            "B_eff={beff}/{B} min_ratio={minr:.3g} edge={edge:.3g}".format(
                i=idx + 1,
                n=len(conditions),
                B=row["B"],
                bits=row["bits"],
                scale=row["scale_granularity"],
                distinct=row["require_distinct_labels"],
                beff=row["B_eff"],
                minr=row["min_lam_over_mp_edge"],
                edge=row["mp_edge"],
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(output_dir / "mp_prediction_grid.csv", rows)
    summary_rows = summarize(rows)
    write_csv(output_dir / "mp_prediction_summary.csv", summary_rows)
    result = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": "mp_prediction_grid",
        "git_sha": git_sha(),
        "config": {
            **deepcopy(base_config),
            "run_tag": str(args.run_tag),
            "mp_prediction_grid": {
                "n_conditions": len(conditions),
                "c_mp": float(args.c_mp),
                "bucket_size": int(args.bucket_size),
                "output_dir": str(output_dir),
            },
        },
        "metrics": {
            "n_conditions": len(rows),
            "summary": summary_rows,
        },
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(base_config["paths"]["results_jsonl"]), result)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def compute_prediction_row(
    *,
    config: dict[str, Any],
    condition: dict[str, Any],
    batch_index: int,
    labels: torch.Tensor,
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    c_mp: float,
    wall_time_sec: float,
) -> dict[str, Any]:
    Y = Y_hat.detach()
    device = resolve_device(str(config.get("device", "auto")))
    work_device = device if device.type == "cuda" else torch.device("cpu")
    Y_work = Y.to(device=work_device, dtype=torch.float32)
    delta_work = delta.detach().to(device=work_device, dtype=torch.float32)
    N, d = Y_work.shape
    row_mean = Y_work.mean(dim=1, keepdim=True)
    Y_zc = Y_work - row_mean
    cov = Y_zc.matmul(Y_zc.T) / float(d)
    if str(condition["rounding"]) == "stochastic":
        cov = cov - torch.diag(delta_work.square() / 12.0)
    cov = (cov + cov.T) * 0.5
    eig_vals = torch.linalg.eigvalsh(cov)
    top_vals = eig_vals[-int(condition["B"]) :].flip(0).detach().cpu().float()
    sigma2 = float((delta_work.square() / 12.0).mean().detach().cpu())
    mp_edge = float(sigma2 * (1.0 + (float(N) / float(d)) ** 0.5) ** 2)
    threshold = float(c_mp) * mp_edge
    B_eff = int((top_vals > threshold).sum().item())
    ratios = [float(value / mp_edge) if mp_edge > 0.0 else float("inf") for value in top_vals]
    labels_list = [int(label) for label in labels.detach().cpu().view(-1).tolist()]
    labels_unique_count = len(set(labels_list))
    return {
        "run_tag": str(condition["run_tag"]),
        "seed": int(condition["seed"]),
        "batch_index": int(batch_index),
        "B": int(condition["B"]),
        "bits": int(condition["bits"]),
        "rounding": str(condition["rounding"]),
        "scale_granularity": str(condition["scale_granularity"]),
        "bucket_size": condition.get("bucket_size", ""),
        "require_distinct_labels": bool(condition["require_distinct_labels"]),
        "labels_unique_count": int(labels_unique_count),
        "labels_duplicate_count": int(len(labels_list) - labels_unique_count),
        "microbatch_size": int(config["data"].get("microbatch_size", condition["B"])),
        "c_mp": float(c_mp),
        "sigma2": sigma2,
        "mp_edge": mp_edge,
        "mp_threshold": threshold,
        "B_eff": B_eff,
        "B_eff_fraction": float(B_eff / max(1, int(condition["B"]))),
        "lam_over_mp_edge": json.dumps(ratios),
        "min_lam_over_mp_edge": float(min(ratios)),
        "kept_min_lam_over_mp_edge": float(min(ratios[:B_eff])) if B_eff > 0 else float("nan"),
        "discarded_max_lam_over_mp_edge": float(max(ratios[B_eff:])) if B_eff < len(ratios) else float("nan"),
        "boundary_margin": float(min(abs(value - float(c_mp)) for value in ratios)),
        "near_boundary_count_pm20pct": int(sum(abs(value - float(c_mp)) <= 0.2 * float(c_mp) for value in ratios)),
        "wall_time_sec": float(wall_time_sec),
    }


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    bits = parse_ints(args.bits)
    batch_sizes = parse_ints(args.batch_sizes)
    seeds = parse_ints(args.seeds)
    extra_bits = set(parse_ints(args.extra_granularity_bits))
    extra_B = set(parse_ints(args.extra_granularity_B))
    scales = [item.strip() for item in str(args.scale_granularities).split(",") if item.strip()]
    label_modes = {item.strip().lower() for item in str(args.label_modes).split(",") if item.strip()}
    conditions: list[dict[str, Any]] = []
    for seed in seeds:
        for B in batch_sizes:
            for bit in bits:
                if "distinct" in label_modes:
                    conditions.append(make_condition(seed, B, bit, "per_row", require_distinct_labels=True, args=args))
                if "distinct" in label_modes and B in extra_B and bit in extra_bits:
                    for scale in scales:
                        if scale == "per_row":
                            continue
                        conditions.append(make_condition(seed, B, bit, scale, require_distinct_labels=True, args=args))
            if int(B) == 512 and bool(args.include_B512_label_mixed) and "mixed" in label_modes:
                for bit in bits:
                    conditions.append(make_condition(seed, B, bit, "per_row", require_distinct_labels=False, args=args))
    return conditions


def make_condition(
    seed: int,
    B: int,
    bits: int,
    scale: str,
    *,
    require_distinct_labels: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    label = "distinct" if require_distinct_labels else "mixed"
    tag = f"{args.run_tag}_B{B}_bits{bits}_seed{seed}_scale{scale}_{label}"
    return {
        "run_tag": tag,
        "seed": int(seed),
        "B": int(B),
        "bits": int(bits),
        "rounding": "stochastic",
        "scale_granularity": scale,
        "bucket_size": int(args.bucket_size) if scale == "per_bucket" else "",
        "require_distinct_labels": bool(require_distinct_labels),
    }


def condition_config(base: dict[str, Any], condition: dict[str, Any], *, c_mp: float, bucket_size: int) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["run_tag"] = str(condition["run_tag"])
    cfg["seed"] = int(condition["seed"])
    cfg.setdefault("data", {})["batch_size"] = int(condition["B"])
    cfg["data"]["n_batches"] = 1
    cfg["data"]["microbatch_size"] = min(32, int(condition["B"]))
    cfg["data"]["require_distinct_labels"] = bool(condition["require_distinct_labels"])
    cfg.setdefault("stage1", {})["n_components"] = int(condition["B"])
    cfg["stage1"]["c_mp"] = float(c_mp)
    cfg.setdefault("quantization", {})["bits"] = int(condition["bits"])
    cfg["quantization"]["rounding"] = str(condition["rounding"])
    cfg["quantization"]["scale_granularity"] = str(condition["scale_granularity"])
    cfg["quantization"]["bucket_size"] = int(bucket_size)
    return cfg


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["B"], row["bits"], row["scale_granularity"], row["require_distinct_labels"])
        groups.setdefault(key, []).append(row)
    for (B, bits, scale, distinct), group in sorted(groups.items()):
        out.append(
            {
                "B": B,
                "bits": bits,
                "scale_granularity": scale,
                "require_distinct_labels": distinct,
                "n": len(group),
                "B_eff_mean": float(sum(int(row["B_eff"]) for row in group) / len(group)),
                "B_eff_min": min(int(row["B_eff"]) for row in group),
                "B_eff_max": max(int(row["B_eff"]) for row in group),
                "mp_edge_mean": float(sum(float(row["mp_edge"]) for row in group) / len(group)),
                "labels_unique_count_mean": float(sum(int(row["labels_unique_count"]) for row in group) / len(group)),
                "labels_duplicate_count_mean": float(
                    sum(int(row["labels_duplicate_count"]) for row in group) / len(group)
                ),
                "min_lam_over_mp_edge_mean": float(
                    sum(float(row["min_lam_over_mp_edge"]) for row in group) / len(group)
                ),
                "near_boundary_count_pm20pct_sum": int(
                    sum(int(row["near_boundary_count_pm20pct"]) for row in group)
                ),
            }
        )
    return out


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
