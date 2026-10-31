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

from .config import apply_overrides, load_config
from .quantizer import assert_cell_membership, quantize
from .run import append_jsonl, git_sha
from .seed import seed_all, torch_generator
from .uia import (
    load_or_collect_uia_feature_batch,
    mp_prediction_from_quantized,
    numerical_rank_from_singular_values,
    resolve_uia_device,
    run_fc_only_fedavg_checkpoints,
    singular_values_from_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UIA/FedAvg spectrum diagnostic without ICA, Stage 2, or FIA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="m5_uia_spectrum")
    parser.add_argument("--output-dir", type=Path, default=Path("results/m5_uia_spectrum"))
    parser.add_argument("--n-values", default="16,32")
    parser.add_argument("--T-values", default="1,2,5,10")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--rounding", default="stochastic")
    parser.add_argument("--scale-granularity", default="per_row")
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-indices", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--c-mp", type=float, default=3.0)
    parser.add_argument("--rank-rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--rank-abs-tol", type=float, default=1.0e-8)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = apply_overrides(load_config(args.config), args.overrides)
    base_config = condition_base_config(base_config, args)
    base_seed = int(base_config["seed"])
    seed_all(base_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_uia_device(base_config)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"UIA spectrum device={device} name={device_name}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []

    n_values = parse_ints(args.n_values)
    T_values = parse_ints(args.T_values)
    seeds = parse_ints(args.seeds) if str(args.seeds).strip() else [base_seed]
    batch_indices = parse_ints(args.batch_indices) if str(args.batch_indices).strip() else [int(args.batch_index)]
    start_all = time.perf_counter()
    rank_failed = False

    try:
        for seed in seeds:
            cfg_seed = deepcopy(base_config)
            cfg_seed["seed"] = int(seed)
            for batch_index in batch_indices:
                for n in n_values:
                    feature = load_or_collect_uia_feature_batch(
                        cfg_seed,
                        n=n,
                        batch_index=int(batch_index),
                        force=bool(args.force_cache),
                    )
                    checkpoint_start = time.perf_counter()
                    observations = run_fc_only_fedavg_checkpoints(
                        feature,
                        lr=float(args.lr),
                        checkpoints=T_values,
                        device=device,
                    )
                    checkpoint_wall = time.perf_counter() - checkpoint_start
                    for T in T_values:
                        obs = observations[int(T)]
                        condition_start = time.perf_counter()
                        row = process_condition(
                            args=args,
                            base_config=cfg_seed,
                            feature=feature,
                            obs=obs,
                            device=device,
                            device_name=device_name,
                            n=int(n),
                            T=int(T),
                            seed=int(seed),
                            batch_index=int(batch_index),
                            condition_start=condition_start,
                            checkpoint_wall=checkpoint_wall,
                        )
                        rank_gate_pass = bool(row["summary"]["rank_gate_pass"])
                        summary_rows.append(row["summary"])
                        if int(T) == 1:
                            sanity_rows.append(row["sanity"])
                        source_rows.extend(row["source_rows"])
                        spectrum_rows.extend(row["spectrum_rows"])

                        write_outputs(output_dir, summary_rows, source_rows, spectrum_rows, sanity_rows)
                        print(
                            "seed={seed} batch={batch} n={n} T={T} rank={rank}/{n} B_eff={beff}/{n} "
                            "b_cv={bcv:.4g} gia_cv={gcv:.4g} sp={sp:.4g} "
                            "min_ratio={minr:.4g} time={sec:.1f}s".format(
                                seed=seed,
                                batch=batch_index,
                                n=n,
                                T=T,
                                rank=row["summary"]["rank_svd"],
                                beff=row["summary"]["B_eff_pred"],
                                bcv=row["summary"]["b_norm_cv"],
                                gcv=row["summary"]["gia_a_norm_cv"],
                                sp=row["summary"]["spearman_final_loss_b_norm"],
                                minr=row["summary"]["min_lam_over_mp_edge"],
                                sec=row["summary"]["wall_time_sec"],
                            ),
                            flush=True,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if not rank_gate_pass:
                            rank_failed = True
                            raise GateFailure(
                                f"UIA true rank exceeded n: rank={row['summary']['rank_svd']} n={n} T={T}"
                            )
    finally:
        write_outputs(output_dir, summary_rows, source_rows, spectrum_rows, sanity_rows)

    result = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": "m5_uia_spectrum",
        "git_sha": git_sha(),
        "config": {
            **deepcopy(base_config),
            "uia_spectrum": {
                "n_values": n_values,
                "T_values": T_values,
                "seeds": seeds,
                "batch_indices": batch_indices,
                "bits": int(args.bits),
                "rounding": str(args.rounding),
                "scale_granularity": str(args.scale_granularity),
                "bucket_size": int(args.bucket_size),
                "lr": float(args.lr),
                "c_mp": float(args.c_mp),
                "rank_failed": bool(rank_failed),
                "output_dir": str(output_dir),
            },
        },
        "metrics": {
            "n_conditions": len(summary_rows),
            "rank_failed": bool(rank_failed),
            "summary": summary_rows,
        },
        "artifacts": {
            "summary_csv": str(output_dir / "summary.csv"),
            "source_norms_csv": str(output_dir / "source_norms.csv"),
            "lam_spectrum_csv": str(output_dir / "lam_spectrum.csv"),
            "t1_sanity_csv": str(output_dir / "t1_sanity.csv"),
        },
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(base_config["paths"]["results_jsonl"]), result)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"UIA spectrum complete in {result['wall_time_sec']:.1f}s", flush=True)


def process_condition(
    *,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    feature: Any,
    obs: Any,
    device: torch.device,
    device_name: str,
    n: int,
    T: int,
    seed: int,
    batch_index: int,
    condition_start: float,
    checkpoint_wall: float,
) -> dict[str, Any]:
    rank_info = compute_rank_info_from_factors(
        obs.B_cumulative,
        feature.Z,
        device=device,
        rel_tol=float(args.rank_rel_tol),
        abs_tol=float(args.rank_abs_tol),
    )
    centered_rank_info = compute_rank_info_from_factors(
        obs.B_cumulative,
        feature.Z - feature.Z.mean(dim=1, keepdim=True),
        device=device,
        rel_tol=float(args.rank_rel_tol),
        abs_tol=float(args.rank_abs_tol),
    )
    t1_sanity = (
        t1_sanity_metrics(feature, obs, args=args, n=n, batch_index=batch_index, device=device)
        if int(T) == 1
        else {}
    )
    rank_gate_pass = int(rank_info["rank"]) <= int(n)

    mp = {}
    if rank_gate_pass:
        quant_seed = quantization_seed(seed=seed, batch_index=int(batch_index), n=n, T=T)
        qobs = quantize(
            obs.Y_factor,
            bits=int(args.bits),
            rounding=str(args.rounding),
            generator=torch_generator(quant_seed, device=obs.Y_factor.device),
            scale_granularity=str(args.scale_granularity),
            bucket_size=int(args.bucket_size),
        )
        assert_cell_membership(obs.Y_factor, qobs.Y_hat, qobs.delta)
        mp = mp_prediction_from_quantized(
            qobs.Y_hat,
            qobs.delta,
            requested_rank=int(n),
            rounding=qobs.rounding,
            c_mp=float(args.c_mp),
            device=device,
        )
        q_delta_stats = delta_stats(qobs.delta)
    else:
        quant_seed = None
        q_delta_stats = {}

    b_norm = obs.B_cumulative.norm(dim=0)
    gia_norm = feature.gia_A.norm(dim=0)
    b_stats = vector_stats(b_norm)
    gia_stats = vector_stats(gia_norm)
    final_loss = obs.final_per_sample_loss
    spearman_loss_b = spearmanr(final_loss.numpy(), b_norm.numpy())
    spearman_initial_loss_b = spearmanr(obs.initial_per_sample_loss.numpy(), b_norm.numpy())
    wall_time_sec = time.perf_counter() - condition_start

    summary_row = {
        "run_tag": str(args.run_tag),
        "seed": seed,
        "batch_index": int(batch_index),
        "n": int(n),
        "T": int(T),
        "bits": int(args.bits),
        "rounding": str(args.rounding),
        "scale_granularity": str(args.scale_granularity),
        "bucket_size": int(args.bucket_size) if str(args.scale_granularity) == "per_bucket" else "",
        "lr": float(args.lr),
        "c_mp": float(args.c_mp),
        "device": str(device),
        "device_name": str(device_name),
        "rank_svd": int(rank_info["rank"]),
        "rank_tol": float(rank_info["tol"]),
        "rank_centered_svd": int(centered_rank_info["rank"]),
        "rank_delta_over_lr_svd": "",
        "rank_gate_pass": bool(rank_gate_pass),
        "B_eff_pred": int(mp.get("B_eff_pred", -1)),
        "sigma2": float(mp.get("sigma2", float("nan"))),
        "mp_edge": float(mp.get("mp_edge", float("nan"))),
        "mp_threshold": float(mp.get("mp_threshold", float("nan"))),
        "lam_over_mp_edge": json.dumps(mp.get("lam_over_mp_edge", [])),
        "min_lam_over_mp_edge": float(mp.get("min_lam_over_mp_edge", float("nan"))),
        "kept_min_lam_over_mp_edge": float(mp.get("kept_min_lam_over_mp_edge", float("nan"))),
        "discarded_max_lam_over_mp_edge": float(mp.get("discarded_max_lam_over_mp_edge", float("nan"))),
        "b_norm_mean": b_stats["mean"],
        "b_norm_std": b_stats["std"],
        "b_norm_cv": b_stats["cv"],
        "b_norm_min": b_stats["min"],
        "b_norm_max": b_stats["max"],
        "b_norm_min_over_max": b_stats["min_over_max"],
        "gia_a_norm_mean": gia_stats["mean"],
        "gia_a_norm_std": gia_stats["std"],
        "gia_a_norm_cv": gia_stats["cv"],
        "gia_a_norm_min": gia_stats["min"],
        "gia_a_norm_max": gia_stats["max"],
        "gia_a_norm_min_over_max": gia_stats["min_over_max"],
        "spearman_final_loss_b_norm": spearman_loss_b,
        "spearman_initial_loss_b_norm": spearman_initial_loss_b,
        "initial_loss_mean": float(obs.initial_per_sample_loss.mean().item()),
        "final_loss_mean": float(obs.final_per_sample_loss.mean().item()),
        "update_factor_rel_l2": float(obs.update_factor_rel_l2),
        "update_factor_max_abs": float(obs.update_factor_max_abs),
        "quantization_seed": "" if quant_seed is None else int(quant_seed),
        "delta_mean": float(q_delta_stats.get("mean", float("nan"))),
        "delta_min": float(q_delta_stats.get("min", float("nan"))),
        "delta_max": float(q_delta_stats.get("max", float("nan"))),
        "t1_gia_A_rel_l2": float(t1_sanity.get("gia_A_rel_l2", float("nan"))),
        "t1_gia_Y_rel_l2": float(t1_sanity.get("gia_Y_rel_l2", float("nan"))),
        "t1_gia_Y_max_abs": float(t1_sanity.get("gia_Y_max_abs", float("nan"))),
        "t1_quantized_gia_Y_rel_l2": float(t1_sanity.get("quantized_gia_Y_rel_l2", float("nan"))),
        "t1_quantized_gia_delta_rel_l2": float(t1_sanity.get("quantized_gia_delta_rel_l2", float("nan"))),
        "wall_time_sec": float(wall_time_sec),
        "checkpoint_wall_time_sec": float(checkpoint_wall),
    }
    if int(T) == 1:
        sanity_row = {key: summary_row[key] for key in summary_row if key.startswith("t1_")} | {
            "run_tag": str(args.run_tag),
            "seed": seed,
            "batch_index": int(batch_index),
            "n": int(n),
            "T": int(T),
        }
    else:
        sanity_row = {}
    return {
        "summary": summary_row,
        "sanity": sanity_row,
        "source_rows": make_source_rows(args, feature, obs, b_norm, gia_norm, n=n, T=T, batch_index=batch_index),
        "spectrum_rows": make_spectrum_rows(
            args,
            mp,
            n=n,
            T=T,
            rank_gate_pass=rank_gate_pass,
            seed=seed,
            batch_index=batch_index,
        ),
    }


def condition_base_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["run_tag"] = str(args.run_tag)
    cfg.setdefault("quantization", {})["bits"] = int(args.bits)
    cfg["quantization"]["rounding"] = str(args.rounding)
    cfg["quantization"]["scale_granularity"] = str(args.scale_granularity)
    cfg["quantization"]["bucket_size"] = int(args.bucket_size)
    cfg.setdefault("stage1", {})["c_mp"] = float(args.c_mp)
    cfg.setdefault("data", {})["n_batches"] = 1
    cfg["data"]["require_distinct_labels"] = True
    return cfg


def compute_rank_info_from_factors(
    B: torch.Tensor,
    Z: torch.Tensor,
    *,
    device: torch.device,
    rel_tol: float,
    abs_tol: float,
) -> dict[str, Any]:
    singular_values = singular_values_from_factors(B, Z, device=device)
    rank, tol = numerical_rank_from_singular_values(singular_values, rel_tol=rel_tol, abs_tol=abs_tol)
    return {
        "rank": int(rank),
        "tol": float(tol),
        "singular_values_top16": [float(value) for value in singular_values[:16].tolist()],
    }


def t1_sanity_metrics(
    feature: Any,
    obs: Any,
    *,
    args: argparse.Namespace,
    n: int,
    batch_index: int,
    device: torch.device,
) -> dict[str, float]:
    gia_A_rel = rel_l2(obs.B_cumulative, feature.gia_A)
    gia_Y_rel = rel_l2(obs.Y_factor, feature.gia_Y)
    gia_Y_max = float((obs.Y_factor - feature.gia_Y).abs().max().item())
    quant_seed = quantization_seed(seed=int(feature.seed), batch_index=int(batch_index), n=n, T=1)
    q_uia = quantize(
        obs.Y_factor,
        bits=int(args.bits),
        rounding=str(args.rounding),
        generator=torch_generator(quant_seed, device=obs.Y_factor.device),
        scale_granularity=str(args.scale_granularity),
        bucket_size=int(args.bucket_size),
    )
    q_gia = quantize(
        feature.gia_Y,
        bits=int(args.bits),
        rounding=str(args.rounding),
        generator=torch_generator(quant_seed, device=feature.gia_Y.device),
        scale_granularity=str(args.scale_granularity),
        bucket_size=int(args.bucket_size),
    )
    return {
        "gia_A_rel_l2": gia_A_rel,
        "gia_Y_rel_l2": gia_Y_rel,
        "gia_Y_max_abs": gia_Y_max,
        "quantized_gia_Y_rel_l2": rel_l2(q_uia.Y_hat, q_gia.Y_hat),
        "quantized_gia_delta_rel_l2": rel_l2(q_uia.delta, q_gia.delta),
    }


def make_source_rows(
    args: argparse.Namespace,
    feature: Any,
    obs: Any,
    b_norm: torch.Tensor,
    gia_norm: torch.Tensor,
    *,
    n: int,
    T: int,
    batch_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = [int(value) for value in feature.labels.tolist()]
    for idx in range(int(n)):
        rows.append(
            {
                "run_tag": str(args.run_tag),
                "seed": int(feature.seed),
                "batch_index": int(batch_index),
                "n": int(n),
                "T": int(T),
                "source_index": int(idx),
                "label": labels[idx],
                "image_path": feature.image_paths[idx] if idx < len(feature.image_paths) else "",
                "b_norm": float(b_norm[idx].item()),
                "gia_a_norm": float(gia_norm[idx].item()),
                "b_norm_over_gia_a_norm": float(b_norm[idx].item() / max(float(gia_norm[idx].item()), 1.0e-30)),
                "initial_loss": float(obs.initial_per_sample_loss[idx].item()),
                "final_loss": float(obs.final_per_sample_loss[idx].item()),
            }
        )
    return rows


def make_spectrum_rows(
    args: argparse.Namespace,
    mp: dict[str, Any],
    *,
    n: int,
    T: int,
    rank_gate_pass: bool,
    seed: int,
    batch_index: int,
) -> list[dict[str, Any]]:
    ratios = list(mp.get("lam_over_mp_edge", []))
    lam_top = list(mp.get("lam_top", []))
    B_eff = int(mp.get("B_eff_pred", -1))
    rows: list[dict[str, Any]] = []
    for idx in range(int(n)):
        rows.append(
            {
                "run_tag": str(args.run_tag),
                "seed": int(seed),
                "batch_index": int(batch_index),
                "n": int(n),
                "T": int(T),
                "component_rank": int(idx),
                "rank_gate_pass": bool(rank_gate_pass),
                "lam_deb": float(lam_top[idx]) if idx < len(lam_top) else float("nan"),
                "lam_over_mp_edge": float(ratios[idx]) if idx < len(ratios) else float("nan"),
                "kept_by_c_mp": bool(idx < B_eff) if B_eff >= 0 else False,
                "c_mp": float(args.c_mp),
                "mp_edge": float(mp.get("mp_edge", float("nan"))),
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "source_norms.csv", source_rows)
    write_csv(output_dir / "lam_spectrum.csv", spectrum_rows)
    write_csv(output_dir / "t1_sanity.csv", sanity_rows)


def quantization_seed(*, seed: int, batch_index: int, n: int, T: int) -> int:
    if int(T) == 1:
        return int(seed) + 1_000_003 * int(batch_index)
    return int(seed) + 1_000_003 * int(batch_index) + 9_173 * int(n) + 1_009 * int(T)


def vector_stats(values: torch.Tensor) -> dict[str, float]:
    vals = values.detach().cpu().float()
    mean = float(vals.mean().item())
    std = float(vals.std(unbiased=False).item())
    min_value = float(vals.min().item())
    max_value = float(vals.max().item())
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean != 0.0 else float("nan"),
        "min": min_value,
        "max": max_value,
        "min_over_max": float(min_value / max(max_value, 1.0e-30)),
    }


def delta_stats(delta: torch.Tensor) -> dict[str, float]:
    vals = delta.detach().cpu().float()
    return {
        "mean": float(vals.mean().item()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
    }


def rel_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    diff = left.detach().cpu().float() - right.detach().cpu().float()
    denom = right.detach().cpu().float().norm().clamp_min(torch.tensor(1.0e-12))
    return float((diff.norm() / denom).item())


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


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
