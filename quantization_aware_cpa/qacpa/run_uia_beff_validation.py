from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import torch

from .alignment import match_sources_by_abs_cosine
from .config import apply_overrides, load_config
from .quantizer import assert_cell_membership, quantize
from .run import append_jsonl, git_sha
from .seed import seed_all, torch_generator
from .stage1 import estimate_mixing_and_sources
from .uia import (
    load_or_collect_uia_feature_batch,
    mp_prediction_from_quantized,
    resolve_uia_device,
    run_fc_only_fedavg_checkpoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate UIA predicted B_eff with ICA + Hungarian matching only.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-tag", default="m5_uia_beff_validation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/m5_uia_beff_validation"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--n-values", default="16,32,64")
    parser.add_argument("--bits", default="2,3,4,8")
    parser.add_argument("--T-values", default="1,10")
    parser.add_argument("--extra-granularities", default="per_tensor,per_bucket")
    parser.add_argument("--extra-granularity-n", default="16,32")
    parser.add_argument("--extra-granularity-bits", type=int, default=4)
    parser.add_argument("--rounding", default="stochastic")
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--c-mp", type=float, default=3.0)
    parser.add_argument("--whitening-tolerance", type=float, default=3.0e-3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = apply_overrides(load_config(args.config), args.overrides)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(int(base_config["seed"]))
    device = resolve_uia_device(base_config)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    conditions = build_conditions(args)
    summary_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    if not bool(args.no_resume):
        summary_rows = read_existing_csv(output_dir / "summary.csv")
        cosine_rows = read_existing_csv(output_dir / "source_cosines.csv")
        spectrum_rows = read_existing_csv(output_dir / "lam_spectrum.csv")
    completed = {condition_key(row) for row in summary_rows}
    start_all = time.perf_counter()

    for idx, condition in enumerate(conditions):
        if condition_key(condition) in completed:
            print(
                "[{i}/{ncond}] skip seed={seed} n={n} T={T} bits={bits} scale={scale}".format(
                    i=idx + 1,
                    ncond=len(conditions),
                    seed=condition["seed"],
                    n=condition["n"],
                    T=condition["T"],
                    bits=condition["bits"],
                    scale=condition["scale_granularity"],
                ),
                flush=True,
            )
            continue
        start = time.perf_counter()
        cfg = condition_config(base_config, condition, args)
        feature = load_or_collect_uia_feature_batch(
            cfg,
            n=int(condition["n"]),
            batch_index=int(condition["batch_index"]),
            force=bool(args.force_cache),
        )
        obs = run_fc_only_fedavg_checkpoints(
            feature,
            lr=float(args.lr),
            checkpoints=[int(condition["T"])],
            device=device,
        )[int(condition["T"])]
        quant_seed = condition_quant_seed(condition)
        qobs = quantize(
            obs.Y_factor,
            bits=int(condition["bits"]),
            rounding=str(condition["rounding"]),
            generator=torch_generator(quant_seed, device=obs.Y_factor.device),
            scale_granularity=str(condition["scale_granularity"]),
            bucket_size=int(args.bucket_size),
        )
        assert_cell_membership(obs.Y_factor, qobs.Y_hat, qobs.delta)
        predicted = mp_prediction_from_quantized(
            qobs.Y_hat,
            qobs.delta,
            requested_rank=int(condition["n"]),
            rounding=qobs.rounding,
            c_mp=float(args.c_mp),
            device=device,
        )
        stage1 = run_stage1(cfg, qobs.Y_hat, qobs.delta, condition, args, device)
        alignment = match_sources_by_abs_cosine(feature.Z, stage1.S_hat)
        source_cos = [float(value) for value in alignment.per_source_abs_cosine]
        measured = int(sum(value > 0.9 for value in source_cos))
        summary_row = {
            "run_tag": str(args.run_tag),
            "condition_index": idx,
            "seed": int(condition["seed"]),
            "batch_index": int(condition["batch_index"]),
            "n": int(condition["n"]),
            "T": int(condition["T"]),
            "bits": int(condition["bits"]),
            "rounding": str(condition["rounding"]),
            "scale_granularity": str(condition["scale_granularity"]),
            "bucket_size": int(args.bucket_size) if str(condition["scale_granularity"]) == "per_bucket" else "",
            "lr": float(args.lr),
            "c_mp": float(args.c_mp),
            "device": str(device),
            "device_name": str(device_name),
            "predicted_B_eff": int(predicted["B_eff_pred"]),
            "stage1_B_eff": int(stage1.B_eff),
            "measured_recovered_cos09": measured,
            "measured_over_predicted": float(measured / max(1, int(predicted["B_eff_pred"]))),
            "measured_over_n": float(measured / max(1, int(condition["n"]))),
            "source_cos_mean": float(sum(source_cos) / max(1, len(source_cos))),
            "source_cos_min": float(min(source_cos)) if source_cos else float("nan"),
            "source_cos_per_source": json.dumps(source_cos),
            "matched_true_indices": json.dumps(alignment.true_indices),
            "unmatched_true_indices": json.dumps(alignment.unmatched_true_indices),
            "lam_over_mp_edge": json.dumps(predicted["lam_over_mp_edge"]),
            "min_lam_over_mp_edge": float(predicted["min_lam_over_mp_edge"]),
            "kept_min_lam_over_mp_edge": float(predicted["kept_min_lam_over_mp_edge"]),
            "discarded_max_lam_over_mp_edge": float(predicted["discarded_max_lam_over_mp_edge"]),
            "sigma2": float(predicted["sigma2"]),
            "mp_edge": float(predicted["mp_edge"]),
            "stage1_ica_method": str(stage1.ica_method),
            "stage1_ica_n_iter": int(stage1.ica_n_iter),
            "stage1_cpa_ica_l1": float(stage1.cpa_ica_l1),
            "stage1_reconstruction_rel_l2": float(stage1.reconstruction_rel_l2),
            "wall_time_sec": float(time.perf_counter() - start),
        }
        summary_rows.append(summary_row)
        cosine_rows.extend(make_cosine_rows(args, condition, alignment, source_cos))
        spectrum_rows.extend(make_spectrum_rows(args, condition, predicted))
        write_outputs(output_dir, summary_rows, cosine_rows, spectrum_rows)
        print(
            "[{i}/{ncond}] seed={seed} n={n} T={T} bits={bits} scale={scale} "
            "pred={pred} measured={meas} min_cos={mincos:.4g} time={sec:.1f}s".format(
                i=idx + 1,
                ncond=len(conditions),
                seed=condition["seed"],
                n=condition["n"],
                T=condition["T"],
                bits=condition["bits"],
                scale=condition["scale_granularity"],
                pred=summary_row["predicted_B_eff"],
                meas=summary_row["measured_recovered_cos09"],
                mincos=summary_row["source_cos_min"],
                sec=summary_row["wall_time_sec"],
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    render_prediction_scatter(summary_rows, output_dir / "predicted_vs_measured_beff.png")
    result = {
        "schema_version": 1,
        "run_tag": str(args.run_tag),
        "milestone": "m5_uia_beff_validation",
        "git_sha": git_sha(),
        "config": {
            **deepcopy(base_config),
            "uia_beff_validation": {
                "n_conditions": len(conditions),
                "conditions": conditions,
                "lr": float(args.lr),
                "c_mp": float(args.c_mp),
                "output_dir": str(output_dir),
            },
        },
        "metrics": {"summary": summary_rows},
        "artifacts": {
            "summary_csv": str(output_dir / "summary.csv"),
            "source_cosines_csv": str(output_dir / "source_cosines.csv"),
            "lam_spectrum_csv": str(output_dir / "lam_spectrum.csv"),
            "predicted_vs_measured_png": str(output_dir / "predicted_vs_measured_beff.png"),
        },
        "wall_time_sec": float(time.perf_counter() - start_all),
    }
    append_jsonl(Path(base_config["paths"]["results_jsonl"]), result)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"UIA B_eff validation complete in {result['wall_time_sec']:.1f}s", flush=True)


def run_stage1(
    cfg: dict[str, Any],
    Y_hat: torch.Tensor,
    delta: torch.Tensor,
    condition: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Any:
    stage1_cfg = cfg["stage1"]
    return estimate_mixing_and_sources(
        Y_hat,
        delta,
        n_components=int(condition["n"]),
        rounding=str(condition["rounding"]),
        debias=bool(stage1_cfg.get("debias", True)),
        seed=int(condition["seed"]) + 10_000 * int(condition["T"]) + 1_000 * int(condition["bits"]),
        max_iter=int(stage1_cfg.get("fastica_max_iter", 1000)),
        tol=float(stage1_cfg.get("fastica_tol", 1.0e-4)),
        fun=str(stage1_cfg.get("fastica_fun", "logcosh")),
        c_mp=float(args.c_mp),
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
        whitening_tolerance=float(args.whitening_tolerance),
        device=device,
    )


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_ints(args.seeds)
    n_values = parse_ints(args.n_values)
    bits_values = parse_ints(args.bits)
    T_values = parse_ints(args.T_values)
    extra_grans = [item.strip() for item in str(args.extra_granularities).split(",") if item.strip()]
    extra_ns = set(parse_ints(args.extra_granularity_n))
    conditions: list[dict[str, Any]] = []
    for seed in seeds:
        for n in n_values:
            for T in T_values:
                for bits in bits_values:
                    conditions.append(make_condition(args, seed=seed, n=n, T=T, bits=bits, scale="per_row"))
                if int(n) in extra_ns:
                    for scale in extra_grans:
                        conditions.append(
                            make_condition(
                                args,
                                seed=seed,
                                n=n,
                                T=T,
                                bits=int(args.extra_granularity_bits),
                                scale=scale,
                            )
                        )
    return conditions


def make_condition(
    args: argparse.Namespace,
    *,
    seed: int,
    n: int,
    T: int,
    bits: int,
    scale: str,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "batch_index": int(args.batch_index),
        "n": int(n),
        "T": int(T),
        "bits": int(bits),
        "rounding": str(args.rounding),
        "scale_granularity": str(scale),
        "bucket_size": int(args.bucket_size) if str(scale) == "per_bucket" else "",
    }


def condition_config(base_config: dict[str, Any], condition: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg["run_tag"] = str(args.run_tag)
    cfg["seed"] = int(condition["seed"])
    cfg.setdefault("data", {})["batch_size"] = int(condition["n"])
    cfg["data"]["n_batches"] = 1
    cfg["data"]["microbatch_size"] = min(32, int(condition["n"]))
    cfg["data"]["require_distinct_labels"] = True
    cfg.setdefault("stage1", {})["n_components"] = int(condition["n"])
    cfg["stage1"]["c_mp"] = float(args.c_mp)
    cfg.setdefault("quantization", {})["bits"] = int(condition["bits"])
    cfg["quantization"]["rounding"] = str(condition["rounding"])
    cfg["quantization"]["scale_granularity"] = str(condition["scale_granularity"])
    cfg["quantization"]["bucket_size"] = int(args.bucket_size)
    return cfg


def make_cosine_rows(
    args: argparse.Namespace,
    condition: dict[str, Any],
    alignment: Any,
    source_cos: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos, value in enumerate(source_cos):
        rows.append(
            {
                "run_tag": str(args.run_tag),
                "seed": int(condition["seed"]),
                "batch_index": int(condition["batch_index"]),
                "n": int(condition["n"]),
                "T": int(condition["T"]),
                "bits": int(condition["bits"]),
                "scale_granularity": str(condition["scale_granularity"]),
                "match_position": int(pos),
                "true_index": int(alignment.true_indices[pos]),
                "estimated_index": int(alignment.estimated_indices[pos]),
                "source_cosine": float(value),
                "recovered_cos09": bool(value > 0.9),
            }
        )
    return rows


def make_spectrum_rows(
    args: argparse.Namespace,
    condition: dict[str, Any],
    predicted: dict[str, Any],
) -> list[dict[str, Any]]:
    ratios = list(predicted["lam_over_mp_edge"])
    lam_top = list(predicted["lam_top"])
    B_eff = int(predicted["B_eff_pred"])
    rows: list[dict[str, Any]] = []
    for idx, ratio in enumerate(ratios):
        rows.append(
            {
                "run_tag": str(args.run_tag),
                "seed": int(condition["seed"]),
                "batch_index": int(condition["batch_index"]),
                "n": int(condition["n"]),
                "T": int(condition["T"]),
                "bits": int(condition["bits"]),
                "scale_granularity": str(condition["scale_granularity"]),
                "component_rank": int(idx),
                "lam_deb": float(lam_top[idx]),
                "lam_over_mp_edge": float(ratio),
                "kept_by_c_mp": bool(idx < B_eff),
                "c_mp": float(args.c_mp),
            }
        )
    return rows


def render_prediction_scatter(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    colors = {"per_row": "#2563eb", "per_tensor": "#dc2626", "per_bucket": "#059669"}
    markers = {1: "o", 10: "s"}
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for row in rows:
        scale = str(row["scale_granularity"])
        T = int(row["T"])
        ax.scatter(
            int(row["predicted_B_eff"]),
            int(row["measured_recovered_cos09"]),
            color=colors.get(scale, "#6b7280"),
            marker=markers.get(T, "o"),
            s=64,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
    max_axis = max(max(int(row["n"]) for row in rows), max(int(row["predicted_B_eff"]) for row in rows)) + 2
    ax.plot([0, max_axis], [0, max_axis], color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted B_eff")
    ax.set_ylabel("measured recovered count (cos > 0.9)")
    ax.set_title("UIA B_eff Prediction Validation")
    ax.grid(True, alpha=0.25)
    legend_handles = []
    for scale, color in colors.items():
        legend_handles.append(plt.Line2D([], [], linestyle="", marker="o", color=color, label=scale))
    t_labels = {1: "T=1 (= FedSGD)", 10: "T=10 (= FedAvg)"}
    for T, marker in markers.items():
        legend_handles.append(
            plt.Line2D([], [], linestyle="", marker=marker, color="#374151", label=t_labels.get(T, f"T={T}"))
        )
    ax.legend(handles=legend_handles, loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def condition_quant_seed(condition: dict[str, Any]) -> int:
    scale_offset = {"per_row": 0, "per_tensor": 17, "per_bucket": 31}.get(str(condition["scale_granularity"]), 43)
    return (
        int(condition["seed"])
        + 1_000_003 * int(condition["batch_index"])
        + 9_173 * int(condition["n"])
        + 1_009 * int(condition["T"])
        + 101 * int(condition["bits"])
        + scale_offset
    )


def condition_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
    return (
        int(row["seed"]),
        int(row["batch_index"]),
        int(row["n"]),
        int(row["T"]),
        int(row["bits"]),
        str(row["scale_granularity"]),
    )


def write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    cosine_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "source_cosines.csv", cosine_rows)
    write_csv(output_dir / "lam_spectrum.csv", spectrum_rows)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def read_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
