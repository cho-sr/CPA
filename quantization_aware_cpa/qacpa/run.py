from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .diagnostics import compute_diagnostics
from .gates import (
    factorization_gate,
    oracle_hinge_gate,
    run_static_milestone0_gates,
)
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .victim import load_or_collect_victim_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QACPA Milestone 0-1 pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)

    start_all = time.perf_counter()
    static_gates = run_static_milestone0_gates(seed=seed)
    n_batches = int(config["data"].get("n_batches", 1))
    print(
        "Milestone 0 static gates passed: "
        + ", ".join(check.name for check in static_gates),
        flush=True,
    )

    for batch_index in range(n_batches):
        batch_start = time.perf_counter()
        victim = load_or_collect_victim_batch(
            config,
            batch_index=batch_index,
            force=bool(args.force_cache),
        )
        qobs = load_or_quantize_observation(victim, config, force=bool(args.force_cache))
        batch_gates = [
            factorization_gate(victim.Y, victim.A_true, victim.S_true),
            oracle_hinge_gate(victim.A_true, victim.S_true, qobs.Y_hat, qobs.delta),
        ]

        stage1_cfg = config["stage1"]
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
            c_mp=float(stage1_cfg.get("c_mp", 1.5)),
            mp_truncation=bool(stage1_cfg.get("mp_truncation", True)),
            fastica_algorithm=str(stage1_cfg.get("fastica_algorithm", "parallel")),
        )

        diag_cfg = config["diagnostics"]
        diagnostics = compute_diagnostics(
            A_hat=stage1.A_hat,
            S_hat=stage1.S_hat,
            A_true=victim.A_true,
            S_true=victim.S_true,
            Y_hat=qobs.Y_hat,
            delta=qobs.delta,
            seed=seed + 10_000 * batch_index,
            gamma_columns=int(diag_cfg.get("gamma_columns", 512)),
            lp_method=str(diag_cfg.get("lp_method", "highs")),
            lp_feasibility_tol=float(diag_cfg.get("lp_feasibility_tol", 1.0e-7)),
        )

        row = make_result_row(
            config=config,
            batch_index=batch_index,
            victim=victim,
            static_gates=static_gates,
            batch_gates=batch_gates,
            stage1=stage1,
            diagnostics=diagnostics,
            wall_time_sec=time.perf_counter() - batch_start,
        )
        append_jsonl(Path(config["paths"]["results_jsonl"]), row)
        print(
            "batch={batch} rho={rho:.6f} gamma={gamma:.6g} "
            "feasible={feasible:.3f} B_eff={beff}/{B} recovered={recovered}/{beff} ({overB}/{B}) "
            "cond_A_hat={cond_hat:.6g} cond_A_true={cond_true:.6g} cos={cosines}".format(
                batch=batch_index,
                rho=diagnostics.rho,
                gamma=diagnostics.gamma,
                feasible=diagnostics.feasible_column_fraction,
                beff=diagnostics.B_eff,
                B=diagnostics.B_requested,
                recovered=diagnostics.n_sources_recovered,
                overB=diagnostics.n_sources_recovered,
                cond_hat=diagnostics.cond_A_hat,
                cond_true=diagnostics.cond_A_true,
                cosines=format_float_list(diagnostics.source_cos_per_source),
            ),
            flush=True,
        )

    print(
        f"Milestone 1 complete for {n_batches} batches in {time.perf_counter() - start_all:.1f}s",
        flush=True,
    )


def make_result_row(
    *,
    config: dict[str, Any],
    batch_index: int,
    victim: Any,
    static_gates: list[Any],
    batch_gates: list[Any],
    stage1: Any,
    diagnostics: Any,
    wall_time_sec: float,
) -> dict[str, Any]:
    diag = asdict(diagnostics)
    stage1_dict = asdict(stage1)
    stage1_dict.pop("A_hat", None)
    stage1_dict.pop("S_hat", None)
    stage1_dict.pop("Y_recon", None)
    labels = [int(label) for label in victim.labels.tolist()]
    sample_losses = [float(value) for value in victim.per_sample_loss.detach().cpu().tolist()]
    source_records = [
        {
            "source_index": int(idx),
            "label": labels[idx],
            "loss": sample_losses[idx],
            "a_true_norm": diagnostics.a_true_norm_per_source[idx],
            "matched": idx in set(diagnostics.matched_true_indices),
        }
        for idx in range(len(labels))
    ]
    matched_records = [source_records[idx] for idx in diagnostics.matched_true_indices]
    unmatched_records = [source_records[idx] for idx in diagnostics.unmatched_true_indices]

    return {
        "schema_version": 1,
        "run_tag": str(config.get("run_tag", "v3")),
        "milestone": 1,
        "batch_index": int(batch_index),
        "seed": int(config["seed"]),
        "git_sha": git_sha(),
        "labels": labels,
        "per_sample_loss": sample_losses,
        "image_paths": list(victim.image_paths),
        "config": config,
        "milestone0": {
            "static_gates": [asdict(check) for check in static_gates],
            "batch_gates": [asdict(check) for check in batch_gates],
        },
        "metrics": {
            "consistency_rate": diagnostics.consistency_rate,
            "residual_violation": diagnostics.residual_violation,
            "feasible_column_fraction": diagnostics.feasible_column_fraction,
            "infeasible_column_fraction": diagnostics.infeasible_column_fraction,
            "source_cos_per_source": diagnostics.source_cos_per_source,
            "source_cos": diagnostics.alignment.mean_abs_cosine,
            "n_sources_recovered": diagnostics.n_sources_recovered,
            "B_requested": diagnostics.B_requested,
            "B_eff": diagnostics.B_eff,
            "recovered_over_beff": diagnostics.recovered_over_beff,
            "recovered_over_B": diagnostics.recovered_over_B,
            "scale_err": None,
            "psnr": None,
            "ssim": None,
            "lpips": None,
        },
        "diagnostics": {
            "rho": diagnostics.rho,
            "gamma": diagnostics.gamma,
            "mean_axis_extent": diagnostics.mean_axis_extent,
            "ica_rmse": diagnostics.ica_rmse,
            "cond_A_hat": diagnostics.cond_A_hat,
            "cond_A_true": diagnostics.cond_A_true,
            "a_hat_norm_per_source": diagnostics.a_hat_norm_per_source,
            "a_true_norm_per_source": diagnostics.a_true_norm_per_source,
            "a_true_norm_matched": diagnostics.a_true_norm_matched,
            "a_true_norm_unmatched": diagnostics.a_true_norm_unmatched,
            "sample_loss_per_source": sample_losses,
            "label_per_source": labels,
            "matched_true_indices": diagnostics.matched_true_indices,
            "unmatched_true_indices": diagnostics.unmatched_true_indices,
            "matched_source_records": matched_records,
            "unmatched_source_records": unmatched_records,
            "source_records": source_records,
            "gamma_columns": diagnostics.gamma_columns,
            "gamma_infeasible_lps": diagnostics.gamma_infeasible_lps,
            "gamma_failed_lps": diagnostics.gamma_failed_lps,
            "alignment": diag["alignment"],
        },
        "stage1": stage1_dict,
        "wall_time_sec": float(wall_time_sec),
    }


def format_float_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        fallback = git_sha_from_head(Path(__file__).resolve())
        return fallback if fallback else "unavailable"
    return completed.stdout.strip()


def git_sha_from_head(start: Path) -> str | None:
    for parent in [start.parent, *start.parents]:
        git_path = parent / ".git"
        if git_path.is_dir():
            return read_git_head(git_path)
        if git_path.is_file():
            content = git_path.read_text(encoding="utf-8", errors="replace").strip()
            prefix = "gitdir:"
            if content.startswith(prefix):
                target = Path(content[len(prefix):].strip())
                if not target.is_absolute():
                    target = (parent / target).resolve()
                return read_git_head(target)
    return None


def read_git_head(git_dir: Path) -> str | None:
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8", errors="replace").strip()
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8", errors="replace").strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha.strip()
        return None
    return head or None


if __name__ == "__main__":
    main()
