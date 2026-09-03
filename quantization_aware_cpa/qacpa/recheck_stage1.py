from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import align_estimate_to_truth
from .cache import load_or_quantize_observation
from .config import apply_overrides, load_config
from .seed import seed_all
from .stage1 import estimate_mixing_and_sources
from .victim import load_or_collect_victim_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recheck Stage-1 per-source cosine without LP diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed = int(config["seed"])
    seed_all(seed)
    stage1_cfg = config["stage1"]
    n_batches = int(config["data"].get("n_batches", 1))

    for batch_index in range(n_batches):
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
            c_mp=float(stage1_cfg.get("c_mp", 1.5)),
            mp_truncation=bool(stage1_cfg.get("mp_truncation", True)),
            fastica_algorithm=str(stage1_cfg.get("fastica_algorithm", "parallel")),
        )
        _A_aligned, _S_aligned, alignment = align_estimate_to_truth(
            stage1.A_hat,
            stage1.S_hat,
            victim.S_true,
        )
        cosines = alignment.per_source_abs_cosine
        recovered = sum(value > 0.9 for value in cosines)
        labels = [int(label) for label in victim.labels.tolist()]
        print(
            "batch={batch} B_eff={beff}/{B} recovered={recovered}/{beff} source_cos={cosines} "
            "lam_over_mp_edge={ratios} cond_A_hat={cond:.6g} whiten_offdiag={offdiag:.3g} "
            "fastica_n_iter={n_iter} converged={converged} labels={labels}".format(
                batch=batch_index,
                beff=stage1.B_eff,
                B=stage1.B_requested,
                recovered=recovered,
                cosines="[" + ", ".join(f"{value:.6f}" for value in cosines) + "]",
                ratios="[" + ", ".join(f"{value:.3f}" for value in stage1.lam_over_mp_edge_all) + "]",
                cond=stage1.cond_A_hat,
                offdiag=stage1.whitened_cov_offdiag_abs_max,
                n_iter=stage1.fastica_n_iter,
                converged=stage1.fastica_converged,
                labels=labels,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
