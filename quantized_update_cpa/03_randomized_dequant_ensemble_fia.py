#!/usr/bin/env python3
"""Complete randomized dequantization ensemble: align, average, run FIA."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from common import REPO_ROOT, add_attack_args, build_experiment_model, get_device, output_pickle_path

CPA_SRC = REPO_ROOT / "cocktail_party_attack" / "src"
if str(CPA_SRC) not in sys.path:
    sys.path.insert(0, str(CPA_SRC))

from eval_utils import EmbEval, ImageEval  # noqa: E402
from feature_inversion import Direct, GradientMatching  # noqa: E402
from utils import normalize  # noqa: E402


STAGE3_NAME = "03_randomized_dequant_ensemble_cpa"
OUT_NAME = "03_randomized_dequant_ensemble_cpa_nearest_ensemble_avg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="tiny_imagenet")
    parser.add_argument("--model", type=str, default="vgg16")
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--rounding", type=str, default="nearest")
    parser.add_argument("--ensemble_size", type=int, default=8)
    parser.add_argument("--global_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--stage3_exp_root",
        type=Path,
        default=Path("exp/tiny_imagenet/vgg16/attack/cp_direct/nodef"),
    )
    parser.add_argument(
        "--output_exp_root",
        type=Path,
        default=Path("exp/tiny_imagenet/vgg16/attack/cp_direct/nodef"),
    )
    parser.add_argument("--wait_for_members", action="store_true")
    parser.add_argument("--wait_poll_seconds", type=int, default=60)
    add_attack_args(parser)
    return parser.parse_args()


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def member_paths(args: argparse.Namespace) -> list[Path]:
    prefix = f"{STAGE3_NAME}_{args.rounding}"
    return [
        args.stage3_exp_root / f"{prefix}_k{idx:02d}_uia" / f"{args.n_samples}_rec.pkl"
        for idx in range(args.ensemble_size)
    ]


def wait_for(paths: list[Path], poll_seconds: int) -> None:
    while True:
        missing = [p for p in paths if not p.exists() or p.stat().st_size == 0]
        if not missing:
            return
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] waiting for {missing[0]}", flush=True)
        time.sleep(poll_seconds)


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.reshape(x.shape[0], -1).astype(np.float64, copy=False)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def align_to_reference(reference: np.ndarray, member: np.ndarray) -> tuple[np.ndarray, float]:
    sim = unit_rows(reference) @ unit_rows(member).T
    rows, cols = linear_sum_assignment(-np.abs(sim))
    aligned = np.empty_like(reference)
    scores = []
    for row, col in zip(rows, cols):
        sign = 1.0 if sim[row, col] >= 0 else -1.0
        aligned[row] = member[col] * sign
        scores.append(abs(float(sim[row, col])))
    return aligned, float(np.mean(scores))


def averaged_rec_emb(member_recs: list[dict[str, list[np.ndarray]]], batch: int) -> tuple[np.ndarray, dict[str, float]]:
    reference = np.asarray(member_recs[0]["rec_emb"][batch])
    aligned = [reference]
    scores = [1.0]
    for rec in member_recs[1:]:
        member_aligned, score = align_to_reference(reference, np.asarray(rec["rec_emb"][batch]))
        aligned.append(member_aligned)
        scores.append(score)
    avg = np.stack(aligned, axis=0).mean(axis=0).astype(reference.dtype, copy=False)
    return avg, {
        "alignment_mean_abs_cos": float(np.mean(scores)),
        "alignment_min_abs_cos": float(np.min(scores)),
        "alignment_max_abs_cos": float(np.max(scores)),
    }


def fi_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ds=args.ds,
        tv=args.tv,
        opt_fi="adam",
        sch_fi="cosine",
        lr_fi=args.lr_fi,
        n_iter_fi=args.n_iter_fi,
        gm=args.gm,
        fi=args.fi,
        use_labels=args.use_labels,
    )


def run_fia(args: argparse.Namespace, model: torch.nn.Module, grad_data: dict[str, Any], batch: int, z: np.ndarray):
    device = get_device()
    n = z.shape[0]
    z_t = torch.tensor(z, device=device)
    inp_t = torch.tensor(grad_data["x"][batch][:n], device=device)
    emb_t = torch.tensor(grad_data["z"][batch][:n], device=device)
    labels_t = torch.tensor(grad_data["y"][batch][:n], device=device)
    grads_t = [torch.tensor(g, device=device) for g in grad_data["grad"][batch]]
    local_args = fi_namespace(args)
    fi = (
        GradientMatching(z_t, model, local_args, grads_t, labels_t)
        if args.fi_method == "gm"
        else Direct(z_t, model, local_args, grads_t, labels_t)
    )
    evaluator = ImageEval(inp_t, fix_order_method=None, fix_sign_method=None, ds=args.ds)
    iter_rows = []
    summary = None
    rec = None
    for i in range(args.n_iter_fi):
        losses = fi.step()
        if i % args.n_log_fi == 0 or i == args.n_iter_fi - 1:
            rec = fi.get_rec()
            avg, batch_metrics, _ = evaluator(rec)
            iter_rows.append(
                {
                    "iter": i,
                    "loss": losses["loss"],
                    "loss_fi": losses["loss_fi"],
                    "loss_tv": losses["loss_tv"],
                    "loss_gm": losses["loss_gm"],
                    "psnr": avg["psnr"],
                    "ssim": avg["ssim"],
                    "lpips": avg["lpips"],
                    "time": 0.0,
                }
            )
            if i == args.n_iter_fi - 1:
                summary = pd.DataFrame(batch_metrics)
    if rec is None or summary is None:
        raise RuntimeError("FIA produced no reconstruction")
    return pd.DataFrame(iter_rows), summary, rec.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    paths = member_paths(args)
    if args.wait_for_members:
        wait_for(paths, args.wait_poll_seconds)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing member rec file: {missing[0]}")

    member_recs = [read_pickle(path) for path in paths]
    update_file = output_pickle_path(f"{STAGE3_NAME}_{args.rounding}", args.ds, args.model, args.n_samples)
    grad_data = read_pickle(update_file)
    out_dir = args.output_exp_root / f"{OUT_NAME}_uia"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = build_experiment_model(
        model_name=args.model,
        ds=args.ds,
        h_dim=args.h_dim,
        device=device,
        checkpoint_path=args.global_checkpoint,
    )
    model.eval()

    iter_rows, summary_rows, fi_iters, fi_summaries, align_rows = [], [], [], [], []
    rec_dict = {"inp": [], "emb": [], "rec_emb": [], "rec": []}
    log_path = out_dir / f"{args.n_samples}.log"
    with log_path.open("a") as log:
        print(f"members={len(member_recs)} update_file={update_file}", file=log, flush=True)
        for batch in range(args.n_rounds):
            z_avg, align = averaged_rec_emb(member_recs, batch)
            emb_t = torch.tensor(grad_data["z"][batch][: z_avg.shape[0]], device=device)
            z_t = torch.tensor(z_avg, device=device)
            emb_eval = EmbEval(emb_t, fix_order_method="cs", fix_sign_method=None)
            emb_avg, emb_batch, z_reordered = emb_eval(z_t)
            z_avg = z_reordered.detach().cpu().numpy()

            iter_rows.append(
                {
                    "batch": batch,
                    "iter": 0,
                    "loss": 0.0,
                    "loss_ne": 0.0,
                    "loss_decor": 0.0,
                    "loss_nv": 0.0,
                    "loss_l1": 0.0,
                    "cs": float(emb_avg["cs"]),
                    "time": 0.0,
                }
            )
            summary_rows.extend({"batch": batch, "cs": float(cs)} for cs in emb_batch["cs"])
            align_rows.append({"batch": batch, "cs": float(emb_avg["cs"]), **align})
            print(
                f"batch {batch}: ensemble cs={float(emb_avg['cs']):.6f} "
                f"align={align['alignment_mean_abs_cos']:.6f}",
                flush=True,
            )
            print(
                f"batch {batch}: ensemble cs={float(emb_avg['cs']):.6f} "
                f"align={align['alignment_mean_abs_cos']:.6f}",
                file=log,
                flush=True,
            )

            fi_iter, fi_summary, rec = run_fia(args, model, grad_data, batch, z_avg)
            fi_iter.insert(0, "batch", batch)
            fi_summary.insert(0, "batch", batch)
            fi_iters.append(fi_iter)
            fi_summaries.append(fi_summary)

            n = z_avg.shape[0]
            rec_dict["inp"].append(
                normalize(torch.tensor(grad_data["x"][batch][:n], device=device), method="ds", ds=args.ds)
                .detach()
                .cpu()
                .numpy()
            )
            rec_dict["emb"].append(grad_data["z"][batch][:n])
            rec_dict["rec_emb"].append(z_avg)
            rec_dict["rec"].append(rec)
            m = fi_summary[["psnr", "ssim", "lpips"]].mean()
            print(
                f"batch {batch}: ensemble FIA psnr={m['psnr']:.6f} "
                f"ssim={m['ssim']:.6f} lpips={m['lpips']:.6f}",
                flush=True,
            )
            print(
                f"batch {batch}: ensemble FIA psnr={m['psnr']:.6f} "
                f"ssim={m['ssim']:.6f} lpips={m['lpips']:.6f}",
                file=log,
                flush=True,
            )

    pd.DataFrame(iter_rows).to_pickle(out_dir / f"{args.n_samples}_iter.pkl")
    pd.DataFrame(summary_rows).to_pickle(out_dir / f"{args.n_samples}_summary.pkl")
    pd.concat(fi_iters, ignore_index=True).to_pickle(out_dir / f"{args.n_samples}_fi_iter.pkl")
    pd.concat(fi_summaries, ignore_index=True).to_pickle(out_dir / f"{args.n_samples}_fi_summary.pkl")
    pd.DataFrame(align_rows).to_csv(out_dir / f"{args.n_samples}_alignment_summary.csv", index=False)
    write_pickle(rec_dict, out_dir / f"{args.n_samples}_rec.pkl")
    print(f"Saved ensemble FIA results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
