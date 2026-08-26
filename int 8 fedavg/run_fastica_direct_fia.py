#!/usr/bin/env python3
"""Run Direct FIA from saved int8 FastICA embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SRC = REPO_ROOT / "baseline" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from datasets import ds_mean, ds_std  # noqa: E402
from feature_inversion import Direct  # noqa: E402
from models import get_model  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int 8 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)
DEFAULT_FASTICA_DIR = DEFAULT_RUN_DIR / "fastica_dequantized_aggregate"
DEFAULT_INPUT_PICKLE = DEFAULT_RUN_DIR / "round_020_int8_dequant_aggregate_cpa.pickle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct feature inversion from saved int8 FastICA VGG16 embeddings."
    )
    parser.add_argument("--input_pickle", type=Path, default=DEFAULT_INPUT_PICKLE)
    parser.add_argument("--fastica_dir", type=Path, default=DEFAULT_FASTICA_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--embedding_source",
        type=str,
        default="oracle",
        choices=["oracle", "blind", "raw"],
        help=(
            "oracle uses oracle_reordered_embeddings.npy, blind uses blind_embeddings.npy, "
            "raw uses raw_sources from fastica_embeddings.pkl."
        ),
    )
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--n_sample", type=int, default=-1)
    parser.add_argument("--n_iter_fi", type=int, default=25000)
    parser.add_argument("--n_log_fi", type=int, default=500)
    parser.add_argument("--lr_fi", type=float, default=0.1)
    parser.add_argument("--tv", type=float, default=0.1)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--opt_fi", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--sch_fi", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument(
        "--log_image_metrics",
        action="store_true",
        help="Log PSNR/SSIM/LPIPS at each FIA log step. This is slower and can use more VRAM.",
    )
    parser.add_argument(
        "--image_metric_log_interval",
        type=int,
        default=None,
        help="Iteration interval for image metrics; defaults to --n_log_fi.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_embeddings(fastica_dir: Path, source: str) -> np.ndarray:
    if source == "oracle":
        path = fastica_dir / "oracle_reordered_embeddings.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run with --embedding_source blind instead."
            )
        return np.load(path)
    if source == "blind":
        return np.load(fastica_dir / "blind_embeddings.npy")

    data = read_pickle(fastica_dir / "fastica_embeddings.pkl")
    raw = data.get("raw_sources")
    if raw is None:
        raise KeyError("fastica_embeddings.pkl does not contain raw_sources")
    return np.asarray(raw)


def unnormalize_images(x: torch.Tensor, ds: str) -> torch.Tensor:
    mean = torch.tensor(ds_mean[ds], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ds_std[ds], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


class ImageMetricComputer:
    def __init__(self, device: torch.device, *, skip_lpips: bool) -> None:
        self.ssim_fn = None
        self.ssim_error: Optional[str] = None
        try:
            from torchmetrics.functional import structural_similarity_index_measure

            self.ssim_fn = structural_similarity_index_measure
        except Exception as exc:  # pragma: no cover - depends on local torchmetrics install
            self.ssim_error = str(exc)

        self.lpips_model = None
        self.lpips_error: Optional[str] = None
        if not skip_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

                self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
                self.lpips_model.eval()
            except Exception as exc:  # pragma: no cover - depends on local torchmetrics assets
                self.lpips_error = str(exc)

    def compute(
        self,
        originals: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if originals.shape != reconstructions.shape:
            raise ValueError(
                f"image metric shape mismatch: originals={tuple(originals.shape)}, "
                f"reconstructions={tuple(reconstructions.shape)}"
            )

        psnr_values: List[float] = []
        ssim_values: List[float] = []
        lpips_values: List[float] = []

        with torch.no_grad():
            for orig, rec in zip(originals, reconstructions):
                orig_b = orig.unsqueeze(0)
                rec_b = rec.unsqueeze(0)
                mse = torch.mean((orig_b - rec_b) ** 2).clamp_min(1e-12)
                psnr_values.append(float((10.0 * torch.log10(1.0 / mse)).detach().cpu().item()))
                if self.ssim_fn is not None:
                    ssim_values.append(float(self.ssim_fn(rec_b, orig_b).detach().cpu().item()))
                if self.lpips_model is not None:
                    lpips_in_rec = rec_b * 2.0 - 1.0
                    lpips_in_orig = orig_b * 2.0 - 1.0
                    lpips_values.append(
                        float(self.lpips_model(lpips_in_rec, lpips_in_orig).detach().cpu().item())
                    )

        batch = {
            "psnr": psnr_values,
            "ssim": ssim_values if ssim_values else None,
            "lpips": lpips_values if lpips_values else None,
        }
        summary = {
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_min": float(np.min(psnr_values)),
            "psnr_max": float(np.max(psnr_values)),
            "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
            "ssim_min": float(np.min(ssim_values)) if ssim_values else None,
            "ssim_max": float(np.max(ssim_values)) if ssim_values else None,
            "ssim_error": self.ssim_error,
            "lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
            "lpips_min": float(np.min(lpips_values)) if lpips_values else None,
            "lpips_max": float(np.max(lpips_values)) if lpips_values else None,
            "lpips_error": self.lpips_error,
        }
        return summary, batch


def to_uint8_chw(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (np.transpose(image, (1, 2, 0)) * 255.0).round().astype(np.uint8)


def make_pair_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    path: Path,
    *,
    max_items: int = 24,
    pair_cols: int = 4,
) -> None:
    n = min(max_items, int(originals.shape[0]), int(reconstructions.shape[0]))
    if n <= 0:
        return
    label_h = 18
    tile_w = int(originals.shape[-1])
    tile_h = int(originals.shape[-2])
    rows = math.ceil(n / pair_cols)
    canvas = Image.new("RGB", (pair_cols * tile_w * 2, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx in range(n):
        row, col = divmod(idx, pair_cols)
        x0 = col * tile_w * 2
        y0 = row * (tile_h + label_h)
        orig = Image.fromarray(to_uint8_chw(originals[idx]))
        rec = Image.fromarray(to_uint8_chw(reconstructions[idx]))
        canvas.paste(orig, (x0, y0 + label_h))
        canvas.paste(rec, (x0 + tile_w, y0 + label_h))
        draw.text((x0 + 3, y0 + 2), f"orig {idx}", fill=(0, 0, 0))
        draw.text((x0 + tile_w + 3, y0 + 2), "fia rec", fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, eps)


def embedding_order_for_truth(recovered: np.ndarray, true_z: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    sims = row_normalize(recovered.astype(np.float64)) @ row_normalize(true_z.astype(np.float64)).T
    rec_ind, true_ind = linear_sum_assignment(-np.abs(sims))
    order = rec_ind[np.argsort(true_ind)]
    matched = sims[order, np.arange(len(order))]
    return order, {
        "mean_abs_cosine": float(np.mean(np.abs(matched))),
        "mean_signed_cosine": float(np.mean(matched)),
        "min_abs_cosine": float(np.min(np.abs(matched))),
        "median_abs_cosine": float(np.median(np.abs(matched))),
        "max_abs_cosine": float(np.max(np.abs(matched))),
        "recovered_order_for_true_index": [int(x) for x in order.tolist()],
        "matched_signed_cosines": [float(x) for x in matched.tolist()],
    }


def compute_image_metrics(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    skip_lpips: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return ImageMetricComputer(reconstructions.device, skip_lpips=skip_lpips).compute(
        originals,
        reconstructions,
    )


def write_iter_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_int8_input(input_pickle: Path) -> Dict[str, Any]:
    manifest_path = input_pickle.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing int8 FedAvg manifest next to input pickle: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "signal_is_int8_dequantized": manifest.get("signal")
        == "int8_fedavg_server_dequantized_aggregate_update",
        "quant_bits_is_8": manifest.get("quant_bits") == 8,
        "server_input_is_dequantized": manifest.get("server_aggregation_input")
        == "dequantized_client_updates",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Input pickle is not an int8 dequantized FedAvg aggregate: {checks}")
    return {"manifest_path": str(manifest_path), "passed": True, "checks": checks}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_pickle = resolve_path(args.input_pickle)
    fastica_dir = resolve_path(args.fastica_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else fastica_dir / f"direct_fia_{args.embedding_source}"
    )
    if args.fresh_start and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    input_check = validate_int8_input(input_pickle)
    data = read_pickle(input_pickle)
    x_np = np.asarray(data["x"][args.batch_index])
    y_np = np.asarray(data["y"][args.batch_index])
    true_z_np = np.asarray(data["z"][args.batch_index])
    grads_np = data["grad"][args.batch_index]

    embeddings_np = load_embeddings(fastica_dir, args.embedding_source).astype(np.float32)
    n_available = min(len(embeddings_np), len(x_np), len(y_np), len(true_z_np))
    n_sample = n_available if args.n_sample == -1 else min(args.n_sample, n_available)
    if n_sample <= 0:
        raise ValueError("No samples available for FIA")

    embeddings_np = embeddings_np[:n_sample]
    x_np = x_np[:n_sample]
    y_np = y_np[:n_sample]
    true_z_np = true_z_np[:n_sample]

    if args.embedding_source == "oracle":
        eval_order = np.arange(n_sample)
        emb_match = {
            "source": "identity_oracle_reordered_embeddings",
            "recovered_order_for_true_index": [int(x) for x in eval_order.tolist()],
        }
    else:
        eval_order, emb_match = embedding_order_for_truth(embeddings_np, true_z_np)

    x = torch.as_tensor(x_np, device=device, dtype=torch.float32)
    y = torch.as_tensor(y_np, device=device, dtype=torch.long)
    recovered_z = torch.as_tensor(embeddings_np, device=device, dtype=torch.float32)
    grads = [torch.as_tensor(g, device=device, dtype=torch.float32) for g in grads_np]
    original_images = unnormalize_images(x, "imagenet").detach()
    eval_order_tensor = torch.as_tensor(eval_order, device=device, dtype=torch.long)

    image_metric_log_interval = (
        args.n_log_fi if args.image_metric_log_interval is None else args.image_metric_log_interval
    )
    if args.log_image_metrics and image_metric_log_interval <= 0:
        raise ValueError("--image_metric_log_interval must be > 0 when --log_image_metrics is set")
    metric_computer = (
        ImageMetricComputer(device, skip_lpips=args.skip_lpips) if args.log_image_metrics else None
    )

    model = get_model(model_name="vgg16", ds="imagenet", h_dim=None)
    model = model.to(device)
    model.eval()

    fi_args = SimpleNamespace(
        ds="imagenet",
        tv=args.tv,
        opt_fi=args.opt_fi,
        sch_fi=args.sch_fi,
        lr_fi=args.lr_fi,
        n_iter_fi=args.n_iter_fi,
        gm=0.0,
        fi=args.fi,
        use_labels=True,
    )
    fia = Direct(recovered_z, model, fi_args, grads, y)

    rows: List[Dict[str, Any]] = []
    start = time.time()
    last_log = start
    rec = None
    for iteration in range(args.n_iter_fi):
        loss_dict = fia.step()
        should_log_loss = (
            iteration == 0
            or iteration == args.n_iter_fi - 1
            or (args.n_log_fi > 0 and iteration % args.n_log_fi == 0)
        )
        should_log_image_metrics = bool(
            metric_computer is not None
            and (
                iteration == 0
                or iteration == args.n_iter_fi - 1
                or iteration % image_metric_log_interval == 0
            )
        )
        if should_log_loss or should_log_image_metrics:
            now = time.time()
            row = {
                "iter": iteration,
                "loss": loss_dict["loss"],
                "loss_fi": loss_dict["loss_fi"],
                "loss_tv": loss_dict["loss_tv"],
                "seconds_since_last_log": now - last_log,
                "seconds_total": now - start,
            }
            rec = fia.get_rec()
            if should_log_image_metrics and metric_computer is not None:
                rec_images_now = rec.detach().clamp(0.0, 1.0)
                rec_eval_now = rec_images_now[eval_order_tensor]
                metric_summary_now, _metric_batch_now = metric_computer.compute(
                    original_images,
                    rec_eval_now,
                )
                row.update(
                    {
                        "image_metric_order": "eval_order",
                        "psnr_mean": metric_summary_now["psnr_mean"],
                        "psnr_min": metric_summary_now["psnr_min"],
                        "psnr_max": metric_summary_now["psnr_max"],
                        "ssim_mean": metric_summary_now["ssim_mean"],
                        "ssim_min": metric_summary_now["ssim_min"],
                        "ssim_max": metric_summary_now["ssim_max"],
                        "ssim_error": metric_summary_now["ssim_error"],
                        "lpips_mean": metric_summary_now["lpips_mean"],
                        "lpips_min": metric_summary_now["lpips_min"],
                        "lpips_max": metric_summary_now["lpips_max"],
                        "lpips_error": metric_summary_now["lpips_error"],
                    }
                )
            rows.append(row)
            print(
                "iter {iter:05d} | loss {loss:.6f} | loss_fi {loss_fi:.6f} | "
                "loss_tv {loss_tv:.6f} | total {seconds_total:.1f}s".format(**row),
                flush=True,
            )
            last_log = now

    if rec is None:
        rec = fia.get_rec()

    rec_images = rec.detach().clamp(0.0, 1.0)

    rec_eval = rec_images[eval_order_tensor]
    final_metric_computer = metric_computer or ImageMetricComputer(device, skip_lpips=args.skip_lpips)
    metrics_summary, metrics_batch = final_metric_computer.compute(original_images, rec_eval)

    original_np = original_images.detach().cpu().numpy()
    rec_attack_np = rec_images.detach().cpu().numpy()
    rec_eval_np = rec_eval.detach().cpu().numpy()

    image_metric_rows = [row for row in rows if "psnr_mean" in row]
    write_iter_csv(rows, output_dir / "fia_iter.csv")
    write_iter_csv(image_metric_rows, output_dir / "fia_image_metrics_iter.csv")
    write_json(image_metric_rows, output_dir / "fia_image_metrics_iter.json")
    write_pickle(
        {
            "input_pickle": str(input_pickle),
            "input_check": input_check,
            "fastica_dir": str(fastica_dir),
            "embedding_source": args.embedding_source,
            "x_original": original_np,
            "labels": y_np,
            "true_z": true_z_np,
            "recovered_embeddings": embeddings_np,
            "rec_attack_order": rec_attack_np,
            "rec_eval_order": rec_eval_np,
            "eval_order": eval_order.tolist(),
            "iter_rows": rows,
            "image_metric_iter_rows": image_metric_rows,
            "metrics_summary": metrics_summary,
            "metrics_batch": metrics_batch,
            "embedding_match": emb_match,
        },
        output_dir / "fastica_direct_fia_recon.pkl",
    )
    write_json(
        {
            "input_pickle": str(input_pickle),
            "input_check": input_check,
            "fastica_dir": str(fastica_dir),
            "embedding_source": args.embedding_source,
            "output_dir": str(output_dir),
            "device": str(device),
            "n_sample": n_sample,
            "n_iter_fi": args.n_iter_fi,
            "n_log_fi": args.n_log_fi,
            "log_image_metrics": args.log_image_metrics,
            "image_metric_log_interval": image_metric_log_interval
            if args.log_image_metrics
            else None,
            "skip_lpips": args.skip_lpips,
            "lr_fi": args.lr_fi,
            "tv": args.tv,
            "fi": args.fi,
            "opt_fi": args.opt_fi,
            "sch_fi": args.sch_fi,
            "embedding_match": emb_match,
            "metrics_summary": metrics_summary,
            "runtime_seconds": time.time() - start,
        },
        output_dir / "fia_metrics.json",
    )

    png_dir = output_dir / "png"
    make_pair_grid(original_np, rec_attack_np, png_dir / "fastica_fia_attack_order_compare.png")
    make_pair_grid(original_np, rec_eval_np, png_dir / "fastica_fia_eval_order_compare.png")

    print(f"FIA reconstruction saved: {output_dir / 'fastica_direct_fia_recon.pkl'}", flush=True)
    print(f"FIA metrics saved: {output_dir / 'fia_metrics.json'}", flush=True)
    print(f"FIA PNG saved: {png_dir / 'fastica_fia_eval_order_compare.png'}", flush=True)
    print(
        "final metrics | psnr {psnr_mean:.4f} | ssim {ssim_mean:.4f} | lpips {lpips_mean}".format(
            **metrics_summary
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
