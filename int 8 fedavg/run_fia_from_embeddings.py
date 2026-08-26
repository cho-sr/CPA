#!/usr/bin/env python3
"""Run Direct FIA from Naive or randomized client-q,s FastICA embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from assignment import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SRC = REPO_ROOT / "baseline" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from datasets import ds_mean, ds_std  # noqa: E402
from models import get_model  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "int 8 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct feature inversion from client-q,s FastICA embeddings."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--embedding_branch",
        type=str,
        default="randomized",
        choices=["randomized", "naive"],
        help="Use randomized_R*/final_embeddings.npy or naive/naive_embeddings.npy.",
    )
    parser.add_argument("--R", type=int, default=64)
    parser.add_argument(
        "--smoke_R",
        type=int,
        default=None,
        help="Use randomized_R{smoke_R}_smoke/final_embeddings.npy.",
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=None,
        help="Optional explicit embedding .npy path; overrides branch/R selection.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--n_sample", type=int, default=-1)
    parser.add_argument("--n_iter_fi", type=int, default=25000)
    parser.add_argument("--n_log_fi", type=int, default=500)
    parser.add_argument(
        "--fia_batch_size",
        type=int,
        default=4,
        help="Restore embeddings sequentially in chunks. Try 4--5 first; lower it if CUDA OOM occurs.",
    )
    parser.add_argument("--lr_fi", type=float, default=0.1)
    parser.add_argument("--tv", type=float, default=0.1)
    parser.add_argument("--fi", type=float, default=1.0)
    parser.add_argument("--opt_fi", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--sch_fi", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--try_both_signs",
        action="store_true",
        help="For each FIA chunk, try +embedding and -embedding from the same init and keep lower blind FIA loss.",
    )
    parser.add_argument(
        "--serial_signs",
        action="store_true",
        help="When --try_both_signs is set, run + and - sequentially instead of in one larger batch.",
    )
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--skip_ssim", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument(
        "--log_image_metrics",
        action="store_true",
        help="Log PSNR/SSIM/LPIPS at each FIA log step. This reads context images before FIA.",
    )
    parser.add_argument(
        "--image_metric_log_interval",
        type=int,
        default=None,
        help="Iteration interval for image metrics; defaults to --n_log_fi.",
    )
    parser.add_argument("--max_png_items", type=int, default=24)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_rmtree(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to delete outside {root}: {target}")
    if target.exists():
        shutil.rmtree(target)


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
        json.dump(data, handle, indent=2, sort_keys=True)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_int8_quant_meta(meta: Dict[str, Any]) -> None:
    checks = {
        "quant_bits_is_8": meta.get("quant_bits") == 8,
        "qmin_is_int8_symmetric": meta.get("qmin") == -127,
        "qmax_is_int8_symmetric": meta.get("qmax") == 127,
        "rounding_is_nearest": meta.get("rounding") == "nearest",
    }
    if not all(checks.values()):
        raise ValueError(f"Expected final-round client int8 q,s metadata; checks={checks}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_dir_for_args(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.embedding_branch == "naive":
        return run_dir / "naive"
    effective_R = args.smoke_R if args.smoke_R is not None else args.R
    suffix = "_smoke" if args.smoke_R is not None else ""
    return run_dir / f"randomized_R{effective_R}{suffix}"


def embedding_path_for_args(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.embedding_path is not None:
        path = resolve_path(args.embedding_path)
        expected_name = "naive_embeddings.npy" if args.embedding_branch == "naive" else "final_embeddings.npy"
        if path.name != expected_name:
            raise ValueError(
                f"--embedding_branch {args.embedding_branch!r} requires {expected_name}; got {path.name}"
            )
        return path
    branch_dir = embedding_dir_for_args(args, run_dir)
    if args.embedding_branch == "naive":
        return branch_dir / "naive_embeddings.npy"
    return branch_dir / "final_embeddings.npy"


def default_output_dir(args: argparse.Namespace, run_dir: Path, embedding_path: Path) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir)
    if args.embedding_path is not None:
        return embedding_path.parent / f"direct_fia_n{args.n_sample}_iter{args.n_iter_fi}"
    return embedding_dir_for_args(args, run_dir) / f"direct_fia_n{args.n_sample}_iter{args.n_iter_fi}"


def load_model_for_artifact(
    quant_meta: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    model_state_candidates = []
    if quant_meta.get("source_model_state_file"):
        model_state_candidates.append(Path(str(quant_meta["source_model_state_file"])))
    model_state_candidates.append(run_dir / "client_qs" / "final_round_global_model_before.pt")

    loaded_path = None
    for candidate in model_state_candidates:
        candidate = resolve_path(candidate)
        if candidate.is_file():
            loaded_path = candidate
            break
    if loaded_path is None:
        candidates = [str(resolve_path(path)) for path in model_state_candidates]
        raise FileNotFoundError(
            "Missing final-round checkpoint. Re-run "
            "'int 8 fedavg/collect_client_qs_final_round.py' so "
            "client_qs/final_round_global_model_before.pt is created. Checked: "
            + "; ".join(candidates)
        )

    model = get_model(model_name="vgg16", ds="imagenet", h_dim=None, dataparallel=False)
    state = torch.load(loaded_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model, {
        "model_source": "final_round_global_model_before",
        "model_state_path": str(loaded_path),
        "model_state_sha256": sha256_file(loaded_path),
    }


def validate_embedding_feature_dim(embeddings: np.ndarray, model: torch.nn.Module) -> Dict[str, Any]:
    classifier = getattr(model, "classifier", None)
    if classifier is None or len(classifier) < 1 or not hasattr(classifier[0], "in_features"):
        raise ValueError("Model does not expose classifier.0.in_features for feature-dimension validation")
    expected_dim = int(classifier[0].in_features)
    actual_dim = int(np.asarray(embeddings).shape[1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: final_embeddings dim={actual_dim}, "
            f"classifier.0 input dim={expected_dim}"
        )
    return {
        "ica_layer": "classifier.0.weight",
        "fia_feature": "classifier.0 input / flattened VGG feature before classifier.0",
        "embedding_dim": actual_dim,
        "expected_feature_dim": expected_dim,
    }


def unnormalize_images(x: torch.Tensor, ds: str = "imagenet") -> torch.Tensor:
    mean = torch.tensor(ds_mean[ds], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ds_std[ds], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


class ImageMetricComputer:
    def __init__(self, device: torch.device, *, skip_ssim: bool, skip_lpips: bool) -> None:
        self.ssim_fn = None
        self.ssim_error: Optional[str] = None
        if not skip_ssim:
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
        return summary, {
            "psnr": psnr_values,
            "ssim": ssim_values if ssim_values else None,
            "lpips": lpips_values if lpips_values else None,
        }


def clamp_normalized_images(x: torch.Tensor, ds: str = "imagenet") -> torch.Tensor:
    mean = torch.tensor(ds_mean[ds], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(ds_std[ds], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.max(torch.min(x, (1.0 - mean) / std), -mean / std)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def total_variation_per_sample(x: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]), dim=(1, 2, 3))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]), dim=(1, 2, 3))
    return dx + dy


def make_optimizer(params: List[torch.nn.Parameter], args: argparse.Namespace):
    if args.opt_fi == "adam":
        return torch.optim.Adam(params, lr=args.lr_fi)
    if args.opt_fi == "sgd":
        return torch.optim.SGD(params, lr=args.lr_fi)
    raise ValueError(f"unknown optimizer: {args.opt_fi}")


def make_scheduler(opt, args: argparse.Namespace):
    if args.sch_fi == "none":
        return None
    if args.sch_fi == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_iter_fi, last_epoch=-1)
    raise ValueError(f"unknown scheduler: {args.sch_fi}")


def run_direct_fia(
    embeddings: np.ndarray,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    *,
    initial_s_hat: torch.Tensor,
    chunk_start: int,
    sign_variant: str,
    loss_scale: float = 1.0,
    progress_path: Optional[Path] = None,
    original_images_for_logging: Optional[torch.Tensor] = None,
    image_metric_order: str = "provided_original_order",
    metric_computer: Optional[ImageMetricComputer] = None,
    image_metric_log_interval: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], float, np.ndarray]:
    recovered_z = torch.as_tensor(embeddings, device=device, dtype=torch.float32)
    n = int(recovered_z.shape[0])
    if tuple(initial_s_hat.shape) != (n, 3, 224, 224):
        raise ValueError(
            f"initial_s_hat shape mismatch: expected {(n, 3, 224, 224)}, "
            f"got {tuple(initial_s_hat.shape)}"
        )
    if original_images_for_logging is not None:
        if tuple(original_images_for_logging.shape) != (n, 3, 224, 224):
            raise ValueError(
                "original_images_for_logging shape mismatch: expected "
                f"{(n, 3, 224, 224)}, got {tuple(original_images_for_logging.shape)}"
            )
        original_images_for_logging = original_images_for_logging.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    s_hat = torch.nn.Parameter(initial_s_hat.detach().clone().to(device=device, dtype=torch.float32))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    opt = make_optimizer([s_hat], args)
    sch = make_scheduler(opt, args)
    cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)

    rows: List[Dict[str, Any]] = []
    start = time.time()
    last_log = start
    for iteration in range(args.n_iter_fi):
        with torch.no_grad():
            s_hat.copy_(clamp_normalized_images(s_hat, "imagenet"))
        opt.zero_grad(set_to_none=True)
        _pred, z_hat = model(s_hat, return_z=True)
        loss_fi = (1.0 - cosine(recovered_z, z_hat)).mean()
        loss_tv = total_variation(s_hat) if args.tv > 0 else torch.zeros((), device=device)
        loss_unscaled = args.fi * loss_fi + args.tv * loss_tv
        loss = float(loss_scale) * loss_unscaled
        loss.backward()
        opt.step()
        if sch is not None:
            sch.step()

        should_log_loss = (
            iteration == 0
            or iteration == args.n_iter_fi - 1
            or (args.n_log_fi > 0 and iteration % args.n_log_fi == 0)
        )
        should_log_image_metrics = bool(
            metric_computer is not None
            and original_images_for_logging is not None
            and (
                iteration == 0
                or iteration == args.n_iter_fi - 1
                or iteration % image_metric_log_interval == 0
            )
        )
        if should_log_loss or should_log_image_metrics:
            now = time.time()
            row = {
                "chunk_start": chunk_start,
                "chunk_end": chunk_start + n,
                "sign_variant": sign_variant,
                "iter": iteration,
                "loss": float(loss.detach().cpu().item()),
                "loss_unscaled": float(loss_unscaled.detach().cpu().item()),
                "loss_fi": float(loss_fi.detach().cpu().item()),
                "loss_tv": float(loss_tv.detach().cpu().item()),
                "seconds_since_last_log": now - last_log,
                "seconds_total": now - start,
            }
            if should_log_image_metrics and metric_computer is not None:
                with torch.no_grad():
                    rec_now = unnormalize_images(clamp_normalized_images(s_hat, "imagenet"), "imagenet")
                    metric_summary_now, _metric_batch_now = metric_computer.compute(
                        original_images_for_logging,
                        rec_now,
                    )
                row.update(
                    {
                        "image_metric_order": image_metric_order,
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
            if progress_path is not None:
                write_json(
                    {
                        "status": "running",
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_start + n,
                        "sign_variant": sign_variant,
                        "iter": iteration,
                        "n_iter_fi": args.n_iter_fi,
                        "fia_batch_size": args.fia_batch_size,
                        "try_both_signs": bool(args.try_both_signs),
                        "serial_signs": bool(args.serial_signs),
                        "latest": row,
                    },
                    progress_path,
                )

    with torch.no_grad():
        s_final = clamp_normalized_images(s_hat, "imagenet")
        _pred, z_final = model(s_final, return_z=True)
        loss_fi_by_sample = 1.0 - cosine(recovered_z, z_final)
        loss_tv_by_sample = (
            total_variation_per_sample(s_final)
            if args.tv > 0
            else torch.zeros((n,), device=device, dtype=torch.float32)
        )
        final_loss_by_sample = args.fi * loss_fi_by_sample + args.tv * loss_tv_by_sample
        rec = unnormalize_images(s_final, "imagenet")
    final_loss_np = final_loss_by_sample.detach().cpu().numpy().astype(np.float64, copy=False)
    final_blind_loss = float(np.mean(final_loss_np)) if len(final_loss_np) else float("nan")
    return rec.detach(), rows, final_blind_loss, final_loss_np


def initial_images_for_chunk(n: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((n, 3, 224, 224), device=device, dtype=torch.float32, generator=generator)


def run_direct_fia_chunked(
    embeddings: np.ndarray,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    *,
    progress_path: Optional[Path] = None,
    original_images_for_logging: Optional[torch.Tensor] = None,
    truth_index_for_recovered_for_logging: Optional[np.ndarray] = None,
    metric_computer: Optional[ImageMetricComputer] = None,
    image_metric_log_interval: int = 0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    rec_chunks: List[torch.Tensor] = []
    rows_all: List[Dict[str, Any]] = []
    selected_embeddings = np.asarray(embeddings, dtype=np.float32).copy()
    sign_rows: List[Dict[str, Any]] = []
    total_chunks = math.ceil(len(embeddings) / args.fia_batch_size)

    for start in range(0, len(embeddings), args.fia_batch_size):
        end = min(start + args.fia_batch_size, len(embeddings))
        chunk = np.asarray(embeddings[start:end], dtype=np.float32)
        init = initial_images_for_chunk(len(chunk), device, args.seed + start)
        original_chunk_for_logging = None
        if original_images_for_logging is not None:
            if truth_index_for_recovered_for_logging is not None:
                truth_indices = torch.as_tensor(
                    truth_index_for_recovered_for_logging[start:end],
                    dtype=torch.long,
                )
                original_chunk_for_logging = original_images_for_logging[truth_indices]
            else:
                original_chunk_for_logging = original_images_for_logging[start:end]

        chunk_index = (start // args.fia_batch_size) + 1
        print(
            f"FIA chunk {chunk_index:03d}/{total_chunks:03d} | samples {start}:{end} | "
            f"try_both_signs={args.try_both_signs} | serial_signs={args.serial_signs}",
            flush=True,
        )

        if args.try_both_signs and not args.serial_signs:
            candidate_embeddings = np.concatenate([chunk, -chunk], axis=0).astype(np.float32, copy=False)
            combined_init = init.repeat((2, 1, 1, 1))
            combined_originals_for_logging = (
                original_chunk_for_logging.repeat((2, 1, 1, 1))
                if original_chunk_for_logging is not None
                else None
            )
            rec_all, rows, _blind_loss, loss_by_sample = run_direct_fia(
                candidate_embeddings,
                model,
                args,
                device,
                initial_s_hat=combined_init,
                chunk_start=start,
                sign_variant="+/-parallel",
                loss_scale=2.0,
                progress_path=progress_path,
                original_images_for_logging=combined_originals_for_logging,
                image_metric_order="oracle_matched_truth_for_recovered_chunk"
                if truth_index_for_recovered_for_logging is not None
                else "chunk_attack_order",
                metric_computer=metric_computer,
                image_metric_log_interval=image_metric_log_interval,
            )
            n_chunk = len(chunk)
            variant_losses = {
                "+": float(np.mean(loss_by_sample[:n_chunk])),
                "-": float(np.mean(loss_by_sample[n_chunk:])),
            }
            if variant_losses["-"] < variant_losses["+"]:
                best = {
                    "sign_variant": "-",
                    "rec": rec_all[n_chunk:],
                    "rows": rows,
                    "blind_loss": variant_losses["-"],
                    "embeddings": -chunk,
                    "candidate_blind_losses": variant_losses,
                }
            else:
                best = {
                    "sign_variant": "+",
                    "rec": rec_all[:n_chunk],
                    "rows": rows,
                    "blind_loss": variant_losses["+"],
                    "embeddings": chunk,
                    "candidate_blind_losses": variant_losses,
                }
        else:
            variants = [("+", chunk)]
            if args.try_both_signs:
                variants.append(("-", -chunk))

            best = None
            candidate_blind_losses: Dict[str, float] = {}
            for sign_variant, candidate_embeddings in variants:
                rec, rows, blind_loss, _loss_by_sample = run_direct_fia(
                    candidate_embeddings,
                    model,
                    args,
                    device,
                    initial_s_hat=init,
                    chunk_start=start,
                    sign_variant=sign_variant,
                    progress_path=progress_path,
                    original_images_for_logging=original_chunk_for_logging,
                    image_metric_order="oracle_matched_truth_for_recovered_chunk"
                    if truth_index_for_recovered_for_logging is not None
                    else "chunk_attack_order",
                    metric_computer=metric_computer,
                    image_metric_log_interval=image_metric_log_interval,
                )
                candidate_blind_losses[sign_variant] = float(blind_loss)
                candidate = {
                    "sign_variant": sign_variant,
                    "rec": rec,
                    "rows": rows,
                    "blind_loss": blind_loss,
                    "embeddings": candidate_embeddings,
                    "candidate_blind_losses": dict(candidate_blind_losses),
                }
                if best is None or blind_loss < float(best["blind_loss"]):
                    best = candidate
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if best is None:
            raise RuntimeError("No FIA sign candidate was evaluated")
        rec_chunks.append(best["rec"])
        rows_all.extend(best["rows"])
        selected_embeddings[start:end] = np.asarray(best["embeddings"], dtype=np.float32)
        sign_rows.append(
            {
                "chunk_start": start,
                "chunk_end": end,
                "selected_sign_variant": best["sign_variant"],
                "selected_blind_loss": float(best["blind_loss"]),
                "candidate_blind_losses": best.get("candidate_blind_losses"),
                "tried_both_signs": bool(args.try_both_signs),
                "signs_parallelized": bool(args.try_both_signs and not args.serial_signs),
            }
        )

    rec_all = torch.cat(rec_chunks, dim=0)
    if progress_path is not None:
        write_json(
            {
                "status": "complete",
                "chunks": sign_rows,
                "fia_batch_size": args.fia_batch_size,
                "try_both_signs": bool(args.try_both_signs),
                "serial_signs": bool(args.serial_signs),
            },
            progress_path,
        )
    return rec_all, rows_all, {
        "policy": "try_both_signs_choose_lower_blind_fia_loss"
        if args.try_both_signs
        else "use_saved_blind_global_sign",
        "chunks": sign_rows,
        "fia_batch_size": args.fia_batch_size,
        "signs_parallelized": bool(args.try_both_signs and not args.serial_signs),
    }, selected_embeddings


def row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def embedding_order_for_truth(recovered: np.ndarray, true_z: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    similarity = row_normalize(true_z.astype(np.float64)) @ row_normalize(recovered.astype(np.float64)).T
    true_rows, recovered_cols = linear_sum_assignment(-np.abs(similarity))
    order = np.zeros(len(true_rows), dtype=np.int64)
    matched = np.zeros(len(true_rows), dtype=np.float64)
    signs = np.ones(len(true_rows), dtype=np.float64)
    for true_row, recovered_col in zip(true_rows, recovered_cols):
        value = float(similarity[true_row, recovered_col])
        order[true_row] = int(recovered_col)
        matched[true_row] = value
        signs[true_row] = 1.0 if value >= 0.0 else -1.0
    return order, {
        "policy": "oracle_hungarian_abs_cosine_eval_only",
        "true_z_used_only_after_fia": True,
        "mean_abs_cosine": float(np.mean(np.abs(matched))),
        "median_abs_cosine": float(np.median(np.abs(matched))),
        "min_abs_cosine": float(np.min(np.abs(matched))),
        "max_abs_cosine": float(np.max(np.abs(matched))),
        "recovered_order_for_true_index": order.tolist(),
        "matched_signed_cosines": matched.tolist(),
        "signs_for_eval_only": signs.tolist(),
    }


def to_uint8_chw(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return (np.transpose(image, (1, 2, 0)) * 255.0).round().astype(np.uint8)


def make_pair_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    path: Path,
    *,
    max_items: int,
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
        canvas.paste(Image.fromarray(to_uint8_chw(originals[idx])), (x0, y0 + label_h))
        canvas.paste(Image.fromarray(to_uint8_chw(reconstructions[idx])), (x0 + tile_w, y0 + label_h))
        draw.text((x0 + 3, y0 + 2), f"orig {idx}", fill=(0, 0, 0))
        draw.text((x0 + tile_w + 3, y0 + 2), "fia rec", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def compute_metrics(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    skip_ssim: bool,
    skip_lpips: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return ImageMetricComputer(
        reconstructions.device,
        skip_ssim=skip_ssim,
        skip_lpips=skip_lpips,
    ).compute(originals, reconstructions)


def main() -> None:
    args = parse_args()
    if args.fia_batch_size < 1:
        raise ValueError("--fia_batch_size must be >= 1")
    image_metric_log_interval = (
        args.n_log_fi if args.image_metric_log_interval is None else args.image_metric_log_interval
    )
    if args.log_image_metrics and image_metric_log_interval <= 0:
        raise ValueError("--image_metric_log_interval must be > 0 when --log_image_metrics is set")
    set_seed(args.seed)
    run_dir = resolve_path(args.run_dir)
    context_path = run_dir / "client_qs" / "final_round_context.pkl"
    quant_meta_path = run_dir / "client_qs" / "final_round_client_quant_meta.json"
    if not quant_meta_path.is_file():
        raise FileNotFoundError(quant_meta_path)
    if not context_path.is_file():
        raise FileNotFoundError(context_path)
    embedding_path = embedding_path_for_args(args, run_dir)
    if not embedding_path.is_file():
        raise FileNotFoundError(embedding_path)
    output_dir = default_output_dir(args, run_dir, embedding_path)
    if args.fresh_start:
        safe_rmtree(output_dir, allowed_root=run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_meta = read_json(quant_meta_path)
    validate_int8_quant_meta(quant_meta)
    embeddings_np = np.load(embedding_path).astype(np.float32, copy=False)
    n_available = len(embeddings_np)
    n_sample = n_available if args.n_sample == -1 else min(args.n_sample, n_available)
    if n_sample <= 0:
        raise ValueError("No samples available for FIA")

    embeddings_np = embeddings_np[:n_sample]

    device = choose_device(args.device)
    context_for_logging = None
    original_images_for_logging = None
    truth_index_for_recovered_for_logging = None
    embedding_match_for_logging = None
    if args.log_image_metrics:
        context_for_logging = read_pickle(context_path)
        x_np_for_logging = np.asarray(context_for_logging["x"], dtype=np.float32)[:n_sample]
        original_images_for_logging = unnormalize_images(
            torch.as_tensor(x_np_for_logging, dtype=torch.float32),
            "imagenet",
        ).detach()
        true_z_np_for_logging = np.asarray(context_for_logging["z"], dtype=np.float32)[:n_sample]
        eval_order_for_logging, embedding_match_for_logging = embedding_order_for_truth(
            embeddings_np,
            true_z_np_for_logging,
        )
        truth_index_for_recovered_for_logging = np.empty_like(eval_order_for_logging)
        truth_index_for_recovered_for_logging[eval_order_for_logging] = np.arange(
            len(eval_order_for_logging),
            dtype=np.int64,
        )
    metric_computer = (
        ImageMetricComputer(device, skip_ssim=args.skip_ssim, skip_lpips=args.skip_lpips)
        if args.log_image_metrics
        else None
    )
    model, model_info = load_model_for_artifact(quant_meta, run_dir, device)
    feature_info = validate_embedding_feature_dim(embeddings_np, model)
    print(
        f"Direct FIA | embeddings={embedding_path} | n_sample={n_sample} | "
        f"fia_batch_size={args.fia_batch_size} | device={device} | "
        f"model_source={model_info['model_source']} | feature_dim={feature_info['embedding_dim']}",
        flush=True,
    )

    rec_images, iter_rows, sign_selection, selected_embeddings = run_direct_fia_chunked(
        embeddings_np,
        model,
        args,
        device,
        progress_path=output_dir / "progress.json",
        original_images_for_logging=original_images_for_logging,
        truth_index_for_recovered_for_logging=truth_index_for_recovered_for_logging,
        metric_computer=metric_computer,
        image_metric_log_interval=image_metric_log_interval,
    )

    # Evaluation access starts here unless --log_image_metrics already read context images.
    context = context_for_logging if context_for_logging is not None else read_pickle(context_path)
    x_np = np.asarray(context["x"], dtype=np.float32)[:n_sample]
    y_np = np.asarray(context["y"])[:n_sample]
    original_images = unnormalize_images(torch.as_tensor(x_np, device=device, dtype=torch.float32), "imagenet")
    true_z_np = np.asarray(context["z"], dtype=np.float32)[:n_sample]
    eval_order, embedding_match = embedding_order_for_truth(selected_embeddings, true_z_np)
    rec_eval = rec_images[torch.as_tensor(eval_order, device=device, dtype=torch.long)]
    final_metric_computer = metric_computer or ImageMetricComputer(
        device,
        skip_ssim=args.skip_ssim,
        skip_lpips=args.skip_lpips,
    )
    metrics_summary, metrics_batch = final_metric_computer.compute(original_images, rec_eval)

    original_np = original_images.detach().cpu().numpy()
    rec_attack_np = rec_images.detach().cpu().numpy()
    rec_eval_np = rec_eval.detach().cpu().numpy()

    image_metric_rows = [row for row in iter_rows if "psnr_mean" in row]
    write_csv(iter_rows, output_dir / "fia_iter.csv")
    write_csv(image_metric_rows, output_dir / "fia_image_metrics_iter.csv")
    write_json(image_metric_rows, output_dir / "fia_image_metrics_iter.json")
    write_pickle(
        {
            "run_dir": str(run_dir),
            "context_path": str(context_path),
            "embedding_path": str(embedding_path),
            "embedding_branch": args.embedding_branch,
            "x_original": original_np,
            "labels": y_np,
            "recovered_embeddings": embeddings_np,
            "selected_embeddings_for_fia": selected_embeddings,
            "sign_selection": sign_selection,
            "rec_attack_order": rec_attack_np,
            "rec_eval_order": rec_eval_np,
            "eval_order": eval_order.tolist(),
            "iter_rows": iter_rows,
            "image_metric_iter_rows": image_metric_rows,
            "embedding_match_for_image_metric_logging": embedding_match_for_logging,
            "embedding_match": embedding_match,
            "metrics_summary": metrics_summary,
            "metrics_batch": metrics_batch,
            "model_info": model_info,
            "feature_info": feature_info,
        },
        output_dir / "direct_fia_recon.pkl",
    )
    manifest = {
        "schema_version": 1,
        "stage": "direct_fia_from_client_qs_fastica_embeddings",
        "attack_inputs": {
            "embedding_path": str(embedding_path.resolve()),
            "embedding_branch": args.embedding_branch,
            "embedding_sha256": sha256_file(embedding_path),
            "checkpoint_path": model_info["model_state_path"],
            "checkpoint_sha256": model_info["model_state_sha256"],
        },
        "eval_inputs": {
            "context_path": str(context_path.resolve()),
            "context_access_policy": (
                "context pickle read before FIA for x-image metric logging; "
                "true_z used only to build metric-order matching; labels/true_z not used for optimization"
                if args.log_image_metrics
                else "read only after FIA optimization completes"
            ),
        },
        "settings": {
            "n_sample": n_sample,
            "n_iter_fi": args.n_iter_fi,
            "n_log_fi": args.n_log_fi,
            "log_image_metrics": args.log_image_metrics,
            "image_metric_log_interval": image_metric_log_interval
            if args.log_image_metrics
            else None,
            "skip_ssim": args.skip_ssim,
            "skip_lpips": args.skip_lpips,
            "image_metric_truth_matching": "oracle_hungarian_abs_cosine_metric_only"
            if args.log_image_metrics
            else None,
            "fia_batch_size": args.fia_batch_size,
            "lr_fi": args.lr_fi,
            "tv": args.tv,
            "fi": args.fi,
            "opt_fi": args.opt_fi,
            "sch_fi": args.sch_fi,
            "seed": args.seed,
            "try_both_signs": args.try_both_signs,
            "serial_signs": args.serial_signs,
            "signs_parallelized": sign_selection["signs_parallelized"],
            "sign_policy": sign_selection["policy"],
        },
        "feature_info": feature_info,
        "true_z_policy": (
            "context pickle is read before FIA when --log_image_metrics is set; "
            "true_z is used only for metric-order matching and final eval, not optimization"
            if args.log_image_metrics
            else "not read until final evaluation after FIA"
        ),
    }
    write_json(manifest, output_dir / "manifest.json")
    write_json(
        {
            "run_dir": str(run_dir),
            "context_path": str(context_path),
            "embedding_path": str(embedding_path),
            "embedding_branch": args.embedding_branch,
            "embedding_sha256": manifest["attack_inputs"]["embedding_sha256"],
            "checkpoint_path": model_info["model_state_path"],
            "checkpoint_sha256": model_info["model_state_sha256"],
            "output_dir": str(output_dir),
            "device": str(device),
            "n_sample": n_sample,
            "n_iter_fi": args.n_iter_fi,
            "n_log_fi": args.n_log_fi,
            "log_image_metrics": args.log_image_metrics,
            "image_metric_log_interval": image_metric_log_interval
            if args.log_image_metrics
            else None,
            "skip_ssim": args.skip_ssim,
            "skip_lpips": args.skip_lpips,
            "image_metric_truth_matching": "oracle_hungarian_abs_cosine_metric_only"
            if args.log_image_metrics
            else None,
            "fia_batch_size": args.fia_batch_size,
            "lr_fi": args.lr_fi,
            "tv": args.tv,
            "fi": args.fi,
            "opt_fi": args.opt_fi,
            "sch_fi": args.sch_fi,
            "true_z_policy": (
                "used before FIA only for image-metric ordering when --log_image_metrics is set; "
                "never used for optimization"
                if args.log_image_metrics
                else "used only after FIA for eval-order matching and metrics"
            ),
            "embedding_match_for_image_metric_logging": embedding_match_for_logging,
            "sign_selection": sign_selection,
            "embedding_match": embedding_match,
            "metrics_summary": metrics_summary,
            "model_info": model_info,
            "feature_info": feature_info,
        },
        output_dir / "fia_metrics.json",
    )

    png_dir = output_dir / "png"
    make_pair_grid(
        original_np,
        rec_attack_np,
        png_dir / "randomized_fia_attack_order_compare.png",
        max_items=args.max_png_items,
    )
    make_pair_grid(
        original_np,
        rec_eval_np,
        png_dir / "randomized_fia_eval_order_compare.png",
        max_items=args.max_png_items,
    )
    print(f"FIA reconstruction saved: {output_dir / 'direct_fia_recon.pkl'}", flush=True)
    print(f"FIA metrics saved: {output_dir / 'fia_metrics.json'}", flush=True)
    print(f"FIA manifest saved: {output_dir / 'manifest.json'}", flush=True)
    print(f"FIA PNG saved: {png_dir / 'randomized_fia_eval_order_compare.png'}", flush=True)
    print(
        "final metrics | psnr {psnr_mean:.4f} | ssim {ssim_mean} | lpips {lpips_mean}".format(
            **metrics_summary
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
