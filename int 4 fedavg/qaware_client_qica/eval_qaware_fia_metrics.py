#!/usr/bin/env python3
"""Post-evaluate q-aware FIA reconstructions with PSNR/SSIM/LPIPS."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "int 4 fedavg"
    / "imagenet"
    / "vgg16"
    / "clients3_rounds20_spc8_seed42"
    / "qaware_client_qica"
    / "qica_embedding_qaware_fia_attack_layer_oracle_context_n24_it1000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute missing q-aware FIA image metrics from a saved recon pkl.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recon_pkl", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip_ssim", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--update_manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update_progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class ImageMetricComputer:
    def __init__(self, device: torch.device, *, skip_ssim: bool, skip_lpips: bool) -> None:
        self.ssim_fn = None
        self.ssim_error: str | None = None
        if not skip_ssim:
            try:
                from torchmetrics.functional import structural_similarity_index_measure

                self.ssim_fn = structural_similarity_index_measure
            except Exception as exc:
                self.ssim_error = str(exc)

        self.lpips_model = None
        self.lpips_error: str | None = None
        if not skip_lpips:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

                self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
                self.lpips_model.eval()
            except Exception as exc:
                self.lpips_error = str(exc)

    def compute(self, originals: torch.Tensor, reconstructions: torch.Tensor) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        mse_values: List[float] = []
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        lpips_values: List[float] = []
        with torch.no_grad():
            for idx, (orig, rec) in enumerate(zip(originals, reconstructions)):
                orig_b = orig.unsqueeze(0)
                rec_b = rec.unsqueeze(0)
                mse = torch.mean((orig_b - rec_b) ** 2).clamp_min(1e-12)
                psnr = 10.0 * torch.log10(1.0 / mse)
                row: Dict[str, Any] = {
                    "sample": idx,
                    "mse": float(mse.detach().cpu()),
                    "psnr": float(psnr.detach().cpu()),
                    "ssim": None,
                    "lpips": None,
                }
                mse_values.append(float(row["mse"]))
                psnr_values.append(float(row["psnr"]))
                if self.ssim_fn is not None:
                    value = float(self.ssim_fn(rec_b, orig_b).detach().cpu().item())
                    row["ssim"] = value
                    ssim_values.append(value)
                if self.lpips_model is not None:
                    value = float(self.lpips_model(rec_b * 2.0 - 1.0, orig_b * 2.0 - 1.0).detach().cpu().item())
                    row["lpips"] = value
                    lpips_values.append(value)
                rows.append(row)

        summary: Dict[str, Any] = {
            "mse_mean": float(np.mean(mse_values)),
            "mse_min": float(np.min(mse_values)),
            "mse_max": float(np.max(mse_values)),
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
        return summary, rows


def metric_batch_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in ("mse", "psnr", "ssim", "lpips"):
        values = [row.get(key) for row in rows]
        result[key] = values if any(value is not None for value in values) else None
    return result


def update_json_metrics(path: Path, metrics_summary: Dict[str, Any]) -> None:
    if not path.is_file():
        return
    data = read_json(path)
    data["metrics_summary"] = metrics_summary
    write_json(data, path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    recon_pkl = args.recon_pkl.resolve() if args.recon_pkl is not None else output_dir / "end_to_end_qaware_fia_recon.pkl"
    if not recon_pkl.is_file():
        raise FileNotFoundError(recon_pkl)

    data = read_pickle(recon_pkl)
    originals = torch.as_tensor(np.asarray(data["x_original"], dtype=np.float32), device=choose_device(args.device))
    reconstructions = torch.as_tensor(np.asarray(data["rec_attack_order"], dtype=np.float32), device=originals.device)
    metric_computer = ImageMetricComputer(
        originals.device,
        skip_ssim=bool(args.skip_ssim),
        skip_lpips=bool(args.skip_lpips),
    )
    metrics_summary, metrics_by_sample = metric_computer.compute(originals, reconstructions)
    metrics_batch = metric_batch_from_rows(metrics_by_sample)

    write_csv(metrics_by_sample, output_dir / "image_metrics_by_sample.csv")
    write_json(
        {
            "run_tag": data.get("run_tag", "qaware_post_eval"),
            "metrics_summary": metrics_summary,
            "metrics_by_sample": metrics_by_sample,
            "metrics_batch": metrics_batch,
            "source_recon_pkl": str(recon_pkl),
        },
        output_dir / "fia_metrics.json",
    )
    data["metrics_summary"] = metrics_summary
    data["metrics_by_sample"] = metrics_by_sample
    data["metrics_batch"] = metrics_batch
    write_pickle(data, recon_pkl)
    if args.update_manifest:
        update_json_metrics(output_dir / "manifest.json", metrics_summary)
    if args.update_progress:
        update_json_metrics(output_dir / "progress.json", metrics_summary)

    print(
        "q-aware FIA metrics | psnr {psnr_mean:.4f} | ssim {ssim_mean} | lpips {lpips_mean}".format(
            **metrics_summary
        ),
        flush=True,
    )
    print(f"saved: {output_dir / 'fia_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
