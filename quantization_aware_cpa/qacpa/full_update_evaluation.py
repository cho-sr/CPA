"""Offline evaluation. This module is never imported by the attack pipeline."""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import torch


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_MEAN = (.485, .456, .406)
_STD = (.229, .224, .225)


def _image_tensor(payload, key: str, role: str, *, require_labels: bool = True) -> torch.Tensor:
    if not isinstance(payload, Mapping) or key not in payload or (require_labels and "labels" not in payload):
        raise ValueError(f"{role} payload must contain {key} and labels")
    images = payload[key]
    if (not isinstance(images, torch.Tensor) or images.layout != torch.strided
            or images.dtype not in (torch.float32, torch.float64)):
        raise ValueError(f"{role} images must be strided FP32 or FP64 tensors")
    if images.ndim != 4 or images.shape[0] == 0 or images.shape[1:] != (3, 224, 224):
        raise ValueError(f"{role} images must have nonempty [N,3,224,224] shape")
    if not bool(torch.isfinite(images).all()):
        raise ValueError(f"{role} images must be finite")
    return images


def _labels(payload, n: int, role: str) -> torch.Tensor:
    labels = payload["labels"]
    if (not isinstance(labels, torch.Tensor) or labels.layout != torch.strided
            or labels.dtype not in _INTEGER_DTYPES or tuple(labels.shape) != (n,)):
        raise ValueError(f"{role} labels must be an integer tensor with exact shape [N]")
    # Promote only after integer validation: uint8 comparisons with 1000 can
    # otherwise wrap the scalar to the narrow dtype on some torch versions.
    integer_labels = labels.to(dtype=torch.int64)
    if bool(((integer_labels < 0) | (integer_labels >= 1000)).any()):
        raise ValueError(f"{role} labels must be ImageNet indices in [0,999]")
    return labels


@torch.no_grad()
def evaluate_candidate(candidate_path: Path, truth_path: Path, *, include_labels: bool = True) -> dict:
    """Use one predeclared pixel-MSE Hungarian matching for all reported metrics.

    Evaluation reads exactly the two supplied files. It cannot select an attack
    checkpoint or feed a match back into the attack. LPIPS and scientific success
    thresholds are not implicitly selected. Truth contains ImageNet-normalized
    RGB values; candidate pixels must already be in [0,1]. Inverse-normalization
    permits only eight input-dtype epsilons of pixel-space roundoff at the bounds,
    checked before clamping. Larger errors are rejected, never silently repaired.
    include_labels=False evaluates images only without reading or inventing labels;
    the default preserves the existing label validation and accuracy output.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from torchmetrics.functional.image import structural_similarity_index_measure

    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    truth = torch.load(truth_path, map_location="cpu", weights_only=True)
    pixels_input = _image_tensor(candidate, "pixels", "Candidate", require_labels=include_labels)
    images_input = _image_tensor(truth, "x", "Truth", require_labels=include_labels)
    if pixels_input.shape != images_input.shape:
        raise ValueError("Candidate and truth image shapes must match exactly")
    if include_labels:
        candidate_labels = _labels(candidate, len(pixels_input), "Candidate")
        truth_labels = _labels(truth, len(images_input), "Truth")
    pixels, images = pixels_input.double(), images_input.double()
    if bool(((pixels < 0) | (pixels > 1)).any()):
        raise ValueError("Candidate pixels must be in [0,1]")
    mean = images.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_STD).view(1, 3, 1, 1)
    reference = images * std + mean
    tolerance = 8.0 * torch.finfo(images_input.dtype).eps
    if (not bool(torch.isfinite(reference).all())
            or bool(((reference < -tolerance) | (reference > 1.0 + tolerance)).any())):
        raise ValueError("Truth must be ImageNet-normalized RGB representable in [0,1]")
    clipped_values = int(((reference < 0) | (reference > 1)).sum())
    reference = reference.clamp(0, 1)
    n = pixels.shape[0]
    costs = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            costs[i, j] = float((pixels[i] - reference[j]).square().mean())
    candidate_rows, truth_rows = linear_sum_assignment(costs)
    per_image = []
    for i, j in zip(candidate_rows.tolist(), truth_rows.tolist()):
        mse = float(costs[i, j])
        if mse == 0 and not torch.equal(pixels[i], reference[j]):
            raise ValueError("MSE underflow for unequal images; cannot report a perfect pixel match")
        psnr = -10 * math.log10(mse) if mse > 0 else None
        ssim = float(structural_similarity_index_measure(pixels[i:i+1], reference[j:j+1],
                                                         data_range=1.0))
        if not math.isfinite(ssim):
            raise ValueError("SSIM produced a nonfinite value")
        per_image.append({"candidate_index": i, "truth_index": j, "mse": mse, "psnr_db": psnr,
                          "perfect_pixel_match": mse == 0, "ssim": ssim})
        if include_labels:
            per_image[-1]["label_correct"] = int(candidate_labels[i]) == int(truth_labels[j])
    finite_psnr = [item["psnr_db"] for item in per_image if item["psnr_db"] is not None]
    return {"matching": "Hungarian_pixel_MSE_once_for_all_metrics", "n_images": n,
            "all_images_included": True, "candidate_selection": "none_single_supplied_artifact",
            "per_image": per_image, "metric_dtype": "float64",
            "truth_denormalization_tolerance_abs": tolerance,
            "truth_denormalization_clipped_values": clipped_values,
            "perfect_pixel_matches": sum(item["perfect_pixel_match"] for item in per_image),
            "psnr_null_meaning": "positive_infinity_for_perfect_pixel_match",
            "mean_psnr_null_meaning": "positive_infinity_if_any_image_is_a_perfect_pixel_match",
            "mean_mse": sum(item["mse"] for item in per_image) / n,
            "mean_psnr_db": sum(finite_psnr) / n if len(finite_psnr) == n else None,
            "mean_ssim": sum(item["ssim"] for item in per_image) / n,
            **({"label_accuracy": sum(item["label_correct"] for item in per_image) / n} if include_labels else {}),
            "lpips": "not_computed", "reconstruction_success": "no_threshold_registered"}
