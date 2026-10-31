"""Metric summary helpers for sparse-update CPA experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from common import filesystem_path


SUMMARY_METRICS = (
    "lpips",
    "psnr",
    "ssim",
    "embedding_cosine_similarity",
    "reconstruction_success_rate",
)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    result = []
    for row in rows:
        value = _finite_float(row.get(key))
        if value is not None:
            result.append(value)
    return result


def _summary_row(metric: str, values: Sequence[float], *, threshold: float | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "n": 0,
        "threshold": threshold,
    }
    if values:
        row.update(
            {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "n": int(len(values)),
            }
        )
    return row


def summarize_metric_rows(
    *,
    image_rows: Sequence[Mapping[str, Any]],
    eval_entries: Sequence[Mapping[str, Any]],
    success_lpips_threshold: float,
) -> list[dict[str, Any]]:
    """Build compact rows for metric_summary.csv.

    ``eval_entries`` follows the runner's oracle_evaluation.json shape.  The
    embedding cosine is evaluation-only and is never fed back into CPA/FIA.
    """
    rows = [
        _summary_row("lpips", _values(image_rows, "lpips")),
        _summary_row("psnr", _values(image_rows, "psnr")),
        _summary_row("ssim", _values(image_rows, "ssim")),
    ]

    embedding_values: list[float] = []
    for entry in eval_entries:
        embedding = entry.get("embedding")
        if not isinstance(embedding, Mapping):
            continue
        per_sample = embedding.get("per_sample_cs")
        if isinstance(per_sample, Sequence) and not isinstance(per_sample, (str, bytes)):
            embedding_values.extend(value for value in (_finite_float(v) for v in per_sample) if value is not None)
        else:
            value = _finite_float(embedding.get("mean_cs"))
            if value is not None:
                embedding_values.append(value)
    rows.append(_summary_row("embedding_cosine_similarity", embedding_values))

    success_flags = []
    for value in _values(image_rows, "lpips"):
        success_flags.append(1.0 if value <= success_lpips_threshold else 0.0)
    rows.append(
        _summary_row(
            "reconstruction_success_rate",
            success_flags,
            threshold=float(success_lpips_threshold),
        )
    )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(filesystem_path(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
