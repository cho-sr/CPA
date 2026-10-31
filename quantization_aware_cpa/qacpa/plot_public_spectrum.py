"""Plot already-computed public MP diagnostic results, with no model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .mp_spectrum import mp_quantiles


def find_spectra(value, path=""):
    """Read diagnostic fit records regardless of the runner's grouping layout."""
    if isinstance(value, dict):
        if (value.get("schema_version") == "mp_spectrum_quantile_fit_v1"
                and "raw_eigenvalues_desc" in value):
            yield path, value
        else:
            for key, item in value.items():
                yield from find_spectra(item, f"{path}/{key}".strip("/"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from find_spectra(item, f"{path}/{index}".strip("/"))


def render(report_path: Path, output: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = list(find_spectra(report))
    if not records:
        raise ValueError("No computed MP diagnostic spectra in the supplied report")
    if output.exists():
        raise FileExistsError(output)
    fig, axes = plt.subplots(len(records), 3, figsize=(15, 3.8 * len(records)), squeeze=False)
    palette = ["#176b87", "#a63a50", "#4a7c59"]
    for row, (name, record) in enumerate(records):
        color = palette[row % len(palette)]
        label = next((label for key, label in {
            "positive_all": "All positive-radius rows",
            "at_or_above_floor": "Rows at or above the public step floor",
            "below_floor": "Positive rows below the public step floor",
        }.items() if key in name), name)
        values = np.asarray([np.nan if v is None else v for v in record["raw_eigenvalues_desc"]])
        ranks = np.arange(1, len(values) + 1)
        first = axes[row, 0]
        first.semilogy(ranks, np.where(values > 0, values, np.nan), color=color, lw=1.2)
        first.axvline(16, color="#777777", lw=0.8, ls=":", label="rank 16")
        if record.get("edge") is not None:
            first.axhline(record["edge"], color="#c77c02", ls="--", lw=1, label="fitted MP edge")
        fit_status = "finite fit; validity not established" if record["status"] == "OK" else record["status"]
        first.set(title=f"{label}\nm={record['m']}; {fit_status}",
                  xlabel="Descending eigenvalue rank", ylabel="Raw covariance eigenvalue")
        first.legend(fontsize=8)
        second, third = axes[row, 1:]
        fit = record.get("primary_fit") or {}
        if fit.get("status") == "OK":
            indices = np.asarray(fit["ascending_indices"], dtype=int)
            observed = values[::-1][indices]
            predicted = record["sigma2"] * np.asarray(fit["mp_unit_quantiles"])
            second.plot(predicted, observed, ".", color=color, ms=2, alpha=0.65)
            lower, upper = min(predicted.min(), observed.min()), max(predicted.max(), observed.max())
            second.plot([lower, upper], [lower, upper], "--", color="#777777", lw=1)
            second.set(title="Primary bulk quantile fit (p=0.2–0.8)",
                       xlabel="Fitted MP quantile", ylabel="Observed eigenvalue")
            positions = (np.arange(len(values)) + 0.5) / len(values)
            keep = np.arange(len(values)) < max(0, len(values) - record["exclude_top"])
            theory = mp_quantiles(record["beta"], positions[keep]) * record["sigma2"]
            ratio = np.maximum(values[::-1][keep], 0) / theory
            third.plot(positions[keep], ratio, color=color, lw=1)
            third.axhline(1, color="#777777", ls="--", lw=1)
            third.axvspan(0.2, 0.8, color=color, alpha=0.07)
            third.set_yscale("symlog", linthresh=1)
            third.set(title="Observed / fitted MP across retained ranks",
                      xlabel="Original rank probability", ylabel="Eigenvalue ratio (symlog)")
        else:
            for axis in (second, third):
                axis.text(0.5, 0.5, record.get("reason") or "Fit unavailable",
                          ha="center", va="center", wrap=True, transform=axis.transAxes)
                axis.set_axis_off()
        for axis in axes[row]:
            axis.grid(alpha=0.18)
    fig.suptitle("Public deterministic FC1 spectrum diagnostic — descriptive fit only", fontsize=15)
    fig.text(0.5, 0.008,
             "No true updates, images, or labels. MP resemblance is not an iid-noise or reconstruction-success test.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(render(args.report, args.output))


if __name__ == "__main__":
    main()
