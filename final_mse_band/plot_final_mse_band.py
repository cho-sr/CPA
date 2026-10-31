"""Plot final MSE for measured 4-bit and 8-bit candidate groups."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
TABLES = ROOT / "metric_winners_v2" / "metric_winners_v2" / "tables"


def summarize(bit: int) -> dict:
    with (TABLES / f"bit{bit:02d}_all100_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or [int(row["candidate_id"]) for row in rows] != list(range(100)):
        raise ValueError(f"Expected candidates 0..99 for {bit}-bit")
    if any(int(row["bits"]) != bit for row in rows):
        raise ValueError(f"Bit identity mismatch for {bit}-bit")

    values = [float(row["mse"]) for row in rows]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Nonfinite MSE for {bit}-bit")
    random = values[1:]
    return {
        "bits": bit,
        "step": 25000,
        "dequantization_mse": values[0],
        "random_min_mse": min(random),
        "random_max_mse": max(random),
        "random_mean_mse": sum(random) / len(random),
        "random_n": len(random),
    }


def main() -> None:
    rows = [summarize(bit) for bit in (4, 8)]
    with (OUTPUT / "final_mse_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = [0, 1]
    center = [row["dequantization_mse"] for row in rows]
    minimum = [row["random_min_mse"] for row in rows]
    maximum = [row["random_max_mse"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=180)
    ax.fill_between(x, minimum, maximum, color="#AEB6C2", alpha=0.7,
                    label="Random min–max (n=99)", zorder=1)
    ax.plot(x, minimum, color="#828C99", linewidth=0.8, zorder=2)
    ax.plot(x, maximum, color="#828C99", linewidth=0.8, zorder=2)
    ax.plot(x, center, color="#2457C5", linewidth=2.8, marker="o", markersize=7,
            label="Dequantization (center)", zorder=3)
    ax.set_xticks(x, ["4-bit", "8-bit"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_xlabel("Measured bit width", fontsize=11)
    ax.set_ylabel("Final reconstruction MSE (lower is better)", fontsize=11)
    ax.set_title("Final MSE at FIA step 25,000", loc="left", fontsize=16,
                 fontweight="bold", pad=14)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=True)

    fig.subplots_adjust(bottom=0.21, top=0.87, left=0.16, right=0.97)
    summary = "   |   ".join(
        f"{row['bits']}-bit: center {row['dequantization_mse']:.6f}, "
        f"random {row['random_min_mse']:.6f}–{row['random_max_mse']:.6f}"
        for row in rows
    )
    fig.text(0.16, 0.12, summary, fontsize=9.3, color="#273446")
    fig.text(0.16, 0.07,
             "Only 4-bit and 8-bit final results were measured here; connecting segments guide the eye.",
             fontsize=9, color="#59677A")
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"final_mse_center_random_minmax.{suffix}",
                    facecolor="white", dpi=180)
    plt.close(fig)

    manifest = {
        "source_tables": [
            "metric_winners_v2/metric_winners_v2/tables/bit04_all100_metrics.csv",
            "metric_winners_v2/metric_winners_v2/tables/bit08_all100_metrics.csv",
        ],
        "metric": "final 16-image mean MSE",
        "step": 25000,
        "center": "candidate000",
        "band": "pointwise min and max of candidates001..099 at each measured bit width",
        "measured_bit_widths": [4, 8],
        "intermediate_steps_available": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote final MSE plot and summary to {OUTPUT}")


if __name__ == "__main__":
    main()
