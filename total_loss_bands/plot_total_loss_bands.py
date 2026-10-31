"""Plot recorded CPA/FIA total loss for center and randomized candidates.

Run with the project plotting environment:
    C:/miniconda/envs/DLG/python.exe total_loss_bands/plot_total_loss_bands.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
BITS = (4, 8)
STAGES = ("cpa", "fia")
STEP_LABELS = {1: "1", 5000: "5k", 10000: "10k", 15000: "15k", 20000: "20k", 25000: "25k"}
BLUE = "#2457C5"
GRAY = "#AEB6C2"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect(bit: int, stage: str) -> list[dict]:
    root = PROJECT_ROOT / f"bit{bit:02d}"
    candidates = [root / f"candidate{i:03d}" for i in range(100)]
    if not all(path.is_dir() for path in candidates):
        raise ValueError(f"Expected candidate000..candidate099 in {root}")

    series = []
    for candidate_id, path in enumerate(candidates):
        complete = read_json(path / "complete.json")
        identity = complete["identity"]
        expected_kind = "center" if candidate_id == 0 else "random"
        if (complete["state"] != "COMPLETED"
                or identity["bits"] != bit
                or identity["candidate_id"] != candidate_id
                or identity["kind"] != expected_kind
                or complete[f"{stage}_completed"] != 25000):
            raise ValueError(f"Unexpected completion/identity: {path}")

        log = read_json(path / f"{stage}_result.json")["log"]
        points = [(row["iter"] + 1, float(row["loss"])) for row in log]
        if not points or any(not math.isfinite(value) for _, value in points):
            raise ValueError(f"Missing or nonfinite total loss: {path}")
        series.append(points)

    steps = [step for step, _ in series[0]]
    if steps != [1, 5000, 10000, 15000, 20000, 25000]:
        raise ValueError(f"Unexpected logged steps: {steps}")
    if any([step for step, _ in points] != steps for points in series):
        raise ValueError(f"Candidate step mismatch: {root}, {stage}")

    rows = []
    for position, step in enumerate(steps):
        random_values = [points[position][1] for points in series[1:]]
        rows.append({
            "bits": bit,
            "stage": stage,
            "step": step,
            "dequantization": series[0][position][1],
            "random_min": min(random_values),
            "random_max": max(random_values),
            "random_mean": sum(random_values) / len(random_values),
            "random_n": len(random_values),
        })
    return rows


def draw(ax, rows: list[dict], *, legend: bool = True) -> None:
    x = [row["step"] for row in rows]
    center = [row["dequantization"] for row in rows]
    minimum = [row["random_min"] for row in rows]
    maximum = [row["random_max"] for row in rows]

    ax.fill_between(x, minimum, maximum, color=GRAY, alpha=0.68,
                    label="Random min–max (n=99)", zorder=1)
    ax.plot(x, minimum, color="#828C99", linewidth=0.8, alpha=0.85, zorder=2)
    ax.plot(x, maximum, color="#828C99", linewidth=0.8, alpha=0.85, zorder=2)
    ax.plot(x, center, color=BLUE, linewidth=2.8, marker="o", markersize=4.5,
            label="Dequantization (center)", zorder=3)
    ax.set_xlim(-350, 25350)
    ax.set_xticks(x, [STEP_LABELS[step] for step in x])
    ax.grid(axis="both", color="#D7DCE2", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(loc="best", frameon=True, fontsize=9)


def save_individual(rows: list[dict]) -> str:
    bit = rows[0]["bits"]
    stage = rows[0]["stage"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=170)
    draw(ax, rows)
    ax.set_title(f"{bit}-bit | {stage.upper()} total loss", loc="left",
                 fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Iteration (attack optimization step)", fontsize=11)
    ax.set_ylabel("Total loss", fontsize=11)
    last = rows[-1]
    fig.subplots_adjust(bottom=0.23, top=0.87, left=0.12, right=0.97)
    fig.text(0.12, 0.13,
             f"At step 25,000: dequantization {last['dequantization']:.6g}  |  "
             f"random min–max {last['random_min']:.6g}–{last['random_max']:.6g}",
             fontsize=10, color="#273446")
    fig.text(0.12, 0.075,
             "Six recorded steps; straight segments join measurements. Shading is min–max, not SD.",
             fontsize=9, color="#59677A")
    filename = f"bit{bit:02d}_{stage}_total_loss.png"
    fig.savefig(OUTPUT / filename, dpi=170, facecolor="white")
    plt.close(fig)
    return filename


def save_overview(grouped: dict[tuple[int, str], list[dict]]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160, sharey="row")
    for row_index, stage in enumerate(STAGES):
        for column_index, bit in enumerate(BITS):
            ax = axes[row_index, column_index]
            draw(ax, grouped[(bit, stage)], legend=False)
            ax.set_title(f"{bit}-bit | {stage.upper()} total loss", loc="left",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Attack optimization step")
            if column_index == 0:
                ax.set_ylabel("Total loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=10)
    fig.suptitle("Dequantization and randomized candidate ranges", fontsize=17,
                 fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.09, right=0.97,
                        hspace=0.36, wspace=0.17)
    fig.text(0.09, 0.035,
             "Range = min–max across 99 random candidates at each logged step; same y scale across bits within each stage.",
             fontsize=9, color="#59677A")
    filename = "overview_total_loss.png"
    fig.savefig(OUTPUT / filename, dpi=160, facecolor="white")
    plt.close(fig)
    return filename


def main() -> None:
    grouped = {(bit, stage): collect(bit, stage) for bit in BITS for stage in STAGES}
    rows = [row for bit in BITS for stage in STAGES for row in grouped[(bit, stage)]]

    csv_path = OUTPUT / "total_loss_minmax.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    images = [save_individual(grouped[(bit, stage)]) for bit in BITS for stage in STAGES]
    images.append(save_overview(grouped))
    manifest = {
        "source_directories": ["bit04", "bit08"],
        "center": "candidate000 (q * scale)",
        "random_candidates": "candidate001..candidate099",
        "metric": "recorded CPA/FIA total loss",
        "x_axis": "attack optimization step (logged iter + 1)",
        "logged_steps": [1, 5000, 10000, 15000, 20000, 25000],
        "band": "pointwise min and max of 99 random candidate losses",
        "interpolation": "straight segments between recorded values; no additional measurements",
        "artifacts": ["total_loss_minmax.csv", *images],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} PNGs, {len(rows)} CSV rows, and manifest to {OUTPUT}")


if __name__ == "__main__":
    main()
