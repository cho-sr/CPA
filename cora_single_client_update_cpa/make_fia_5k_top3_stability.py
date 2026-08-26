#!/usr/bin/env python3
"""Check whether the 5k FIA-loss top-3 remains top-3 at later checkpoints."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT_CSV = (
    ROOT
    / "outputs_v1"
    / "png_report"
    / "single_client_fia_loss_members"
    / "single_client_fia_loss_members_5k.csv"
)
OUTPUT_DIR = ROOT / "outputs_v1" / "png_report" / "single_client_fia_5k_top3_stability"

KEEP_ORDER = ["0p8", "0p5", "0p3"]
KEEP_LABELS = {"0p8": "Keep 80%", "0p5": "Keep 50%", "0p3": "Keep 30%"}
CHECKPOINTS = [5000, 10000, 15000, 20000, 25000]


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top3_at(rows: list[dict[str, str]], keep: str, iteration: int) -> list[int]:
    selected = [
        row
        for row in rows
        if row["keep"] == keep and int(row["iter"]) == int(iteration)
    ]
    if len(selected) < 3:
        raise ValueError(f"need at least 3 rows for keep={keep} iter={iteration}")
    selected.sort(key=lambda row: float(row["fia_loss"]))
    return [int(row["member"]) for row in selected[:3]]


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for keep in KEEP_ORDER:
        base_top3 = top3_at(rows, keep, 5000)
        base_set = set(base_top3)
        for iteration in CHECKPOINTS:
            current_top3 = top3_at(rows, keep, iteration)
            current_set = set(current_top3)
            retained = [member for member in base_top3 if member in current_set]
            missing = [member for member in base_top3 if member not in current_set]
            new_entries = [member for member in current_top3 if member not in base_set]
            summary.append(
                {
                    "keep": keep,
                    "keep_label": KEEP_LABELS[keep],
                    "checkpoint": iteration,
                    "checkpoint_label": f"{iteration // 1000}k",
                    "top3_5k": ", ".join(f"m{member:03d}" for member in base_top3),
                    "top3_current": ", ".join(f"m{member:03d}" for member in current_top3),
                    "retained_count": len(retained),
                    "retained_members": ", ".join(f"m{member:03d}" for member in retained),
                    "missing_from_5k_top3": ", ".join(f"m{member:03d}" for member in missing),
                    "new_entries": ", ".join(f"m{member:03d}" for member in new_entries),
                    "all_5k_top3_still_top3": len(retained) == 3,
                }
            )
    return summary


def write_csv(summary: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "keep",
        "keep_label",
        "checkpoint",
        "checkpoint_label",
        "top3_5k",
        "top3_current",
        "retained_count",
        "retained_members",
        "missing_from_5k_top3",
        "new_entries",
        "all_5k_top3_still_top3",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot(summary: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update({"font.size": 13.5})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Stability of the 5k FIA-Loss Top-3 Across Checkpoints",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.88,
        "Single-client CoRA / ImageNet / VGG16 / n=24 / R=8",
        ha="center",
        va="center",
        fontsize=16,
        color="#475569",
        transform=ax.transAxes,
    )

    columns = ["Keep", "5k top-3", "10k", "15k", "20k", "25k"]
    cell_text: list[list[str]] = []
    for keep in KEEP_ORDER:
        keep_rows = [row for row in summary if row["keep"] == keep]
        keep_rows.sort(key=lambda row: int(row["checkpoint"]))
        base_top3 = str(keep_rows[0]["top3_5k"])
        cells = [KEEP_LABELS[keep], base_top3]
        for row in keep_rows[1:]:
            retained_count = int(row["retained_count"])
            current = str(row["top3_current"])
            retained = str(row["retained_members"])
            cells.append(f"{retained_count}/3\n{current}\nkept: {retained}")
        cell_text.append(cells)

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.18, 0.175, 0.175, 0.175, 0.175],
        bbox=[0.035, 0.22, 0.93, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.2)
    table.scale(1.0, 2.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
            continue
        if col <= 1:
            cell.set_facecolor("white")
            cell.set_text_props(color="#111827", weight="bold" if col == 0 else "normal")
            continue
        retained_line = cell.get_text().get_text().splitlines()[0]
        retained_count = int(retained_line.split("/")[0])
        if retained_count == 3:
            face, color = "#dcfce7", "#166534"
        elif retained_count == 2:
            face, color = "#fef9c3", "#854d0e"
        else:
            face, color = "#fee2e2", "#991b1b"
        cell.set_facecolor(face)
        cell.set_text_props(color=color, weight="bold")

    ax.text(
        0.5,
        0.13,
        "Each checkpoint cell shows overlap with the 5k top-3, then the checkpoint's current top-3.",
        ha="center",
        va="center",
        fontsize=14,
        color="#475569",
        transform=ax.transAxes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    summary = summarize(rows)
    csv_path = OUTPUT_DIR / "single_client_fia_5k_top3_stability.csv"
    png_path = OUTPUT_DIR / "single_client_fia_5k_top3_stability.png"
    write_csv(summary, csv_path)
    plot(summary, png_path)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
