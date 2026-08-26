#!/usr/bin/env python3
"""Summarize whether the 25k FIA-loss winner appears in the 5k top-3."""

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
OUTPUT_DIR = (
    ROOT
    / "outputs_v1"
    / "png_report"
    / "single_client_fia_5k_top3_final_min"
)
KEEP_LABELS = {
    "0p8": "Keep 80%",
    "0p5": "Keep 50%",
    "0p3": "Keep 30%",
}


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_for(rows: list[dict[str, str]], keep: str, iteration: int) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["keep"] == keep and int(row["iter"]) == int(iteration)
    ]
    selected.sort(key=lambda row: float(row["fia_loss"]))
    return selected


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for keep in ["0p8", "0p5", "0p3"]:
        top3_5k = rows_for(rows, keep, 5000)[:3]
        final_min = rows_for(rows, keep, 25000)[0]
        top3_members = {int(row["member"]) for row in top3_5k}
        final_member = int(final_min["member"])
        summary.append(
            {
                "keep": keep,
                "keep_label": KEEP_LABELS[keep],
                "top3_5k": "\n".join(
                    f"m{int(row['member']):03d} ({float(row['fia_loss']):.5f})"
                    for row in top3_5k
                ),
                "final_min_25k": f"m{final_member:03d} ({float(final_min['fia_loss']):.5f})",
                "hit": final_member in top3_members,
            }
        )
    return summary


def write_csv(summary: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["keep", "keep_label", "top3_5k", "final_min_25k", "hit"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot(summary: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    hit_count = sum(1 for row in summary if bool(row["hit"]))
    ax.text(
        0.5,
        0.93,
        "Does the 25k FIA-Loss Winner Appear in the 5k Top-3?",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.865,
        f"Single-client CoRA / ImageNet / VGG16 / n=24 / R=8    Hit: {hit_count}/{len(summary)}",
        ha="center",
        va="center",
        fontsize=16,
        color="#475569",
        transform=ax.transAxes,
    )

    columns = ["Keep", "5k FIA-loss top-3", "25k final min", "In 5k top-3?"]
    cell_text = [
        [
            str(row["keep_label"]),
            str(row["top3_5k"]),
            str(row["final_min_25k"]),
            "YES" if bool(row["hit"]) else "NO",
        ]
        for row in summary
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="left",
        colLoc="center",
        colWidths=[0.14, 0.49, 0.20, 0.17],
        bbox=[0.07, 0.25, 0.86, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13.5)
    table.scale(1.0, 2.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a", ha="center")
            continue
        if col == 3:
            hit = bool(summary[row - 1]["hit"])
            cell.set_facecolor("#dcfce7" if hit else "#fee2e2")
            cell.set_text_props(weight="bold", color="#166534" if hit else "#991b1b", ha="center")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#111827")

    ax.text(
        0.5,
        0.20,
        "This checks only FIA loss. It is not the CoRA consistency-ranking selection.",
        ha="center",
        va="center",
        fontsize=15,
        color="#475569",
        transform=ax.transAxes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    summary = summarize(rows)
    csv_path = OUTPUT_DIR / "single_client_fia_5k_top3_final_min_compare.csv"
    png_path = OUTPUT_DIR / "single_client_fia_5k_top3_final_min_compare.png"
    write_csv(summary, csv_path)
    plot(summary, png_path)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
