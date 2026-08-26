#!/usr/bin/env python3
"""Show top-4/top-5 5k FIA pruning hit against the 25k FIA-loss winner."""

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
OUTPUT_DIR = ROOT / "outputs_v1" / "png_report" / "single_client_fia_5k_top4_top5_final_min"
KEEP_ORDER = ["0p8", "0p5", "0p3"]
KEEP_LABELS = {"0p8": "Keep 80%", "0p5": "Keep 50%", "0p3": "Keep 30%"}
TOP_KS = [4, 5]


def read_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sorted_checkpoint(rows: list[dict[str, str]], keep: str, iteration: int) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["keep"] == keep and int(row["iter"]) == int(iteration)
    ]
    selected.sort(key=lambda row: float(row["fia_loss"]))
    return selected


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for top_k in TOP_KS:
        for keep in KEEP_ORDER:
            top_rows = sorted_checkpoint(rows, keep, 5000)[:top_k]
            final_row = sorted_checkpoint(rows, keep, 25000)[0]
            top_members = [int(row["member"]) for row in top_rows]
            final_member = int(final_row["member"])
            summary.append(
                {
                    "top_k": top_k,
                    "keep": keep,
                    "keep_label": KEEP_LABELS[keep],
                    "top_members": ", ".join(f"m{int(row['member']):03d}" for row in top_rows),
                    "top_member_losses": "; ".join(
                        f"m{int(row['member']):03d} ({float(row['fia_loss']):.5f})"
                        for row in top_rows
                    ),
                    "final_min": f"m{final_member:03d} ({float(final_row['fia_loss']):.5f})",
                    "hit": final_member in top_members,
                    "final_member_5k_rank": [
                        int(row["member"]) for row in sorted_checkpoint(rows, keep, 5000)
                    ].index(final_member)
                    + 1,
                }
            )
    return summary


def write_csv(summary: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "top_k",
        "keep",
        "keep_label",
        "top_members",
        "top_member_losses",
        "final_min",
        "hit",
        "final_member_5k_rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def make_table(ax: plt.Axes, summary: list[dict[str, object]], top_k: int) -> None:
    ax.axis("off")
    rows = [row for row in summary if int(row["top_k"]) == top_k]
    hit_count = sum(1 for row in rows if bool(row["hit"]))
    ax.text(
        0.5,
        0.98,
        f"5k Top-{top_k} vs 25k FIA-Loss Winner   Hit: {hit_count}/{len(rows)}",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        transform=ax.transAxes,
    )
    columns = ["Keep", f"5k top-{top_k}", "25k final min", "Hit?", "final rank at 5k"]
    cell_text = [
        [
            str(row["keep_label"]),
            str(row["top_members"]),
            str(row["final_min"]),
            "YES" if bool(row["hit"]) else "NO",
            f"#{int(row['final_member_5k_rank'])}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.43, 0.18, 0.12, 0.15],
        bbox=[0.015, 0.06, 0.97, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
            continue
        if col == 3:
            hit = bool(rows[row - 1]["hit"])
            cell.set_facecolor("#dcfce7" if hit else "#fee2e2")
            cell.set_text_props(weight="bold", color="#166534" if hit else "#991b1b")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#111827")


def plot(summary: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Early FIA Pruning Check: Top-4 and Top-5",
        fontsize=27,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "Does the 25k FIA-loss winner appear in the 5k loss top-k?  Single-client CoRA / ImageNet / VGG16 / n=24 / R=8",
        ha="center",
        fontsize=13.5,
        color="#475569",
    )
    for ax, top_k in zip(axes.reshape(-1), TOP_KS):
        make_table(ax, summary, top_k)
    fig.tight_layout(rect=(0.025, 0.03, 0.985, 0.91), h_pad=1.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    summary = summarize(rows)
    csv_path = OUTPUT_DIR / "single_client_fia_5k_top4_top5_final_min_compare.csv"
    png_path = OUTPUT_DIR / "single_client_fia_5k_top4_top5_final_min_compare.png"
    write_csv(summary, csv_path)
    plot(summary, png_path)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
