#!/usr/bin/env python3
"""Compare CoRA-selected members with minimum final embedding-loss members."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = (
    REPO_ROOT
    / "sparsified_update_cpa"
    / "outputs_v1"
    / "author_runs"
    / "imagenet"
    / "vgg16"
    / "attack"
    / "cp_direct"
    / "nodef"
)
RANK_ROOT = (
    Path(__file__).resolve().parent
    / "outputs_v1"
    / "imagenet"
    / "vgg16"
    / "single_client_n24_steps1_trials1_5947f6b7723b4a18"
    / "global_topk"
)
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs_v1"
    / "png_report"
    / "single_client_embedding_loss_selection_compare"
)

KEEP_SPECS = [
    ("0p8", "0p8", "Keep 80%"),
    ("0p5", "0p5", "Keep 50%"),
    ("0p3", "0p3", "Keep 30%"),
]
MEMBERS = list(range(8))
MIN_TOL = 1e-7


def member_run_dir(keep_tag: str, member: int) -> Path:
    return AUTHOR_ROOT / (
        f"cora_single_client_cpa_keep{keep_tag}_n24_steps1_trials1_"
        f"R8_seed1234_m{member:03d}_uia"
    )


def final_embedding_loss(run_dir: Path) -> tuple[int, float]:
    path = run_dir / "losses.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    last_iter: int | None = None
    last_loss: float | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") != "gi":
                continue
            last_iter = int(row["iter"])
            last_loss = float(row["loss"])
    if last_iter is None or last_loss is None:
        raise ValueError(f"{path} has no embedding/gi loss rows")
    return last_iter, last_loss


def read_consistency_ranking(keep_dir_tag: str) -> tuple[int, dict[int, dict[str, float]]]:
    path = RANK_ROOT / f"keep_{keep_dir_tag}" / "R8_seed1234_model_pred" / "consistency_member_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    ranking: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for rank, row in enumerate(csv.DictReader(handle), start=1):
            member = int(row["member"])
            relative_l2 = float(row["mean_relative_l2_to_observed_client_update"])
            ranking[member] = {
                "consistency_rank": float(rank),
                "consistency_relative_l2": relative_l2,
            }
            if rank == 1:
                selected_member = member
    return selected_member, ranking


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keep_tag, keep_dir_tag, keep_label in KEEP_SPECS:
        selected_member, ranking = read_consistency_ranking(keep_dir_tag)
        loss_by_member: dict[int, tuple[int, float]] = {}
        for member in MEMBERS:
            loss_by_member[member] = final_embedding_loss(member_run_dir(keep_tag, member))
        min_loss = min(loss for _iteration, loss in loss_by_member.values())
        for member in MEMBERS:
            final_iter, loss = loss_by_member[member]
            rank_info = ranking[member]
            rows.append(
                {
                    "keep": keep_tag,
                    "keep_percent": int(float(keep_tag.replace("p", ".")) * 100),
                    "keep_label": keep_label,
                    "member": member,
                    "final_embedding_iter": final_iter,
                    "final_embedding_loss": loss,
                    "embedding_loss_delta_from_min": loss - min_loss,
                    "is_embedding_loss_min": abs(loss - min_loss) <= MIN_TOL,
                    "cora_selected_member": selected_member,
                    "is_cora_selected": member == selected_member,
                    "consistency_rank": int(rank_info["consistency_rank"]),
                    "consistency_relative_l2": rank_info["consistency_relative_l2"],
                    "run_dir": str(member_run_dir(keep_tag, member)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "keep",
        "keep_percent",
        "keep_label",
        "member",
        "final_embedding_iter",
        "final_embedding_loss",
        "embedding_loss_delta_from_min",
        "is_embedding_loss_min",
        "cora_selected_member",
        "is_cora_selected",
        "consistency_rank",
        "consistency_relative_l2",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def keep_rows(rows: list[dict[str, object]], keep: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row["keep"] == keep]
    selected.sort(key=lambda row: int(row["member"]))
    return selected


def plot(rows: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=160, sharey=False)
    fig.patch.set_facecolor("white")
    bar_color = "#5b8def"

    for ax, (keep_tag, _keep_dir_tag, keep_label) in zip(axes.reshape(-1), KEEP_SPECS):
        subset = keep_rows(rows, keep_tag)
        members = [int(row["member"]) for row in subset]
        deltas = [float(row["embedding_loss_delta_from_min"]) for row in subset]
        losses = [float(row["final_embedding_loss"]) for row in subset]
        selected_member = int(subset[0]["cora_selected_member"])
        min_members = [int(row["member"]) for row in subset if row["is_embedding_loss_min"] == "True" or row["is_embedding_loss_min"] is True]
        min_loss = min(losses)

        ax.bar(members, deltas, color=bar_color, alpha=0.72, edgecolor="#1e3a8a", linewidth=0.8)
        for member, delta, loss in zip(members, deltas, losses):
            if member == selected_member:
                ax.scatter(
                    [member],
                    [max(delta, 0.0)],
                    marker="*",
                    s=230,
                    color="#111827",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )
            if member in min_members:
                ax.scatter(
                    [member],
                    [max(delta, 0.0)],
                    marker="v",
                    s=130,
                    facecolor="#facc15",
                    edgecolor="#111827",
                    linewidth=1.0,
                    zorder=6,
                )
            ax.text(
                member,
                max(delta, 0.0),
                f"{loss:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
                color="#334155",
            )

        if len(min_members) == len(MEMBERS):
            min_text = "embedding min: all tied"
        else:
            min_text = "embedding min: " + ", ".join(f"m{member:03d}" for member in min_members)
        selected_loss = losses[members.index(selected_member)]
        ax.text(
            0.04,
            0.96,
            f"CoRA selected: m{selected_member:03d}\n"
            f"{min_text}\n"
            f"selected emb loss: {selected_loss:.6f}\n"
            f"min emb loss: {min_loss:.6f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.94,
            },
        )
        ax.set_title(keep_label)
        ax.set_xlabel("member")
        ax.set_xticks(members)
        ax.set_xticklabels([f"m{member:03d}" for member in members], rotation=35)
        ax.set_ylabel("final embedding loss - min")
        ax.grid(True, axis="y", color="#dbe3ef", linewidth=0.9, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")

    legend_handles = [
        Line2D([0], [0], color=bar_color, linewidth=8, alpha=0.72, label="embedding loss delta"),
        Line2D([0], [0], marker="*", color="#111827", linewidth=0, markersize=12, label="CoRA final selected"),
        Line2D(
            [0],
            [0],
            marker="v",
            color="#111827",
            markerfacecolor="#facc15",
            linewidth=0,
            markersize=9,
            label="minimum embedding loss",
        ),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("CoRA Final Selection vs Minimum Embedding Loss", fontsize=25, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.025,
        "ImageNet / VGG16 / single-client n=24 / R=8. Embedding loss is the final CPA GI loss; CoRA selection is compressed-update consistency ranking.",
        ha="center",
        fontsize=12.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0.035, 0.06, 0.99, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    csv_path = OUTPUT_DIR / "single_client_embedding_loss_selection_compare.csv"
    png_path = OUTPUT_DIR / "single_client_embedding_loss_selection_compare.png"
    write_csv(rows, csv_path)
    plot(rows, png_path)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
