#!/usr/bin/env python3
"""Plot single-client CoRA Direct-FIA loss curves for all members."""

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
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs_v1"
    / "png_report"
    / "single_client_fia_loss_members"
)

KEEP_SPECS = [
    ("0p8", "Keep 80%", 7),
    ("0p5", "Keep 50%", 1),
    ("0p3", "Keep 30%", 7),
]
TARGET_ITERS = [5000, 10000, 15000, 20000, 25000]


def member_dir(keep: str, member: int) -> Path:
    return AUTHOR_ROOT / (
        f"cora_single_client_cpa_keep{keep}_n24_steps1_trials1_"
        f"R8_seed1234_m{member:03d}_uia"
    )


def read_fia_rows(run_dir: Path) -> dict[int, tuple[float, bool]]:
    path = run_dir / "losses.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    logged: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") != "fi":
                continue
            iteration = int(row["iter"])
            plot_iter = 25000 if iteration == 24999 else iteration
            logged[plot_iter] = float(row["loss"])
    if not logged:
        raise ValueError(f"{path} has no FIA loss rows")
    logged_iters = sorted(logged)
    rows: dict[int, tuple[float, bool]] = {}
    for target in TARGET_ITERS:
        if target in logged:
            rows[target] = (logged[target], False)
            continue
        lower = [iteration for iteration in logged_iters if iteration < target]
        upper = [iteration for iteration in logged_iters if iteration > target]
        if not lower or not upper:
            raise ValueError(f"{path} cannot interpolate FIA loss at iter={target}")
        lo = lower[-1]
        hi = upper[0]
        ratio = (target - lo) / float(hi - lo)
        value = logged[lo] + (logged[hi] - logged[lo]) * ratio
        rows[target] = (float(value), True)
    return rows


def collect_rows() -> list[dict[str, object]]:
    collected: list[dict[str, object]] = []
    for keep, _label, _selected_member in KEEP_SPECS:
        for member in range(8):
            run_dir = member_dir(keep, member)
            losses = read_fia_rows(run_dir)
            for iteration in TARGET_ITERS:
                loss, interpolated = losses[iteration]
                collected.append(
                    {
                        "keep": keep,
                        "keep_percent": int(float(keep.replace("p", ".")) * 100),
                        "member": member,
                        "iter": iteration,
                        "fia_loss": loss,
                        "interpolated": interpolated,
                        "run_dir": str(run_dir),
                    }
                )
    return collected


def write_csv_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["keep", "keep_percent", "member", "iter", "fia_loss", "interpolated", "run_dir"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def loss_series(rows: list[dict[str, object]], keep: str, member: int) -> tuple[list[int], list[float]]:
    selected = [
        row
        for row in rows
        if row["keep"] == keep and int(row["member"]) == int(member)
    ]
    selected.sort(key=lambda row: int(row["iter"]))
    return [int(row["iter"]) for row in selected], [float(row["fia_loss"]) for row in selected]


def best_member_at(rows: list[dict[str, object]], keep: str, iteration: int) -> tuple[int, float]:
    candidates = [
        row
        for row in rows
        if row["keep"] == keep and int(row["iter"]) == int(iteration)
    ]
    if not candidates:
        raise ValueError(f"no rows for keep={keep} iter={iteration}")
    best = min(candidates, key=lambda row: float(row["fia_loss"]))
    return int(best["member"]), float(best["fia_loss"])


def loss_at(rows: list[dict[str, object]], keep: str, member: int, iteration: int) -> float:
    matches = [
        row
        for row in rows
        if row["keep"] == keep
        and int(row["member"]) == int(member)
        and int(row["iter"]) == int(iteration)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for keep={keep} member={member} iter={iteration}")
    return float(matches[0]["fia_loss"])


def final_loss_table(rows: list[dict[str, object]], keep: str, selected_member: int) -> str:
    best_member, _best_loss = best_member_at(rows, keep, 25000)
    lines = ["25k loss"]
    for member in range(8):
        marker = "*" if member == selected_member else " "
        suffix = " min" if member == best_member else ""
        lines.append(f"{marker}m{member:03d}: {loss_at(rows, keep, member, 25000):.4f}{suffix}")
    return "\n".join(lines)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=160, sharey=True)
    fig.patch.set_facecolor("white")
    colors = plt.cm.tab10.colors

    for ax, (keep, title, selected_member) in zip(axes.reshape(-1), KEEP_SPECS):
        selected_xy: tuple[int, float] | None = None
        for member in range(8):
            xs, ys = loss_series(rows, keep, member)
            is_selected = member == selected_member
            if is_selected:
                selected_xy = (xs[-1], ys[-1])
            ax.plot(
                xs,
                ys,
                marker="*" if is_selected else "o",
                linewidth=4.0 if is_selected else 2.2,
                markersize=11 if is_selected else 5,
                color="#111827" if is_selected else colors[member % len(colors)],
                label=f"m{member:03d}",
                alpha=1.0 if is_selected else 0.78,
                zorder=5 if is_selected else 2,
            )
        for idx, iteration in enumerate(TARGET_ITERS):
            best_member, best_loss = best_member_at(rows, keep, iteration)
            ax.scatter(
                [iteration],
                [best_loss],
                marker="v",
                s=90,
                facecolor="#facc15",
                edgecolor="#111827",
                linewidth=1.0,
                zorder=8,
            )
            ax.annotate(
                f"min m{best_member:03d}",
                xy=(iteration, best_loss),
                xytext=(0, -23 if idx % 2 else 16),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.90,
                },
                zorder=9,
            )
        if selected_xy is None:
            raise RuntimeError(f"selected member m{selected_member:03d} was not plotted")
        final_x, final_y = selected_xy
        ax.text(
            0.975,
            0.965,
            final_loss_table(rows, keep, selected_member),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.2,
            family="monospace",
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.92,
            },
            zorder=10,
        )
        ax.annotate(
            f"selected m{selected_member:03d}",
            xy=(final_x, final_y),
            xytext=(-90, 25),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1.4},
            fontsize=12,
            color="#0f172a",
            ha="right",
        )
        ax.set_title(f"{title}\nselected m{selected_member:03d}")
        ax.set_xlabel("FIA iteration")
        ax.set_xticks(TARGET_ITERS)
        ax.set_xticklabels(["5k", "10k", "15k", "20k", "25k"])
        ax.set_yscale("log")
        ax.grid(True, axis="both", color="#dbe3ef", linewidth=0.9, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
    axes[0].set_ylabel("Direct-FIA loss")
    axes[0].set_ylim(0.025, 0.12)

    member_handles = [
        Line2D(
            [0],
            [0],
            color=colors[member % len(colors)],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=f"m{member:03d}",
        )
        for member in range(8)
    ]
    selected_handle = Line2D(
        [0],
        [0],
        color="#111827",
        marker="*",
        linewidth=4.0,
        markersize=11,
        label="final selected member",
    )
    best_handle = Line2D(
        [0],
        [0],
        color="#111827",
        marker="v",
        linewidth=0,
        markerfacecolor="#facc15",
        markeredgecolor="#111827",
        markersize=8,
        label="lowest loss at checkpoint",
    )
    fig.legend(
        member_handles + [selected_handle, best_handle],
        [handle.get_label() for handle in member_handles] + [
            selected_handle.get_label(),
            best_handle.get_label(),
        ],
        loc="upper center",
        ncol=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        "Single-Client CoRA Direct-FIA Loss by Member",
        fontsize=25,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.025,
        "ImageNet / VGG16 / n=24 / R=8. Log-scale y-axis; yellow markers show lowest-loss member at each checkpoint.",
        ha="center",
        fontsize=13,
        color="#475569",
    )
    fig.tight_layout(rect=(0.035, 0.06, 0.99, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    csv_path = OUTPUT_DIR / "single_client_fia_loss_members_5k.csv"
    png_path = OUTPUT_DIR / "single_client_fia_loss_members_5k.png"
    write_csv_rows(rows, csv_path)
    plot(rows, png_path)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
