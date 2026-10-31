from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render UIA/FedAvg spectrum diagnostics as PNG figures.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/m5_uia_spectrum"))
    parser.add_argument("--output", type=Path, default=Path("results/m5_uia_spectrum/uia_spectrum_summary.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary = read_csv(input_dir / "summary.csv")
    spectrum = read_csv(input_dir / "lam_spectrum.csv")
    if not summary:
        raise ValueError(f"empty or missing summary: {input_dir / 'summary.csv'}")
    if not spectrum:
        raise ValueError(f"empty or missing spectrum: {input_dir / 'lam_spectrum.csv'}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_summary(summary, spectrum, output)
    print(output)


def render_summary(summary: list[dict[str, str]], spectrum: list[dict[str, str]], output: Path) -> None:
    rows = sorted(summary, key=lambda row: (int(row["n"]), int(row["T"])))
    n_values = sorted({int(row["n"]) for row in rows})
    palette = {16: "#2563eb", 32: "#dc2626", 64: "#059669"}

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("UIA / FedAvg Spectrum Diagnostic (4-bit stochastic per-row)", fontsize=17, fontweight="bold")

    ax_beff, ax_cv, ax_sp, ax_spec = axes.ravel()
    for n in n_values:
        color = palette.get(n, None)
        group = [row for row in rows if int(row["n"]) == n]
        T = [int(row["T"]) for row in group]
        beff = [int(row["B_eff_pred"]) for row in group]
        ax_beff.plot(T, beff, marker="o", linewidth=2.2, color=color, label=f"n={n}")
        ax_beff.axhline(n, linestyle=":", linewidth=1.0, color=color, alpha=0.35)

        b_cv = [float(row["b_norm_cv"]) for row in group]
        gia_cv = [float(row["gia_a_norm_cv"]) for row in group]
        ax_cv.plot(T, b_cv, marker="o", linewidth=2.2, color=color, label=f"UIA n={n}")
        ax_cv.plot(T, gia_cv, linestyle="--", linewidth=1.6, color=color, alpha=0.65, label=f"GIA n={n}")

        sp = [float(row["spearman_final_loss_b_norm"]) for row in group]
        ax_sp.plot(T, sp, marker="o", linewidth=2.2, color=color, label=f"n={n}")

    ax_beff.set_title("Predicted B_eff(T)")
    ax_beff.set_xlabel("FedAvg local steps T")
    ax_beff.set_ylabel("predicted B_eff")
    ax_beff.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_beff.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_beff.grid(True, alpha=0.25)
    ax_beff.legend(loc="lower right")

    ax_cv.set_title("Sample Norm CV: ||b_i|| vs GIA ||a_i||")
    ax_cv.set_xlabel("FedAvg local steps T")
    ax_cv.set_ylabel("coefficient of variation")
    ax_cv.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cv.grid(True, alpha=0.25)
    ax_cv.legend(ncol=2, loc="best")

    ax_sp.set_title("Spearman(final loss, ||b_i||)")
    ax_sp.set_xlabel("FedAvg local steps T")
    ax_sp.set_ylabel("Spearman rho")
    ax_sp.set_ylim(0.85, 1.0)
    ax_sp.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_sp.grid(True, alpha=0.25)
    ax_sp.legend(loc="lower left")

    for n in n_values:
        color = palette.get(n, None)
        for row in rows:
            if int(row["n"]) != n:
                continue
            T = int(row["T"])
            spec_rows = [
                item
                for item in spectrum
                if int(item["n"]) == n and int(item["T"]) == T
            ]
            spec_rows.sort(key=lambda item: int(item["component_rank"]))
            ratios = [float(item["lam_over_mp_edge"]) for item in spec_rows]
            ranks = [int(item["component_rank"]) + 1 for item in spec_rows]
            alpha = 0.35 + 0.05 * T
            ax_spec.plot(
                ranks,
                ratios,
                marker=".",
                linewidth=1.4,
                color=color,
                alpha=min(alpha, 0.9),
                label=f"n={n}, T={T}",
            )
    c_mp = float(rows[0]["c_mp"])
    ax_spec.axhline(c_mp, color="#111827", linestyle="--", linewidth=1.2, label=f"c_mp={c_mp:g}")
    ax_spec.set_title("Debiased Spectrum: lambda_i / MP edge")
    ax_spec.set_xlabel("component rank")
    ax_spec.set_ylabel("lambda_i / MP edge")
    ax_spec.set_yscale("log")
    ax_spec.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_spec.grid(True, which="both", alpha=0.25)
    ax_spec.legend(ncol=2, loc="upper right", fontsize=8)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
