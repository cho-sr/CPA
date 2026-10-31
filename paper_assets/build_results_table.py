"""Typeset saved metrics with dataset rows and grouped compression columns."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def values(row, keys):
    # Match the example's compact PSNR display; keep SSIM/LPIPS at four decimals.
    return [f"{float(row[key]):.{2 if i == 0 else 4}f}" for i, key in enumerate(keys)]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def block(title, levels, measured, randomized_label):
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lll" + "r" * len(levels) + "@{}}",
        r"\toprule",
        rf"\multicolumn{{{3 + len(levels)}}}{{l}}{{\textit{{{title}}}}}" + r"\\",
        "Dataset & Method & Metric & " + " & ".join(levels) + r"\\",
        r"\midrule",
    ]
    for dataset in ("CIFAR-100", "ImageNet"):
        for method_index, method in enumerate(("Naive", randomized_label)):
            for metric_index, metric in enumerate(("PSNR", "SSIM", "LPIPS")):
                label = rf"\multirow{{6}}{{*}}{{{dataset}}}" if method_index == metric_index == 0 else ""
                method_label = rf"\multirow{{3}}{{*}}{{{method}}}" if metric_index == 0 else ""
                cells = ["---"] * len(levels) if dataset == "CIFAR-100" else measured[method_index][metric_index::3]
                lines.append(" & ".join([label, method_label, metric, *cells]) + r"\\")
        if dataset == "CIFAR-100":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    return "\n".join(lines)


def build_table():
    quantization = [[], []]
    for bits in (8, 7, 6, 5, 4):
        folder = (
            "signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918"
            if bits == 8 else
            f"signed_int{bits}_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918"
        )
        report = json.loads((ROOT / "quantization_aware_cpa/results_private" / folder / "comparison.json").read_text(encoding="utf-8"))
        means = report["mean_metrics"]
        rank1 = next(key for key in means if key.startswith("rank1_"))
        for index, key in enumerate(("center_25000", rank1)):
            quantization[index].extend(values(means[key], ("mean_psnr_db", "mean_ssim", "mean_lpips")))

    results = ROOT / "quantization_aware_cpa/results"
    sweep = {int(row["keep_percent"]): row for row in read_csv(results / "cpa_fia_topk30_20_10_same80_B16/comparison_metrics.csv")}
    random_rows = read_csv(results / "cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/comparison_metrics.csv")
    random10 = next(row for row in random_rows if row["method"].startswith("Uniform"))
    sparsification = [[], []]
    for keep in (50, 30, 20, 10):
        sparsification[0].extend(values(sweep[keep], ("psnr_mean", "ssim_mean", "lpips_mean")))
        sparsification[1].extend(values(random10, ("psnr_mean", "ssim_mean", "lpips_mean")) if keep == 10 else ["---"] * 3)

    table = r"""% Generated from existing metrics by build_results_table.py; no evaluation.
\begin{table}[!t]
\centering
\caption{Naive versus randomized decoding (PSNR$\uparrow$, SSIM$\uparrow$,
LPIPS$\downarrow$). Dashes denote unavailable results.
Protocols differ between panels (Section~\ref{sec:sparse}).}
\label{tab:compression}
\label{tab:bits}
\small
\setlength{\tabcolsep}{0.8pt}
\renewcommand{\arraystretch}{1.0}
"""
    table += block("(a) Quantization", [f"{b}-bit" for b in (8, 7, 6, 5, 4)], quantization, r"\cora{}")
    table += "\n\\par\\smallskip\n"
    table += block("(b) Sparsification: retained fraction", [rf"{k}\%" for k in (50, 30, 20, 10)], sparsification, "Random fill")
    table += "\n\\end{table}\n"
    (OUT / "combined_results.tex").write_text(table, encoding="utf-8")
    return table


if __name__ == "__main__":
    build_table()
    print("Updated combined_results.tex: one-column table, metrics on rows.")
