"""Render the completed M7 measurements as a Korean summary PNG (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .result_validity import require_valid_no_cpa_control


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "m7_no_cpa_control"
INK = "#17263b"
MUTED = "#546579"
COLORS = {
    "fp32_cpa": "#235bc4", "4bit_cpa": "#dc542b",
    "fp32_no_cpa": "#1693a5", "4bit_no_cpa": "#e5a032",
    "prior_only": "#8156bc", "fp32_shuffled": "#607788",
    "4bit_shuffled": "#9f7850", "dataset_mean": "#318567",
    "gray": "#4c5361", "noise": "#bd5898",
}
LABELS = {
    "fp32_cpa": "fp32 CPA*", "4bit_cpa": "4-bit CPA + dequant",
    "fp32_no_cpa": "fp32 no-CPA", "4bit_no_cpa": "4-bit no-CPA",
    "prior_only": "prior-only FIA", "fp32_shuffled": "fp32 shuffled",
    "4bit_shuffled": "4-bit shuffled", "dataset_mean": "데이터셋 평균 이미지",
    "gray": "회색 (0.5)", "noise": "무작위 이미지 노이즈",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_data():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if summary["report_status"] != "complete":
        raise ValueError("Only a completed M7 report can be plotted")
    groups = {key: [] for key in COLORS}
    for row in read_csv(OUT / "image_success.csv"):
        groups[row["method"]].append(float(row["lpips"]))
    groups["prior_only"] = [float(row["lpips"]) for row in read_csv(
        RESULTS / "m7_prior_only" / "prior_only" / "per_image_metrics.csv")]
    mapping = {"fp32_CPA": "fp32_shuffled", "four_bit_CPA": "4bit_shuffled",
               "dataset_mean": "dataset_mean", "gray_0.5": "gray", "uniform_noise": "noise"}
    for row in read_csv(RESULTS / "m7_recompute_controls" / "per_image_metrics.csv"):
        key = row["method"] if row["control"] == "shuffled" else row["control"]
        groups[mapping[key]].append(float(row["lpips"]))
    arrays = {key: np.asarray(values, dtype=float) for key, values in groups.items()}
    for key, values in arrays.items():
        expected = 800 if "shuffled" in key else 40
        if len(values) != expected or not np.isfinite(values).all():
            raise ValueError(f"Incomplete distribution: {key}")
    for row in summary["grid_2x2"]:
        key = f"{row['precision']}_{'cpa' if row['cpa'] == 'present' else 'no_cpa'}"
        if not np.isclose(arrays[key].mean(), row["lpips_mean"], rtol=0, atol=1e-12):
            raise ValueError(f"Summary/CSV disagree: {key}")
    return summary, arrays


def main():
    require_valid_no_cpa_control()
    summary, groups = load_data()
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if not font_path.is_file():
        raise FileNotFoundError("Korean font required: " + str(font_path))
    font_manager.fontManager.addfont(str(font_path))
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "font.size": 10, "savefig.facecolor": "#f3f6fa"})
    fig = plt.figure(figsize=(18, 12), dpi=180, facecolor="#f3f6fa")

    def text(x, y, value, size=11, color=INK, weight="normal", **kwargs):
        return fig.text(x, y, value, fontsize=size, color=color, weight=weight,
                        va="top", **kwargs)

    def card(x, y, w, h, color="white", edge="#dce4ee"):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.009",
                                     transform=fig.transFigure, facecolor=color,
                                     edgecolor=edge, linewidth=0.7, zorder=0))

    text(.042, .967, "M7  |  4-bit에서 CPA의 복원 기여", 29, weight="bold")
    text(.044, .922, "완료 결과  ·  ImageNet 동일 원본 40장  ·  B=8 × 5배치  ·  victim seed=0  ·  LPIPS는 낮을수록 좋음", 11.5, MUTED)
    primary = summary["group_inference"]["4bit"]
    card(.04, .833, .92, .064, color="#e7f3ee", edge="#c3ded1")
    text(.055, .885, "4-bit CPA는 no-CPA와 prior-only 모두보다 낮은 평균 LPIPS", 18, "#165a43", "bold")
    text(.056, .854, "사전 등록한 집단 비교 기준 통과  ·  이 실험 조건에서 CPA의 복원 기여를 지지", 10.5, "#35715b")
    text(.942, .880, f"p = {primary['iut_p']:.5f}", 20, "#165a43", "bold", ha="right")

    card(.04, .510, .445, .297)
    text(.057, .786, "01   2×2 대조 결과", 14, weight="bold")
    text(.057, .757, "평균 LPIPS / 사전 등록 기준 충족 수†", 10.5, MUTED)
    cells = {(row["precision"], row["cpa"]): row for row in summary["grid_2x2"]}
    table_rows = []
    for precision in ("fp32", "4bit"):
        values = ["fp32" if precision == "fp32" else "4-bit"]
        for kind in ("present", "absent"):
            row = cells[(precision, kind)]
            values.append(f"{row['lpips_mean']:.4f}\n{row['success_count']}/{row['success_denominator']}장")
        table_rows.append(values)
    ax_table = fig.add_axes([.055, .592, .414, .145])
    ax_table.axis("off")
    table = ax_table.table(cellText=table_rows, colLabels=["관측", "CPA 있음*", "CPA 없음\n(no-unmixing)"],
                           colWidths=[.18, .40, .42], cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e7ef")
        cell.set_linewidth(.7)
        cell.set_facecolor("#edf2f8" if r == 0 else ("#fff4ed" if r == 2 else "#ffffff"))
        cell.get_text().set_color(INK)
        if r == 0 or c == 0:
            cell.get_text().set_weight("bold")
        if r == 2 and c == 1:
            cell.get_text().set_color(COLORS["4bit_cpa"])
            cell.get_text().set_weight("bold")
    text(.057, .576, "* fp32: CPA 복원 36장 + 정답 embedding FIA 4장의 혼합값.", 9.2, MUTED)
    text(.057, .553, "4-bit: CPA (미수정) + naive dequant, 40장 모두 CPA 복원.", 9.2, MUTED)
    text(.057, .530, "† 정밀도별 기준이 달라 2/40 대 20/40로 우열을 비교할 수 없음.", 9.2, MUTED)

    card(.505, .510, .455, .297)
    text(.523, .786, "02   조건별 평균 LPIPS", 14, weight="bold")
    ax_bar = fig.add_axes([.665, .551, .255, .212])
    keys = list(groups)
    means = [float(groups[key].mean()) for key in keys]
    ys = np.arange(len(keys))
    ax_bar.barh(ys, means, height=.66, color=[COLORS[key] for key in keys], alpha=.93)
    ax_bar.set_yticks(ys, [LABELS[key] for key in keys], fontsize=9.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax_bar.tick_params(axis="both", length=0, labelsize=9)
    ax_bar.grid(axis="x", color="#e7edf3", linewidth=.6)
    ax_bar.set_axisbelow(True)
    for y, value in zip(ys, means):
        ax_bar.text(value + .015, y, f"{value:.4f}", va="center", fontsize=9, color=INK)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    card(.04, .116, .57, .365)
    text(.057, .460, "03   이미지별 LPIPS 분포 · 모든 조건을 동일 축에 표시", 14, weight="bold")
    ax_hist = fig.add_axes([.082, .222, .503, .198])
    combined = np.concatenate(list(groups.values()))
    bins = np.linspace(min(0, float(combined.min())), max(1, float(combined.max()) * 1.03), 41)
    for key, values in groups.items():
        style = "--" if "shuffled" in key else (":" if key in {"dataset_mean", "gray", "noise"} else "-")
        ax_hist.hist(values, bins=bins, density=True, histtype="step", color=COLORS[key],
                     linewidth=1.9 if key.endswith("cpa") else 1.45, linestyle=style,
                     label=f"{LABELS[key]} ({len(values)})")
    ax_hist.set_xlim(bins[0], bins[-1])
    ax_hist.set_xlabel("이미지별 LPIPS (낮을수록 좋음)", fontsize=10)
    ax_hist.set_ylabel("조건별 밀도", fontsize=10)
    ax_hist.grid(axis="y", color="#e7edf3", linewidth=.6)
    for spine in ("top", "right"):
        ax_hist.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax_hist.spines[spine].set_color("#c9d3df")
    ax_hist.tick_params(labelsize=9)
    handles, labels = ax_hist.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.061, .180), ncol=3,
               fontsize=8.5, frameon=False, handlelength=2.5, columnspacing=1.7, labelspacing=.7)

    card(.63, .116, .33, .365)
    text(.648, .460, "04   4-bit 주 결과와 판정 기준", 14, weight="bold")
    text(.648, .425, "CPA가 낮춘 평균 LPIPS", 11, MUTED)
    text(.648, .401, f"no-CPA 대비  {primary['no_cpa']['mean_effect']:.4f}", 17, COLORS["4bit_cpa"], "bold")
    text(.648, .369, f"prior-only 대비  {primary['prior_only']['mean_effect']:.4f}", 17, COLORS["4bit_cpa"], "bold")
    text(.648, .328, "5개 배치 모두 두 대조군보다 개선.", 10.5)
    text(.648, .305, "배치 평균 차이의 단측 exact sign-flip 검정:\n32개 부호 조합, 각 p=0.03125; IUT p=0.03125.\n배치 차이의 부호 대칭을 가정하며 최소 p=1/32.", 9.4, MUTED, linespacing=1.6)
    text(.648, .239, "기준 충족 수†의 의미", 11, weight="bold")
    text(.648, .214, "본인 배치를 제외한 no-CPA 32장과 prior 32장의\n두 최솟값보다 LPIPS가 엄격히 낮은 이미지 수.\n사전 등록 규칙으로 대조군에서 임계를 산출한 뒤 집계.\n개별 이미지의 통계적 유의성을 뜻하지 않음.", 9.3, MUTED, linespacing=1.6)

    sensitivity = summary["group_inference"]["fp32_same36_sensitivity"]["mean_lpips_by_condition"]
    text(.044, .089, "고정 조건  |  no-CPA는 관측 행을 정규화해 FIA에 직접 입력 · ICA/unmixing 생략  |  FIA 25,000회 · TV=0.1 · Adam lr=0.1", 9.4, MUTED)
    text(.044, .066, f"fp32 보조 확인  |  정답 embedding 4장 제외, 동일 36장: CPA {sensitivity['cpa']:.4f} / no-CPA {sensitivity['no_cpa']:.4f} / prior-only {sensitivity['prior_only']:.4f}", 9.4, MUTED)
    text(.044, .043, "분포 해석  |  shuffled는 40장 × 20 seeds의 종속된 800쌍 · 상수/노이즈는 참조값이며 LPIPS의 수학적 하한이 아님", 9.4, MUTED)
    text(.044, .020, f"run_tag: m7_no_cpa_control / m7_recompute_controls / m7_prior_only  ·  prereg SHA-256: {summary['preregistration_sha256'][:16]}…", 8, "#7b899b")

    path = OUT / "m7_results_summary.png"
    fig.savefig(path, dpi=180, metadata={"Title": "M7 CPA contribution controls", "Description": "Rendered from completed M7 CSV/JSON measurements; no experiment rerun."})
    plt.close(fig)
    sources = [OUT / "summary.json", OUT / "image_success.csv", OUT / "thresholds.json",
               RESULTS / "m7_prior_only" / "prior_only" / "per_image_metrics.csv",
               RESULTS / "m7_recompute_controls" / "per_image_metrics.csv"]
    manifest = {"output": str(path), "dimensions": [3240, 2160],
                "preregistration_sha256": summary["preregistration_sha256"],
                "means": dict(zip(keys, means)),
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT / "m7_results_summary.sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
