"""Recalculate floor-referenced LPIPS gap retention from archived measurements."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .result_validity import require_valid_no_cpa_control

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "codex/docs/M7_FOLLOWUP_ANALYSIS_PROTOCOL.json"
OUTPUT = ROOT / "results/m7_normalized_metrics"
TAG = "m7_normalized_metrics"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def average(rows: list[dict]) -> float:
    values = np.asarray([float(row["lpips"]) for row in rows], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or non-finite LPIPS input")
    return float(values.mean())


def retained_ratio(floor: float, candidate: float, reference: float) -> float:
    if not np.isfinite([floor, candidate, reference]).all() or floor <= reference:
        raise ValueError("The finite reference must be below the floor")
    return (floor - candidate) / (floor - reference)


def generate() -> dict:
    require_valid_no_cpa_control()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(PROTOCOL.with_suffix(".lock.json").read_text(encoding="utf-8"))
    if digest(PROTOCOL) != lock["sha256"]:
        raise ValueError("Follow-up protocol changed after locking")
    paths = {
        "historical": ROOT / "results/v7_m3_fallback_5batch/per_image_metrics_aggregated.csv",
        "no_cpa_4bit": ROOT / "results/m7_no_cpa_control/4bit/per_image_metrics.csv",
        "no_cpa_fp32": ROOT / "results/m7_no_cpa_control/fp32/per_image_metrics.csv",
        "prior": ROOT / "results/m7_prior_only/prior_only/per_image_metrics.csv",
        "controls": ROOT / "results/m7_recompute_controls/per_image_metrics.csv",
        "fp32_matrix": ROOT / "results/m7_recompute_controls/fp32_CPA_pairwise_lpips.npy",
        "fourbit_matrix": ROOT / "results/m7_recompute_controls/four_bit_CPA_pairwise_lpips.npy",
        "matrix_validation": ROOT / "results/m7_recompute_controls/summary.json",
    }
    historical = read_csv(paths["historical"])
    excluded = set(protocol["fp32_reference"]["exclude_global_indices"])
    pure = sorted((r for r in historical if r["method"] == "fp32" and int(r["global_index"]) not in excluded), key=lambda r: int(r["global_index"]))
    fourbit = sorted((r for r in historical if r["method"] == "naive"), key=lambda r: int(r["global_index"]))
    if len(pure) != 36 or len(fourbit) != 40:
        raise ValueError("Required pure36 / all40 sets missing")
    if any("truth_missing" in row["result_dir"] for row in pure):
        raise ValueError("Oracle supplementation entered the fp32 baseline")
    no4, nofp = read_csv(paths["no_cpa_4bit"]), read_csv(paths["no_cpa_fp32"])
    prior, controls = read_csv(paths["prior"]), read_csv(paths["controls"])
    if any(len(v) != 40 for v in (no4, nofp, prior)):
        raise ValueError("Expected all40 controls")
    floor, reference = average(no4), average(pure)
    table = []

    def add(name: str, value: float, n: int | str, scope: str, source: str):
        ratio = retained_ratio(floor, value, reference)
        table.append({"run_tag": TAG, "condition": name, "n_values": n, "lpips": value,
                      "distance_to_floor": floor - value, "retained_ratio": ratio,
                      "retained_percent": 100 * ratio, "scope": scope, "source": source})

    add("fp32 CPA (pure36)", reference, 36, "GIA B8; excludes four oracle supplements", str(paths["historical"]))
    add("4-bit CPA", average(fourbit), 40, "GIA B8 all40; different n from pure36 reference", str(paths["historical"]))
    gma = protocol["normalized_metrics"]["gma"]
    add("GMA fp32 (published)", gma["lpips"], "paper", "UIA n16 published cross-protocol reference only", gma["url"])
    add("prior-only", average(prior), 40, "GIA B8 random-target FIA", str(paths["prior"]))
    validation = json.loads(paths["matrix_validation"].read_text(encoding="utf-8"))
    for name, key, records, indices in (
        ("shuffled fp32 (pure36)", "fp32_matrix", pure, [int(r["global_index"]) for r in pure]),
        ("shuffled 4-bit", "fourbit_matrix", fourbit, list(range(40))),
    ):
        matrix = np.load(paths[key], allow_pickle=False)
        if matrix.shape != (40, 40) or not np.isfinite(matrix).all():
            raise ValueError("Invalid archived LPIPS matrix")
        expected = np.asarray([float(r["lpips"]) for r in records])
        diag = matrix[indices, indices]
        if np.max(np.abs(diag - expected)) > 1e-5:
            raise ValueError("Archived matrix diagonal does not reproduce original metrics")
        subset = matrix[np.ix_(indices, indices)]
        vals = np.concatenate([subset[np.arange(len(indices)), np.random.default_rng(seed).permutation(len(indices))]
                               for seed in protocol["shuffle"]["seeds"]])
        add(name, float(vals.astype(np.float64).mean()), len(vals), f"20 permutations of{len(indices)} existing images; fixed points retained", str(paths[key]))
    add("no-CPA 4-bit (floor)", floor, 40, "designated common floor", str(paths["no_cpa_4bit"]))
    add("no-CPA fp32", average(nofp), 40, "precision sensitivity reference", str(paths["no_cpa_fp32"]))
    for key, name in (("dataset_mean", "constant: full-val mean"), ("gray_0.5", "constant: gray0.5"), ("uniform_noise", "constant: uniform noise")):
        values = [r for r in controls if r["control"] == key]
        if len(values) != 40:
            raise ValueError(f"Expected40 {key} records")
        add(name, average(values), len(values), "GIA40 originals; empirical reference", str(paths["controls"]))
    # A matched36 sensitivity quantifies the sample-scope difference in the requested main ratio.
    four36 = [r for r in fourbit if int(r["global_index"]) not in excluded]
    floor36 = average([r for r in no4 if int(r["global_index"]) not in excluded])
    sensitivity = {"n": 36, "floor": floor36, "fp32": reference, "fourbit": average(four36),
                   "retained_ratio": retained_ratio(floor36, average(four36), reference),
                   "scope": "supplemental exactly matched36; does not replace requested main pure36 versus4bit40 display"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "normalized_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    percent = table[1]["retained_percent"]
    summary = {"schema_version": 1, "run_tag": TAG, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": str(PROTOCOL), "protocol_sha256": lock["sha256"], "code_sha256": digest(Path(__file__)),
               "floor": floor, "fp32_reference": reference, "reference_n": 36, "excluded_global_indices": sorted(excluded),
               "denominator": floor - reference, "fourbit_retained_percent": percent,
               "formula": "(floor-condition)/(floor-fp32_pure36)", "table": table, "matched36_sensitivity": sensitivity,
               "rounded_user_inputs_percent": 100 * retained_ratio(0.7538, 0.5439, 0.4465),
               "inputs": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
               "interpretation": "Floor-referenced normalized LPIPS gap, not a physical or information-theoretic fraction; primary display mixes36/40 samples; published GMA differs in protocol."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# M7 정규화 지표", "", f"run_tag=`{TAG}`; 고정 프로토콜 SHA256 `{lock['sha256']}`.", "",
             f"바닥 F={floor:.10f}, fp32 기준 P={reference:.10f}(36장), 분모 F-P={floor-reference:.10f}.",
             "fp32 기준값은 오라클 보충 4장(global22,24,32,36)을 제외해 재계산했다.", "",
             "잔존 비율=(F-L)/(F-P). 음수는 바닥보다 LPIPS가 높다는 뜻이며0으로 자르지 않는다.", "",
             "| 조건 | LPIPS | 바닥까지의 거리 F-L | 잔존 비율 |", "|---|---:|---:|---:|"]
    lines += [f"| {r['condition']} | {r['lpips']:.6f} | {r['distance_to_floor']:+.6f} | {r['retained_percent']:.2f}% |" for r in table]
    lines += ["", f"**4-bit 양자화 후에도 복원 가능한 신호의 약 {percent:.1f}%가 남는다.**",
              "여기서 '신호'는 no-CPA 대비 LPIPS 차이를 fp32=100%로 정규화한 기술 지표를 뜻한다. 실제 정보량의 비율로 해석하지 않는다.",
              "주 표는 요청한 fp32 36장과4bit40장 범위를 사용한다. 동일36장에 한정한 민감도 분석도 summary.json에 별도 보존했다.",
              f"동일36장 민감도: 잔존 비율 {100*sensitivity['retained_ratio']:.2f}%.",
              f"GMA0.598은 [CPA 논문 Appendix B Table5]({gma['url']})의 UIA n16 인용값이다. 이 행은 현재 GIA B8 실험과 조건이 달라 직접적인 방법 우열 검정에 사용하지 않는다.",
              "Shuffled는 각 조건 전체 집합의20seed 재배정이며 동일 복원 이미지를 반복 사용한다.",
              f"계산 예시(반올림 입력): (0.7538-0.5439)/(0.7538-0.4465)={summary['rounded_user_inputs_percent']:.4f}%.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"run_tag": TAG, "fourbit_retained_percent": percent, "matched36": sensitivity, "output": str(OUTPUT)}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    generate()
