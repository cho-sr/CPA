"""Refresh the requested three documents from completed M7 artifacts.

Safe as a post-run hook: reads scientific outputs; writes only the consolidated
review, bounded current-result sections in spec/status, and its own manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .result_validity import require_valid_no_cpa_control

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "codex/docs"
RESULTS = ROOT / "results"
BEGIN, END = "<!-- M7_FOLLOWUP_RESULTS_BEGIN -->", "<!-- M7_FOLLOWUP_RESULTS_END -->"
PAPER = "https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.pdf#page=11"
INPUTS: dict[str, str] = {}


def read_json(path: Path) -> dict:
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def replace_block(path: Path, block: str, anchor: str) -> None:
    original = path.read_text(encoding="utf-8")
    replacement = f"{BEGIN}\n{block.rstrip()}\n{END}\n\n"
    if BEGIN in original:
        if original.count(BEGIN) != 1 or original.count(END) != 1:
            raise ValueError(f"Ambiguous update markers in {path}")
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", lambda _: replacement,
                      original, count=1, flags=re.S)
    else:
        if original.count(anchor) != 1:
            raise ValueError(f"Missing unique insertion anchor in {path}")
        text = original.replace(anchor, replacement + anchor, 1)
    path.write_text(text, encoding="utf-8")


def generate() -> dict:
    require_valid_no_cpa_control()
    protocol = DOCS / "M7_FOLLOWUP_ANALYSIS_PROTOCOL.json"
    lock = read_json(protocol.with_suffix(".lock.json"))
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != lock["sha256"]:
        raise ValueError("Protocol hash mismatch")
    normalized = read_json(RESULTS / "m7_normalized_metrics/summary.json")
    shuffle = read_json(RESULTS / "m7_shuffle_control/summary.json")
    distribution = read_json(RESULTS / "m7_perimage_distribution/summary.json")
    audit = read_json(RESULTS / "m7_consistency_audit/summary.json")
    for result in (normalized, shuffle, distribution, audit):
        if result["protocol_sha256"] != lock["sha256"]:
            raise ValueError("Analysis protocol mismatch")
    successes = read_csv(RESULTS / "m7_perimage_distribution/success_counts.csv")
    shuffle_table = read_csv(RESULTS / "m7_shuffle_control/summary.csv")
    fp = read_json(RESULTS / "m6_uia_fp32_B16_fia/summary.json")
    four_dir = RESULTS / "m7_uia_4bit_B16"
    four = read_json(four_dir / "summary.json") if (four_dir / "summary.json").exists() else None
    state_path = four_dir / "execution_status.json"
    state = read_json(state_path) if state_path.exists() else {}
    completed = four is not None and four.get("status") == "complete"
    if completed:
        if four.get("n_images") != 80 or four.get("oracle_fills") != 0:
            raise ValueError("UIA4bit completion requires all80 images with no oracle fills")
        for filename, expected in four["output_hashes"].items():
            if hashlib.sha256((four_dir / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"UIA4bit completed artifact changed: {filename}")
    four_status = "완료" if completed else ("실행 중" if state.get("state", state.get("status")) == "running" else "준비 중")
    ratio = normalized["fourbit_retained_percent"]
    threshold = float(distribution["threshold"])
    t1 = next(r for r in audit["beff_recount"] if r["subset"] == "all granularities; T=1")
    lines = ["# 2026-09-09 실험 결과 정리", "",
             f"**TASK 1·2·3·5 완료, TASK 4 UIA 4-bit {four_status}.** 추가 모델 학습 없이 기존 관측·복원·로그를 사용한다.",
             "M7 후속 분석 규칙은 결과를 새로 집계하기 전에 스펙과 별도 JSON에 기록했다. 과거 결과는 이미 알려진 상태였으므로 후속 분석의 사전 고정이며, 최초 실험의 사전 등록으로 소급하지 않는다.",
             f"고정 시각: `{lock['locked_at_utc']}`; [프로토콜](M7_FOLLOWUP_ANALYSIS_PROTOCOL.json) SHA256 `{lock['sha256']}`.", "",
             "**확정 기준값과 통제군**", "",
             "UIA fp32 n=16, T=10, 전체80장: **LPIPS0.4546 / PSNR22.406 / SSIM0.3427**, 임베딩 |cos|>0.9 **79/80**, 오라클 보충 없음.",
             f"CPA 논문 Table5 CPA+FIA 구간은 batch16~128에서 **0.410~0.448**이다. **0.4546은 상단보다0.0066 높은 구간 경계**로 기술한다. 서로 다른 batch 크기의 구간 비교이며 구간 내부 또는 동일 조건 재현 성공을 뜻하지 않는다. [원논문 Table5]({PAPER})",
             "**fp32 기준값은 오라클 보충4장(global22,24,32,36)을 제외해36장으로 재계산한0.4465로 교체했다.** fp32 n=36 항목은 종료한다.",
             "no-CPA 바닥은 fp32 **0.7568**,4-bit **0.7538**로 평균이 압축에 거의 반응하지 않으며 이 점은 통제군 타당성을 지지한다. 다만 분포의 표준편차는 각각0.1862/0.1090이고 아래 공통 임계 통과 수도10/40 대2/40이다. 평균의 유사성을 전체 분포의 동등성으로 확대하지 않는다.", "",
             "**TASK 1 — no-CPA 기준 정규화**", "",
             "`run_tag=m7_normalized_metrics`", "",
             "바닥 F=4-bit no-CPA 평균, fp32 기준 P=오라클 제외36장 평균. 거리=F-L, 잔존비율=(F-L)/(F-P). 계산에는 반올림 전 수치를 사용한다.", "",
             "| 조건 | LPIPS | 바닥까지의 거리 F-L | 잔존 비율 |", "|---|---:|---:|---:|"]
    for row in normalized["table"]:
        lines.append(f"| {row['condition']} | {row['lpips']:.6f} | {row['distance_to_floor']:+.6f} | {row['retained_percent']:.2f}% |")
    lines += ["", f"**4-bit 양자화 후에도 복원 가능한 신호의 약 {ratio:.1f}%가 남는다.** 여기서 신호는 no-CPA 대비 LPIPS 차이를 fp32=100%로 정규화한 기술 지표이며 실제 정보량의 비율을 뜻하지 않는다.",
              f"주 표는 요청한 fp32 36장과4-bit40장 범위다. 동일36장에 한정한 보조 분석은 **{100*normalized['matched36_sensitivity']['retained_ratio']:.2f}%**다. 음수는 바닥보다 LPIPS가 높다는 뜻으로0으로 자르지 않는다.",
              f"GMA fp32 **0.598**은 [논문 Table5 n16의 인용값]({PAPER})이며 현재 GIA B8의 동일 조건 비교값이 아니다.",
              "[정규화 CSV](../../quantization_aware_cpa/results/m7_normalized_metrics/normalized_metrics.csv) · [계산·출처](../../quantization_aware_cpa/results/m7_normalized_metrics/report.md)", "",
              "**TASK 2 — shuffled 통제**", "", "`run_tag=m7_shuffle_control`", "",
              "NumPy seed0~19로 각 조건의 전체 원본 집합을 무작위 순열로 재배정했다. fp32는36장,4-bit는40장,UIA fp32는80장이다. 원래 짝이 유지되는 fixed point도 남기며 재매칭하지 않는다.",
              "no-CPA/prior-only는 소스 복구 없이 FIA를 수행하는 경우를, shuffled는 복원이 있어도 대응 원본을 틀리게 지정하는 경우를 평가한다. 서로 다른 실패 모드다.", "",
              "| 조건 | 원래 LPIPS | shuffled 평균 | 20seed 평균의 SD | 5/95 분위 |", "|---|---:|---:|---:|---:|"]
    for row in shuffle_table:
        # Field names are validated against the analysis artifact below.
        lines.append(f"| {row['condition']} | {float(row['matched_lpips_mean']):.6f} | {float(row['seed_means_mean']):.6f} | {float(row['seed_means_std']):.6f} | {float(row['seed_means_q05']):.6f} / {float(row['seed_means_q95']):.6f} |")
    lines += ["", "4-bit는0.5439에서0.7989로 상승한다. 올바른 짝의 이점이 크므로 결과를 이미지 prior만으로 설명하기 어렵다. Shuffled 평균이 no-CPA 평균과 정확히 같을 필요는 없으며, 이 비교는 대응 관계에 대한 민감도를 보여준다.",
              "표의 SD는20개 seed 평균의 표본 SD(ddof=1)이며, 개별 pair 분포의 평균·SD·5/95분위도 별도 CSV에 있다. 총720/800/1600 pair는 기존 이미지를 반복 사용하므로 독립적인 새 이미지 수가 아니다.",
              "[요약 CSV](../../quantization_aware_cpa/results/m7_shuffle_control/summary.csv) · [20seed별 값](../../quantization_aware_cpa/results/m7_shuffle_control/per_seed_summary.csv) · [모든 pair](../../quantization_aware_cpa/results/m7_shuffle_control/per_image_metrics.csv)", "",
              "**TASK 3 — per-image 분포와 성공 개수**", "", "`run_tag=m7_perimage_distribution`", "",
              f"고정 임계: **LPIPS < {threshold:.10f}**.4-bit no-CPA40장 분포의 하위5% 분위수(선형 보간)이며, candidate를 세기 전에 숫자를 저장했다. 모든 조건에 같은 임계를 적용한다.",
              "이는 경험적 대조군 대비 기술적 성공 개수이며 개별 이미지의 유의성 판정은 아니다. UIA에는 GIA 기준을 적용했으므로 탐색적 비교다. 기존 M7의 leave-batch-out 두 대조군 rank-test 개수는 원시 보고서에 보존하고, 아래 새 개수와 혼용하지 않는다.", "",
              "| 조건 | LPIPS 평균 | 성공 개수 |", "|---|---:|---:|"]
    for row in successes:
        lines.append(f"| {row['label']} | {float(row['mean']):.6f} | {row['success_count']}/{row['success_denominator']} |")
    lines += ["", "Shuffled의 임계 통과 pair는 모두 원래 짝이 남은 fixed point다. 원래 짝이 아닌 shuffled pair에서는 세 조건 모두 통과0건이었다.",
              "![공통 축 per-image LPIPS](../../quantization_aware_cpa/results/m7_perimage_distribution/lpips_histogram_overlay.png)",
              "![4-bit 분포 형태 민감도](../../quantization_aware_cpa/results/m7_perimage_distribution/gia_4bit_modality_sensitivity.png)",
              "4-bit40장의 KDE는 사전에 고정한 Scott bandwidth0.75/1/1.25배 모두 약0.45와0.78 부근의 두 봉우리를 보인다. 이 표본에서는 복원 품질이 고르지 않다는 서술을 지지한다. 하지만 성공은 **29/40으로 다수**이므로 **‘소수의 선명한 복원+다수의 실패’ 또는 ‘소수만 복원’이라고 쓰지 않는다.**40장 표본의 시각적 이봉성을 모집단 혼합분포에 대한 확정적 검정으로 표현하지 않는다.",
              "[고정 임계](../../quantization_aware_cpa/results/m7_perimage_distribution/threshold.json) · [성공 개수](../../quantization_aware_cpa/results/m7_perimage_distribution/success_counts.csv) · [이미지별 값](../../quantization_aware_cpa/results/m7_perimage_distribution/per_image_metrics.csv)", "",
              "**TASK 4 — UIA 4-bit n=16**", "", f"`run_tag=m7_uia_4bit_B16`; 상태: **{four_status}**.", "",
              "기존 UIA fp32의 실제 `(W_initial-W_final)/lr` 업데이트5개와 동일80장을 재사용한다. VGG16 conv동결, n16/T10,4-bit stochastic per-row, full16 Stage1+FIA다. 새 학습,Stage2,`--refine-A`,오라클 보충은 없다. FIA25000iteration,TV0.1,Adam lr0.1,cosine,호출별seed42를 fp32와 동일하게 고정한다.",
              "B_eff 예측은 기존 debiased MP threshold(c_mp=3)로 별도 진단하며, 실제 공격은 full16 구성요소를 유지한다. 복구가 약한 이미지도 전체80장 평가에 포함한다.",
              "기존 stochastic 구현은 `Y+Uniform(-delta/2,delta/2)` 형태의 subtractive-dither 관측 모델이다. 이번 실행은 이를 유지하며, 정수4-bit 전송 코덱 자체를 시험한 결과로 확대하지 않는다."]
    if completed:
        lines += ["", "[UIA3행 비교표와 결과](../../quantization_aware_cpa/results/m7_uia_4bit_B16/report.md)",
                  "복원 평균·이미지별 지표·cos분포·예측/실측 B_eff·runtime과 seed42 무작위 그리드는 위 보고서에 기록돼 있다."]
        comparison_path = four_dir / "three_condition_comparison.csv"
        if comparison_path.exists():
            comparison = read_csv(comparison_path)
            lines += ["", "| 조건 | n / T | 이미지 수 | LPIPS | PSNR | SSIM | cos>0.9 |", "|---|---:|---:|---:|---:|---:|---:|"]
            for row in comparison:
                lines.append(f"| {row['condition']} | {row['n']} / {row['T']} | {row['n_images']} | {float(row['lpips_mean']):.6f} | {float(row['psnr_mean']):.3f} | {float(row['ssim_mean']):.4f} | {row['embedding_cos09']}/{row['n_images']} |")
            lines += ["", "[UIA fp32 / UIA4bit / GIA4bit 비교 CSV](../../quantization_aware_cpa/results/m7_uia_4bit_B16/three_condition_comparison.csv)"]
        metrics = four["metrics"]
        lines += [f"예측 B_eff 합계 **{metrics['predicted_B_eff_total']}/80**, 실측 임베딩 복구 **{metrics['measured_B_eff_cos09_total']}/80**. 두 값은 별도로 측정한다.",
                  f"동일 80장의 평균 LPIPS 변화(4-bit minus fp32): **{four['paired_lpips_delta_4bit_minus_fp32']:+.6f}**.",
                  f"Runtime: `{json.dumps(four['runtime_sec'], ensure_ascii=False)}` seconds.",
                  "![UIA 무작위 12장: not cherry-picked](../../quantization_aware_cpa/results/m7_uia_4bit_B16/reconstruction_grid_not_cherry_picked.png)"]
    else:
        lines += ["", "완료 시 평균LPIPS/PSNR/SSIM,이미지별 지표,임베딩cos분포,예측/실측B_eff,runtime,3행 비교표와 seed42무작위12장 ‘not cherry-picked’ 그리드를 자동 저장하고 이 문서를 갱신한다. 완료 전 성능값을 채우지 않는다."]
    lines += ["", "UIA fp32와 UIA4bit는 동일80장의 비교다. GIA4bit는B8/40장이므로3행 표에서 설정이 다른 문맥 참고행으로 표시한다.", "",
              "**TASK 5 — 수치 정합성**", "", "`run_tag=m7_consistency_audit`", "",
              "fp32의 loss–LPIPS Spearman은 원본 로그에서 오라클을 제외한36장으로 확정한 **+0.031**(p=0.8581)이다.4-bit40장은 **−0.591**(p≈0.0000602)이다. 이전 hybrid 상관값은 현재 본문 수치에서 제외했으며 원시 범위의 출처만 감사CSV에 남겼다.",
              "이 p값은 이미지 단위 기술 통계이며 서로 다른 표본 범위의 두 상관계수 차이를 검정한 값은 아니다.", "",
              f"**예측식은96조건 중81에서 정확했다(MAE0.219). FedSGD에 해당하는T=1부분격자에서는{t1['exact_count']}/{t1['condition_count']}가 정확했다.**", "",
              "| 범위 | 정확한 조건 수 | MAE |", "|---|---:|---:|"]
    for row in audit["beff_recount"]:
        lines.append(f"| {row['subset']} | {row['exact_count']}/{row['condition_count']} | {row['mean_abs_error']:.6f} |")
    lines += ["", "T=1의39/48은 전체96조건에서 분리한 값이다.30/36은 그 안의 per-row core부분격자이며 별도로 표시한다. 저장된 eigenvalue/MP비율과 source cosine으로96개 조건의 예측·실측 개수를 독립적으로 다시 세어 원본CSV와 일치함을 확인했다.",
              "![T별 정사각형 산점도](../../quantization_aware_cpa/results/m7_consistency_audit/predicted_vs_measured_beff_square.png)",
              "n32의21.2는M6 round0의5개배치 예측 `[24,25,22,21,14]` 평균이다. n32/4bit/T1/victimseed0/batch0/round0/per-row 조건을 맞춘 예측값은M5와M6 모두24다. 실측24와 예측평균21.2를 비교했던 범위 차이를 해소했다.",
              "다만 quantization seed는M5 spectrum0,M5 validation294949,M6 rounds293536으로 달라 동일 양자화 입력의 수치 재현을 주장하지 않는다. M6 round spectrum은T1이고 별도 저장T10 UIA dump와 구분한다.",
              "[감사 보고서](../../quantization_aware_cpa/results/m7_consistency_audit/report.md) · [조건별 비교](../../quantization_aware_cpa/results/m7_consistency_audit/n32_axis_comparison.csv)", "",
              "**완료 기록과 남은 범위**", "",
              "기존 UIA fp32 `execution_status.json`·`summary.json`은 complete지만 `queue_status.json`은 failed다. 종료코드가 빈 값으로 처리된 오류로 추정되며,80행CSV·5개FIA완료로그·저장출력 해시로 실제 산출 완료를 확인했다. 실제 OS종료코드는 사후 입증할 수 없다. 원본 큐 상태와 로그는 보존했고 STATUS.md에도 기록했다.",
              "기존 M7의4-bit CPA 대 no-CPA·prior-only의5배치 검정은IUT p=0.03125로 보존한다. 새 정규화 비율과5%분위 성공 개수는 그 검정의 대체 p값이 아니다.",
              "선택A(TV/iteration)·B(granularity오차)는TASK4완료 후의 선택 작업으로 남긴다.79/80 임베딩복구만으로 논문구간과의0.0066격차 원인이FIA라고 확정할 수 없다. GMA4bit,seen-batch round의존성,n방향 곡선,극단적non-IID,(B,bit)격자는 접는다.", ""]
    review = DOCS / "EXPERIMENT_RESULTS_20260909.md"
    review.write_text("\n".join(lines), encoding="utf-8")
    status_block = f"""## M7 follow-up result summary

- Tasks 1/2/3/5: **COMPLETE**; Task 4 `m7_uia_4bit_B16`: **{'COMPLETE' if completed else four_status}**.
- Normalized4bit retained ratio **{ratio:.4f}%**, floor0.75382263, pure36 fp32 reference0.44647295. Matched36 sensitivity76.6870% is separate.
- Shuffled means: fp32pure36 **0.808424**,4bit40 **0.798902**,UIAfp3280 **0.816526**; all20 seeds and per-pair distributions saved.
- Fixed descriptive cutoff **LPIPS < {threshold:.10f}**: fp3236/36,4bit29/40,UIAfp3279/80.4bit shows two empirical modes but success is the majority; do not claim minority-only recovery.
- Canonical fp32 Spearman **+0.031**,4bit **-0.591**. Prediction **81/96(MAE0.219)**; full T1 **39/48**,core per-row T1 **30/36**.
- n32:21.2 is the five-batch prediction mean; matched batch0 predicts24 across runs. Quantizer seeds differ, so this is not identical-input reproduction.
- Pure36 fp32 reporting resolves the prior n=36 issue. Old UIA queue failed/complete discrepancy remains documented below.
- Reproduction and deliverable links: [consolidated review](docs/EXPERIMENT_RESULTS_20260909.md).
"""
    replace_block(ROOT.parent / "codex/STATUS.md", status_block, "## Confirmed follow-up scope")
    spec_block = f"""### Section0.8 execution results

Tasks1/2/3/5 are complete; Task4 status: {four_status}. Frozen rules above were
not changed after calculation.4bit normalized LPIPS retention is{ratio:.4f}%.
The common cutoff is{threshold:.10f}; counts are fp3236/36,4bit29/40,UIA79/80.
The two empirical4bit modes do not support minority-only recovery (29/40pass).
Canonical fp32 rho=+0.031 versus4bit-0.591;81/96 exact(MAE0.219),allT1=39/48.
Per-row-onlyT1=30/36 is a separately labelled subset. The n32 difference is
five-batch mean21.2 versus matched batch0 prediction24; quantizer seeds differ.
See [the current review](EXPERIMENT_RESULTS_20260909.md) for exact scope and sources.
"""
    replace_block(DOCS / "QACPA_SPEC.md", spec_block, "## 0.7 historical authorized successor")
    status_path = ROOT.parent / "codex/STATUS.md"
    status_text = status_path.read_text(encoding="utf-8")
    status_text = re.sub(r"Active milestone:.*", f"Active milestone: **M7 Tasks1/2/3/5 complete; UIA4bit {four_status}**", status_text, count=1)
    status_text = status_text.replace("- Tasks1/2/3/5: preparing recomputation. Task4: preparing saved-update-only runner; launches after shuffle metrics free the GPU. Optional(A)/(B) deferred until Task4 complete.",
                                     "- Tasks1/2/3/5 completed; Task4 uses saved updates only. Current Task4 state is in the result summary above. Optional(A)/(B) remain deferred.")
    status_path.write_text(status_text, encoding="utf-8")
    output = RESULTS / "m7_followup_review"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": lock["sha256"],
                "task4_complete": completed, "task4_display_status": four_status, "inputs": INPUTS,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (review, status_path, DOCS / "QACPA_SPEC.md")}}
    (output / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documentation_refreshed": True, "task4_complete": completed}, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    generate()
