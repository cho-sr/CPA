# 2026-09-09 실험 결과 정리

**정정: no-CPA 대조 실험은 변인 통제 실패로 무효이며, 결과와 이에 의존한 결론을 사용하지 않는다.** 사용자의 최신 판단을 반영해 이전 통제군 타당성 주장을 철회한다. 구체적인 교란 변인은 이 문서에서 추가로 추정하지 않는다.

UIA 4-bit와 독립 분석의 계산 완료 기록은 유지한다. 계산이 완료됐다는 사실이 무효화된 대조 실험의 유효성을 뜻하지 않는다. 원시 파일은 감사 이력으로 보존하며, 이번 정정으로 새 실험을 실행하지 않는다.

방법 명칭 보충: 이 문서의 기존 CPA 결과는 자체 이식 경로이며, 공통 Stage1에
LS-A 계산이 포함되지만 해당 A는 기본 CPA+FIA의 복원 보정에 사용되지 않는다.
원본 CPA/FIA 클래스를 직접 호출하고 LS-A 자체도 계산하지 않는 새 FedAvg naive
대조군은 [M8 별도 프로토콜](FEDAVG_NAIVE_CPA_PROTOCOL.md)로 진행한다. 기존 수치는
보존하며, 새 결과가 나오기 전에 원본 구현과의 동등성을 주장하지 않는다.
M7 후속 분석 규칙은 결과를 새로 집계하기 전에 스펙과 별도 JSON에 기록했다. 과거 결과는 이미 알려진 상태였으므로 후속 분석의 사전 고정이며, 최초 실험의 사전 등록으로 소급하지 않는다. 이 기록은 이력용이며, no-CPA 및 그 파생 결론에 관한 기존 규칙·주장은 위 정정으로 대체된다.
고정 시각: `2026-09-09T07:51:25.473636+00:00`; [프로토콜](M7_FOLLOWUP_ANALYSIS_PROTOCOL.json) SHA256 `24a25980e58356591943559e29474bb777494f426b0834ca6d68f76aa2daae67`.

**유지하는 직접 복원 지표**

UIA fp32 n=16, T=10, 전체80장: **LPIPS0.4546 / PSNR22.406 / SSIM0.3427**, 임베딩 |cos|>0.9 **79/80**, 오라클 보충 없음.
CPA 논문 Table5 CPA+FIA 구간은 batch16~128에서 **0.410~0.448**이다. **0.4546은 상단보다0.0066 높은 구간 경계**로 기술한다. 서로 다른 batch 크기의 구간 비교이며 구간 내부 또는 동일 조건 재현 성공을 뜻하지 않는다. [원논문 Table5](https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.pdf#page=11)
**fp32 기준값은 오라클 보충4장(global22,24,32,36)을 제외해36장으로 재계산한0.4465로 교체했다.** fp32 n=36 항목은 종료한다.
no-CPA의 평균·분포·정밀도별 비교는 유효한 대조 결과로 사용하지 않는다. 기존에 평균의 유사성으로 통제군 타당성을 지지한다고 쓴 해석도 철회한다.

**TASK 1 — no-CPA 기준 정규화 철회**

`run_tag=m7_normalized_metrics`

**무효화된 no-CPA 평균을 기준으로 한 거리·정규화 비율 전체를 철회한다.** 기존 신호 잔존율 68.3%와 동일 36장 보조 분석의 76.69%를 포함해, 이 기준을 사용한 모든 조건의 비율과 비교 순위를 논문·보고서 결론에 사용하지 않는다. 계산식이 일관됐더라도 기준 실험의 변인 통제 실패를 보완하지 못한다. 관련 표와 해석은 현재 결과에서 제거했으며, 기존 산출물은 철회된 계산의 이력이다.

**TASK 2 — shuffled 통제**

`run_tag=m7_shuffle_control`

NumPy seed0~19로 각 조건의 전체 원본 집합을 무작위 순열로 재배정했다. fp32는36장,4-bit는40장,UIA fp32는80장이다. 원래 짝이 유지되는 fixed point도 남기며 재매칭하지 않는다.
Shuffled는 저장된 복원과 원본의 대응을 바꾼 원시 거리 분석이다. 아래 LPIPS와 순열별 분산은 no-CPA 결과를 계산에 사용하지 않으므로 유지한다.

| 조건 | 원래 LPIPS | shuffled 평균 | 20seed 평균의 SD | 5/95 분위 |
|---|---:|---:|---:|---:|
| gia_fp32 | 0.446473 | 0.808424 | 0.011869 | 0.792538 / 0.824281 |
| gia_4bit | 0.543940 | 0.798902 | 0.012408 | 0.781252 / 0.810462 |
| uia_fp32 | 0.454597 | 0.816526 | 0.005440 | 0.807075 / 0.822743 |

4-bit는 올바른 대응의 평균 0.5439에서 shuffled 평균 0.7989로 상승한다. 이는 대응 관계에 대한 민감도를 보여주는 기술적 결과다. 이 수치만으로 CPA 단계의 독립적인 기여도를 확정하거나 무효화된 no-CPA 대조를 대체하지 않는다.
표의 SD는20개 seed 평균의 표본 SD(ddof=1)이며, 개별 pair 분포의 평균·SD·5/95분위도 별도 CSV에 있다. 총720/800/1600 pair는 기존 이미지를 반복 사용하므로 독립적인 새 이미지 수가 아니다.
[요약 CSV](../../quantization_aware_cpa/results/m7_shuffle_control/summary.csv) · [20seed별 값](../../quantization_aware_cpa/results/m7_shuffle_control/per_seed_summary.csv) · [모든 pair](../../quantization_aware_cpa/results/m7_shuffle_control/per_image_metrics.csv)

**no-CPA와 독립적인 원시 참고 지표**

| 조건 | 이미지 수 | LPIPS 평균 |
|---|---:|---:|
| Prior-only | 40 | 0.773775 |
| Constant: full-val mean | 40 | 0.805476 |
| Constant: gray 0.5 | 40 | 0.829199 |
| Constant: uniform noise | 40 | 0.856640 |

이 평균값 자체는 no-CPA에서 계산한 값이 아니므로 원시 기술 통계로 보존한다. 이를 no-CPA 대신 CPA 기여도의 검증 근거로 승격하지 않으며, no-CPA에 의존했던 정규화와 성공 판정은 아래와 같이 철회한다.

**TASK 3 — no-CPA 기반 임계값·성공 판정 철회**

`run_tag=m7_perimage_distribution`

**no-CPA 분포에서 정한 LPIPS 임계값 0.5704289049와 이에 따른 모든 성공 개수·성공률을 철회한다.** CPA, UIA, shuffled, prior-only, 상수 이미지에 적용했던 판정 모두 해당한다. Shuffled fixed point의 임계 통과 여부와 이 판정에 근거한 ‘다수/소수 복원’ 주장도 사용하지 않는다.

기존 M7의 leave-batch-out no-CPA·prior-only 이중 참조 rank-test 성공 개수 역시 no-CPA에 의존하므로 철회한다. 성공표와 관련 그림은 현재 본문에서 제거했다. 원시 이미지별 LPIPS·PSNR·SSIM 값과 no-CPA를 사용하지 않은 분석은 이 정정만으로 무효화되지 않는다.

아래에 유지하는 **임베딩 절대 cosine > 0.9 개수는 별도의 직접 측정값**이다. 철회한 LPIPS 임계 성공 개수와 수치가 우연히 같아도 서로 대체하거나 혼용하지 않는다.

**TASK 4 — UIA 4-bit n=16**

`run_tag=m7_uia_4bit_B16`; 상태: **완료**.

기존 UIA fp32의 실제 `(W_initial-W_final)/lr` 업데이트5개와 동일80장을 재사용한다. VGG16 conv동결, n16/T10,4-bit stochastic per-row, full16 Stage1+FIA다. 새 학습,Stage2,`--refine-A`,오라클 보충은 없다. FIA25000iteration,TV0.1,Adam lr0.1,cosine,호출별seed42를 fp32와 동일하게 고정한다.
B_eff 예측은 기존 debiased MP threshold(c_mp=3)로 별도 진단하며, 실제 공격은 full16 구성요소를 유지한다. 복구가 약한 이미지도 전체80장 평가에 포함한다.
기존 stochastic 구현은 `Y+Uniform(-delta/2,delta/2)` 형태의 subtractive-dither 관측 모델이다. 이번 실행은 이를 유지하며, 정수4-bit 전송 코덱 자체를 시험한 결과로 확대하지 않는다.

[UIA3행 비교표와 결과](../../quantization_aware_cpa/results/m7_uia_4bit_B16/report.md)
복원 평균·이미지별 지표·cos분포·예측/실측 B_eff·runtime과 seed42 무작위 그리드는 위 보고서에 기록돼 있다.

| 조건 | n / T | 이미지 수 | LPIPS | PSNR | SSIM | 임베딩 절대 cos>0.9 |
|---|---:|---:|---:|---:|---:|---:|
| UIA fp32 | 16 / 10 | 80 | 0.454597 | 22.406 | 0.3427 | 79/80 |
| UIA 4-bit | 16 / 10 | 80 | 0.558584 | 21.926 | 0.2976 | 53/80 |
| GIA 4-bit | 8 / 1 | 40 | 0.543940 | 21.536 | 0.3070 | 29/40 |

[UIA fp32 / UIA4bit / GIA4bit 비교 CSV](../../quantization_aware_cpa/results/m7_uia_4bit_B16/three_condition_comparison.csv)
예측 B_eff 합계 **55/80**, 실측 임베딩 복구 **53/80**. 두 값은 별도로 측정한다.
동일 80장의 평균 LPIPS 변화(4-bit minus fp32): **+0.103986**.
Runtime: `{"update_collection": 0.0, "quantization": 7.919079100014642, "MP_diagnostic": 2.7267095999559388, "stage1": 199.70086310012266, "fia": 9562.973591000191}` seconds.
![UIA 무작위 12장: not cherry-picked](../../quantization_aware_cpa/results/m7_uia_4bit_B16/reconstruction_grid_not_cherry_picked.png)

UIA fp32와 UIA4bit는 동일80장의 비교다. GIA4bit는B8/40장이므로3행 표에서 설정이 다른 문맥 참고행으로 표시한다.

동일 80장의 직접 CPA+FIA 지표만 사용하는 [`client_update_fp32_vs_4bit` 그래프 묶음](../../quantization_aware_cpa/results/client_update_fp32_vs_4bit/)은 no-CPA나 그 파생 임계값을 사용하지 않으므로 이번 정정의 무효화 대상이 아니다.

**TASK 5 — 수치 정합성**

`run_tag=m7_consistency_audit`

fp32의 loss–LPIPS Spearman은 원본 로그에서 오라클을 제외한36장으로 확정한 **+0.031**(p=0.8581)이다.4-bit40장은 **−0.591**(p≈0.0000602)이다. 이전 hybrid 상관값은 현재 본문 수치에서 제외했으며 원시 범위의 출처만 감사CSV에 남겼다.
이 p값은 이미지 단위 기술 통계이며 서로 다른 표본 범위의 두 상관계수 차이를 검정한 값은 아니다.

**예측식은96조건 중81에서 정확했다(MAE0.219). FedSGD에 해당하는T=1부분격자에서는39/48가 정확했다.**

| 범위 | 정확한 조건 수 | MAE |
|---|---:|---:|
| all granularities; all T | 81/96 | 0.218750 |
| all granularities; T=1 | 39/48 | 0.229167 |
| all granularities; T=10 | 42/48 | 0.208333 |
| core per-row; all T | 62/72 | 0.180556 |
| core per-row; T=1 | 30/36 | 0.194444 |
| core per-row; T=10 | 32/36 | 0.166667 |

T=1의39/48은 전체96조건에서 분리한 값이다.30/36은 그 안의 per-row core부분격자이며 별도로 표시한다. 저장된 eigenvalue/MP비율과 source cosine으로96개 조건의 예측·실측 개수를 독립적으로 다시 세어 원본CSV와 일치함을 확인했다.
![T별 정사각형 산점도](../../quantization_aware_cpa/results/m7_consistency_audit/predicted_vs_measured_beff_square.png)
n32의21.2는M6 round0의5개배치 예측 `[24,25,22,21,14]` 평균이다. n32/4bit/T1/victimseed0/batch0/round0/per-row 조건을 맞춘 예측값은M5와M6 모두24다. 실측24와 예측평균21.2를 비교했던 범위 차이를 해소했다.
다만 quantization seed는M5 spectrum0,M5 validation294949,M6 rounds293536으로 달라 동일 양자화 입력의 수치 재현을 주장하지 않는다. M6 round spectrum은T1이고 별도 저장T10 UIA dump와 구분한다.
[감사 보고서](../../quantization_aware_cpa/results/m7_consistency_audit/report.md) · [조건별 비교](../../quantization_aware_cpa/results/m7_consistency_audit/n32_axis_comparison.csv)

**완료 기록과 남은 범위**

기존 UIA fp32 `execution_status.json`·`summary.json`은 complete지만 `queue_status.json`은 failed다. 종료코드가 빈 값으로 처리된 오류로 추정되며,80행CSV·5개FIA완료로그·저장출력 해시로 실제 산출 완료를 확인했다. 실제 OS종료코드는 사후 입증할 수 없다. 원본 큐 상태와 로그는 보존했고 STATUS.md에도 기록했다.
**기존 5배치 IUT p=0.03125에 따른 ‘4-bit CPA가 no-CPA와 prior-only 모두보다 우수하여 CPA 기여도를 검증했다’는 결론을 철회한다.** no-CPA 대조가 무효이므로 이 결합 검정과 이를 인용한 기여도 주장은 사용할 수 없다. 원시 검정 출력은 감사 이력으로만 보존한다. 정규화 비율·LPIPS 임계 성공 개수·과거 rank-test 성공 개수도 복원 성능의 유효한 근거로 인용하지 않는다.
선택A(TV/iteration)·B(granularity오차)는TASK4완료 후의 선택 작업으로 남긴다.79/80 임베딩복구만으로 논문구간과의0.0066격차 원인이FIA라고 확정할 수 없다. GMA4bit,seen-batch round의존성,n방향 곡선,극단적non-IID,(B,bit)격자는 접는다.
