# 논문 수정 방향과 결과 근거

수정본: [test_revised.tex](test_revised.tex). 사용자가 붙인 초안을 바탕으로 다시 구성했다. 기존 `test.tex`, 첨부 원문, 연구 코드와 실험 결과는 변경하지 않았다. 신규 실험을 실행하지 않았다.

## 1. 논문이 답하려는 질문

“압축된 업데이트의 불확실한 값을 랜덤하게 채워 복원을 시도했을 때, 몇 bit와 어느 희소화 수준까지 복원 품질이 유지되고 어디서 무너지는가?”

기존의 ‘한 번만 운 좋으면 된다’와 새로운 CoRA의 우월성 주장을 중심에서 뺐다. 대신 실제 구현된 랜덤 방법 → 압축 강도별 측정 → 적용 범위와 한계 순으로 재구성했다. 추가할 내용은 LaTeX 마지막의 T1–T8 TODO로 모았다.

현재 쓸 수 있는 결론:

- 동일 16장의 공개 점수 rank-1 결과에서 8→4bit로 낮아질수록 PSNR과 LPIPS가 악화된다. 랜덤 복원이 center 복원보다 항상 낫지는 않다.
- 동일 80장의 **zero-fill** 실험에서는 유지율 50→30→20→10%로 낮아질수록 PSNR/SSIM/LPIPS가 모두 악화된다.
- 같은80 프로토콜의 **random-fill은 유지율 10%에서만** 비교됐다. LPIPS는 조금 낮아졌지만 PSNR/SSIM은 조금 악화됐다. 계산량도 훨씬 많다.
- “8bit 성공, 4bit 실패”, “유지율 30% 성공, 20% 실패” 같은 분류는 아직 확정할 수 없다. 사전 정의된 이미지 복원 성공 기준과 통일된 평가가 필요하다. 평균 LPIPS가 어떤 값보다 낮다는 것과 개별 이미지 성공률은 다르다.

표에서 keep는 유지율, remove는 제거율이다. 예를 들어 **keep 10% = 제거율/희소화율 90%**다.

## 2. 초안에서 바로잡은 내용

| 초안의 주장 또는 표현 | 수정 이유 및 처리 |
|---|---|
| “One lucky draw is enough” 및 56회면 90% 성공 | 측정된 독립 시행 성공 확률이 없다. 제목과 핵심 주장 및 수치 삭제. |
| `p=0.04`가 랜덤 시도당 성공 확률 | 과거 값은 LPIPS≤0.5인 이미지 **1/24**의 비율이다. source matching 문제까지 있어 시행 성공 확률로 사용할 수 없다. |
| sender의 압축이 stochastic | sender는 deterministic quantization/top-k, randomness는 attacker의 completion에 있다. |
| calibrated per-tensor model | 최근 실제 방법은 bounded uniform sampling이다. 학습 prior는 TODO로 이동. |
| 모든 랜덤 후보가 recompressed-update 점수로 선택됨 | bit sweep은 feature cosine distance + 0.1 TV. same80 random-fill은 재압축 업데이트 relative L2. 구분해서 서술. |
| R8 ensemble, rank selection, client 조건을 하나의 CoRA로 통합 | 과거 source 평균 방식과 최근 whole-candidate selection은 별개다. main results는 확인된 두 트랙으로 한정. |
| 4bit 작업 완료 / public gate 통과 = 복원 성공 | 완료와 gate는 이미지 품질 판정이 아니다. 성공 경계 미확정 명시. |
| ImageNet/CIFAR-100, FC2/TinyImageNet 수치 혼합 | 초안의 FP32 26.44/.794/.243 등은 현재 표와 대응 provenance를 확인하지 못해 제외. CIFAR-100 확장은 완료된 결과로 쓰지 않음. |
| 희소화 유지율 50%는 정보 절반 손실 | same80에서는 이미 50% 이상이 0이어서 top-k50이 수치적으로 lossless다. 실제 입력 audit을 본문에 반영. |
| 최종 후보 중 LPIPS가 가장 좋은 것을 대표값으로 보고 | bit sweep은 5k 때 고정된 공개 rank-1만 대표값으로 사용. sparse도 공개 점수의 최종 whole-batch 선택만 사용. |

## 3. 본문 표 1: 8–4bit, 동일 16장

각 setting에서 center와 random의 최종 FIA는 25,000회다. Random은 8개 후보의 5,000회 공개 screening에서 rank-1으로 고정된 후보다. rank-2/3의 더 좋은 LPIPS를 골라 바꾸지 않았다.

| bit | center PSNR | random PSNR | center SSIM | random SSIM | center LPIPS | random LPIPS | random member |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 12.0569 | 12.0568 | 0.2924 | 0.3070 | 0.5594 | 0.5558 | 01 |
| 7 | 11.7955 | 11.8314 | 0.2764 | 0.2827 | 0.6118 | 0.6332 | 00 |
| 6 | 11.6397 | 11.6675 | 0.2596 | 0.2690 | 0.6961 | 0.6659 | 01 |
| 5 | 11.5524 | 11.5458 | 0.2620 | 0.2558 | 0.7826 | 0.7490 | 05 |
| 4 | 11.5197 | 11.5028 | 0.2921 | 0.2576 | 0.8071 | 0.8085 | 06 |

확인한 원본 JSON (`mean_metrics`의 `center_25000`, `rank1_*_25000`):

- [8bit](quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918/comparison.json)
- [7bit](quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)
- [6bit](quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)
- [5bit](quantization_aware_cpa/results_private/signed_int5_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)
- [4bit](quantization_aware_cpa/results_private/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)

방법론 확인 코드:

- [signed_bits_randomized.py](quantization_aware_cpa/qacpa/signed_bits_randomized.py): uniform cell sampling, scale anchor, exact re-encoding, permutation/sign-only alignment.
- [signed_bits_screen3_runner.py](quantization_aware_cpa/qacpa/signed_bits_screen3_runner.py): 8개 후보, top3 continuation, 공개 rank 고정.
- [signed_int8_screen3_fia.py](quantization_aware_cpa/qacpa/signed_int8_screen3_fia.py): feature cosine + TV screening과 상태 보존.
- [signed_int8_screen3_evaluation.py](quantization_aware_cpa/qacpa/signed_int8_screen3_evaluation.py): private 평가, pixel-MSE Hungarian assignment를 세 지표에 공통 적용.

별도 GD-only 4bit 실험은 backend도 바뀌므로 ‘FastICA만 제거한 단일요인 ablation’으로 넣지 않았다.

## 4. 본문 표 2: same80 유지율 비교 및 random-fill

5개 batch ×16장, T=10, 동일 관측/모델/seed/CPA/FIA 조건의 zero-fill sweep이다. Random은 마지막 한 행뿐이다.

| 유지율 | 제거율 | 방법 | PSNR | SSIM | LPIPS |
|---:|---:|---|---:|---:|---:|
| 50% | 50% | Zero fill | 22.2878 | 0.3344 | 0.4864 |
| 30% | 70% | Zero fill | 22.1371 | 0.3141 | 0.5098 |
| 20% | 80% | Zero fill | 21.8720 | 0.2906 | 0.5536 |
| 10% | 90% | Zero fill | 21.5305 | 0.2631 | 0.6323 |
| 10% | 90% | Random fill | 21.5104 | 0.2610 | 0.6300 |

근거:

- [유지율별 comparison_metrics.csv](quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/comparison_metrics.csv)
- [유지율별 report.md](quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/report.md)
- [Top-k50 실제 입력 audit](quantization_aware_cpa/results/cpa_fia_topk50_same80_B16/sparsification_input_audit.md): nonzero fraction 38.22–42.09%, relative L2=0.
- [Top-k30/20/10 실제 입력 audit](quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/sparsification_input_audit.csv)
- [Random-fill comparison_metrics.csv](quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/comparison_metrics.csv)
- [Random-fill report.md](quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/report.md)
- [Random-fill 실행 코드](quantization_aware_cpa/qacpa/run_random_fill_same80.py), [sampling 및 scoring 코드](quantization_aware_cpa/qacpa/random_fill.py)

동일 80장의 대응 PNG:

- [유지율 50%](quantization_aware_cpa/results/cpa_fia_topk50_same80_B16/all80_reconstruction_landscape.png)
- [유지율 30%](quantization_aware_cpa/results/cpa_fia_topk30_same80_B16/all80_reconstruction_landscape.png)
- [유지율 20%](quantization_aware_cpa/results/cpa_fia_topk20_same80_B16/all80_reconstruction_landscape.png)
- [유지율 10% zero-fill](quantization_aware_cpa/results/cpa_fia_topk10_same80_B16/all80_reconstruction_landscape.png)
- [유지율 10% random-fill](quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/all80_original_vs_reconstruction_16x9.png)

Random-fill은 batch당 CPA 총500k + FIA 총160k + replay23회다. zero-fill은 CPA25k + FIA25k다. Random이 근소하게 개선한 LPIPS만 강조해 방법 자체가 우월하다고 쓰면 안 된다. quantization 표와 이 표는 이미지 집합, CPA backend, matching이 달라 직접 대조하지 않는다.

## 5. 과거 R8 희소화 sweep: 본문에서 제외한 이유

과거 위치 패턴:

`sparsified_update_cpa/outputs_v1/author_runs/imagenet/vgg16/attack/cp_direct/nodef/02_randomized_desparse_ensemble_cpa_blindagg_keep0p{8,5,3,1}_n64_steps10_trials3_fia8_R8_seed1234_uia/`

여기는 `one_sample_per_client`, client64개, trials3, 총 source192개 중 FIA 이미지24장이다. `fia8`은 trial당 평가 이미지8장이지 FIA iteration8회가 아니다. 클라이언트별 top-k completion 후 평균하고, 각 CPA source를 blind alignment/scaling 후 평균한다. 최근 same80의 단일-client B16 및 후보 선택과 다르다.

원래 `metric_summary.csv`에 저장된 수치:

| 유지율 | PSNR | SSIM | LPIPS | LPIPS≤0.5 이미지 비율 |
|---:|---:|---:|---:|---:|
| 80% | 12.3917 | 0.3362 | 0.7675 | 1/24 |
| 50% | 12.3563 | 0.3359 | 0.7695 | 1/24 |
| 30% | 12.3358 | 0.3360 | 0.7673 | 1/24 |
| 10% | 12.3640 | 0.3397 | 0.7714 | 1/24 |

그러나 기존 재평가 파일에서 매칭을 바꾸면 결과가 크게 달라진다:

- keep30 `metric_summary_oracle_embedding_dataset.csv`: LPIPS 0.523979, 10/24.
- keep10 `metric_summary_oracle_embedding.csv`: LPIPS 0.511553, 13/24.
- keep30의 `metric_summary_recomputed_dataset.csv`는 original direct-order 수치와 일치한다.
- [평가 코드](sparsified_update_cpa/evaluate_reconstructions_only.py)의 direct는 복원 row i를 원본 row i와 비교한다. oracle_embedding은 저장된 source Hungarian assignment로 대응을 정한다. source permutation이 있는 결과에서 둘은 동일한 평가가 아니다.
- keep80 manifest에는 FP32와 sparse attack-layer의 hash가 같고 relative L2=0이라고 기록돼 있다. 명목상 유지율만으로 실제 정보 손실량을 단정하면 안 된다.

따라서 이 4.17%를 랜덤 시행 성공률로 해석하거나 `1-(1-p)^K`에 대입하면 안 된다. 이론식은 고정된 사건/고정된 관측에 대한 독립·동일분포 시행이라는 조건에서만 해당 형태로 사용할 수 있으며, 현재 기록은 그 p를 측정한 것이 아니다. 과거 재평가의 더 높은 수치 역시 새 success boundary의 근거로 즉시 채택하지 않았다.

## 6. TODO 우선순위

1. **T1 필수:** 성공 정의. held-out 기준으로 threshold와 판정 단위를 정하고, 결과를 본 뒤 유리한 경계를 고르지 않는다.
2. **T2–T4 필수:** 같은 backend·이미지·공개 selector·예산·matching으로 랜덤 희소화 sweep, 독립 반복, 계산량 대조군을 갖춘다. 기존 파일만 재평가해 해결 가능한 것부터 구분한다.
3. **T5 권장:** 고정 이미지 index의 8/7/6/5/4bit 및 유지율별 비교 그림. 좋은 사례만 고르지 않고 실패 사례와 전체 평가 집합도 공개한다.
4. **T6 선택:** 후보 수/selector/분리 품질 ablation. learned prior나 CIFAR-100/FC2는 현재 완료된 기여로 쓰지 않는다.
5. **T7–T8 주장에 맞춰:** utility/full-model bytes 및 related work/제출 형식 확인.

최종 결론 문장 틀(수치를 채우기 전에는 본문 확정 주장으로 사용 금지):

> Under the fixed attack budget and the preregistered recovery criterion, reconstruction remained successful at [bit/retention levels], but failed to meet the criterion at [levels]. Randomized decoding [observed effect], without establishing a general privacy guarantee.

## 7. 참고문헌 및 문서 확인

핵심 배경 설명은 공식 논문에 연결했다: gradient reconstruction의 [DLG, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html), ICA 기반 CPA의 [ICML 2023 논문](https://proceedings.mlr.press/v202/kariyappa23a.html), perceptual metric의 [LPIPS, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html). 최신 related-work 전체 검토를 끝냈다는 뜻은 아니다.

LaTeX는 원래 ICASSP `spconf` 구조를 유지하고 bibliography를 파일 내부에 넣었다. 이 환경의 PATH에서 `pdflatex`, `tectonic`, `latexmk`를 찾지 못했으므로 PDF 렌더링 및 페이지 수는 확인하지 못했다. conference template의 `spconf.sty`가 필요하다. TODO는 기본적으로 보이도록 설정했으며, 해결한 뒤 `\showtodosfalse`로 숨길 수 있다. 미해결 TODO를 숨기는 것이 주장의 근거를 보충하지는 않는다.
