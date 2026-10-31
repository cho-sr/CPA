# CoRA 중심 개편 초안의 근거

현재 초안은 [test_cora.tex](test_cora.tex)이다. `test.tex`와 `test_revised.tex`는 보존했다. 새 실험, 이미지 복원, 재평가는 수행하지 않았다. CODEX_PLANNER의 문서 작성 및 기존 결과 검토 범위다.

**최신 형식 기준:** 사용자는 모든 후속 논문 편집에서 `test.tex`의 포맷을 따르도록 지정했다. `article + spconf`, Times 본문, 2단, 제목·저자·절 제목·캡션 스타일을 동일하게 사용한다. 일반 article/geometry 대체 레이아웃으로 돌아가지 않는다. 최신 Table 1은 예시처럼 **Dataset / Method / 압축 조건별 PSNR·SSIM·LPIPS**로 구성하며 CIFAR-100과 ImageNet은 행 그룹이다. 최신 PDF는 5페이지이고 미리보기는 `paper_preview/dataset_rows_spconf`에 있다. 아래 이전 편집 단계의 6페이지·일반 article 설명은 과거 기록이다.

## 사용자 지시 반영

- 논문의 중심을 랜덤 CoRA의 방법과 비트별 복원 품질로 변경했다.
- 성공률, 성공확률, 누적 침해확률, `p=0.04`, 독립 시행 확률식, 성공 판정 임계값과 이를 정하라는 TODO를 새 본문에서 모두 제외했다.
- 이미지 수를 복원 성과의 대체 표현으로 사용하지 않았다. 재현에 필요한 배치 크기 `B=16`은 실험 설정에만 명시했다.
- `K=8`은 후보 8개를 만들고 공개 점수로 선택하는 절차다. 평균 ensemble 또는 private LPIPS best-of-8로 쓰지 않았다.
- 원본의 3.2 한 부분만 고치는 대신 제목·초록·서론·방법·실험·논의·결론을 같은 주장에 맞춰 다시 작성했다.

이 문서는 기존 `PAPER_REVISION_NOTES_20260920.md`의 **편집 방향과 TODO 우선순위**를 대체한다. 기존 메모의 성공 기준을 먼저 정하라는 요구는 더 이상 이 초안에 적용하지 않는다. 과거 결과와 기록 자체는 보존한다.

## 새 구조와 원본에서 교체되는 부분

| 새 위치 | 역할 | 원본에서 대체하는 내용 |
|---|---|---|
| 제목·초록·1장 | CoRA와 비트별 복원 품질을 핵심 질문으로 제시 | lucky draw, probabilistic privacy, calibrated sampling 주장 |
| 2장 | 단일-client local update, deterministic signed per-tensor codec | 서로 다른 Setup A/B와 압축 방식이 섞인 설명 |
| 3장 | cell sampling → source separation → FIA → 공개 점수 선택 | 실제 bit sweep과 다른 recompressed-update selector |
| 4.1 | 모델·로컬 학습·공개/비공개 정보·평가 matching | 실험마다 다르던 조건을 한 표에 섞은 비교 |
| **4.2** | **CIFAR-100/ImageNet을 행 그룹으로 묶은 통합 표: 조건별 naive/random 비교** | 이전 3.1–3.4의 중심 실험 논증; CIFAR-100 수치는 미기입 |
| 4.3 | 희소화의 별도 실험 조건과 결과 | 유지율별 zero filling / random filling 비교 |
| 4.4 | naive / 첫 draw / 공개 rank-1의 차이 | random 1과 K=8 avg의 혼동 |
| 4.5 및 Figures 1–2 | 모든 비트·희소화 유지율의 naive/random 복원 이미지 대조 | 성공 비율과 확률 그래프 |
| 4.6·5장·6장 | 실제 계산 예산·실험 범위·CoRA의 한계 | 반복하면 확률적으로 공격된다는 결론 |

비트별 결과는 CoRA를 설명한 **후**, Experimental Evaluation의 첫 주요 결과로 배치했다. 최신 요청에 따라 CIFAR-100과 ImageNet을 한 표의 행 그룹으로 통합하고, 양자화와 희소화는 서로 다른 패널로 구분했다. 두 패널의 attack backend와 평가 matching이 다르므로 절대 점수를 압축 방식 간의 통제 비교로 해석하지 않는다. 과거 source averaging 결과는 포함하지 않았다.

## Table 1과 Figure 1의 원자료

각 JSON의 `mean_metrics.center_25000`과 `mean_metrics.rank1_member_XX_25000`을 사용했다. `rank1`은 FIA 5,000회 공개 screening에서 고정한 순위이며 최종 private metric 순위가 아니다.

| bit | naive PSNR | CoRA PSNR | naive SSIM | CoRA SSIM | naive LPIPS | CoRA LPIPS | 선택 member |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 12.0569 | 12.0568 | 0.2924 | 0.3070 | 0.5594 | 0.5558 | 01 |
| 7 | 11.7955 | 11.8314 | 0.2764 | 0.2827 | 0.6118 | 0.6332 | 00 |
| 6 | 11.6397 | 11.6675 | 0.2596 | 0.2690 | 0.6961 | 0.6659 | 01 |
| 5 | 11.5524 | 11.5458 | 0.2620 | 0.2558 | 0.7826 | 0.7490 | 05 |
| 4 | 11.5197 | 11.5028 | 0.2921 | 0.2576 | 0.8071 | 0.8085 | 06 |

직접 확인한 JSON:

- [8bit](quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918/comparison.json)
- [7bit](quantization_aware_cpa/results_private/signed_int7_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)
- [6bit](quantization_aware_cpa/results_private/signed_int6_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)
- [5bit](quantization_aware_cpa/results_private/signed_int5_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)
- [4bit](quantization_aware_cpa/results_private/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/comparison.json)

4비트 첫 draw는 동일 JSON의 `rank3_member_00_25000`이다: PSNR `11.555842555517009`, SSIM `0.2614190003468897`, LPIPS `0.8055475391447544`. 별도의 K=1 대조 실험으로 표시하면 안 된다. 8/6/5비트 member 00은 이 프로토콜에서 25,000회까지 이어진 후보가 아니므로 해당 endpoint가 없다.

이 평가 JSON에는 같은 프로토콜의 source CS 표가 없다. 다른 실험의 CS 또는 FP32 값을 가져와 채우지 않았다. 과거 `.135 / 11.30 / .106 / .737` 계열 수치는 현재 ImageNet per-tensor 표의 random 1 값으로 사용하지 않는다.

## 방법론 근거와 용어

- [등록 설정](quantization_aware_cpa/results/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep/registration.json): 후보 수, seed, separator, FIA, continuation, ranking 설정.
- [codec](quantization_aware_cpa/qacpa/signed_bits_observation.py): signed per-tensor, Qmax=`2^(b-1)-1`, FP32 scalar scale, deterministic nearest-even.
- [sampler](quantization_aware_cpa/qacpa/signed_bits_randomized.py): finite-precision endpoints, uniform interpolation, extremal anchor, exact code/scale reproduction, sign/permutation-only alignment.
- [runner](quantization_aware_cpa/qacpa/signed_bits_screen3_runner.py): center + 8 members, top-3 continuation.
- [FIA와 공개 score](quantization_aware_cpa/qacpa/signed_int8_screen3_fia.py): feature cosine distance + 0.1 normalized TV, optimizer/scheduler state 보존.
- [evaluator](quantization_aware_cpa/qacpa/signed_int8_screen3_evaluation.py): private data는 selection 고정 이후 평가에 사용; candidate별 pixel-MSE Hungarian assignment를 PSNR/SSIM/LPIPS에 공통 적용.

CoRA의 기존 약어 확장 `Consistency-Ranked restart Attack`은 유지하되, 이 초안에서 consistency는 **feature agreement + TV**임을 명시했다. 기존 원문의 “local training 후 재압축한 update와의 일치 점수”와는 다르다. Calibrated prior, 원문 CPA 그대로의 실행, 모든 비트에서 우수한 성능을 주장하지 않는다.

관측 manifest의 update 부호는 `theta_after10_minus_theta_start`다. 본문도 `U = W(T) - W(0) = -eta sum(A_t) H`로 맞췄다. 비트 수는 numerical precision이며 artifact의 실제 code storage는 int8이다. 따라서 packed bandwidth 절감 실측을 주장하지 않는다.

FIA 계산량은 랜덤 탐색 `8×5k + 3×20k = 100k`, 별도 center 25k를 포함하면 125k다. Center source는 공개 정렬 참조에도 쓰인다. 최종 candidate별 iteration은 같지만 총 탐색 예산은 같지 않다.

## 복원 그림과 본문 표현

초기 초안에서 사용한 **ground truth + naive baseline + public ranks** 전체 PNG는 다음 경로에 보존되어 있다. 최신 본문은 아래의 새 paired panels로 대체했다.

- [8bit 전체 그림](quantization_aware_cpa/results_private/signed_int8_screen3_repair_B16_T10_20260918/evaluation_final_20260918/figures_with_baseline_20260918/contact_sheet_b8_25000_with_baseline.png)
- [4bit 전체 그림](quantization_aware_cpa/results_private/signed_int4_randomized8_screen3_fia_B16_T10_20260918_sweep/evaluation_recovery_20260918/figures_with_baseline_20260918/contact_sheet_b4_25000_with_baseline.png)

이 기존 전체 그림의 열 순서는 GT / naive / public rank 1 / rank 2 / rank 3이다. CoRA는 **세 번째 열**이다. 새 본문 그림은 첫 두 등록 이미지(index 0, 1)를 모든 조건에 고정해 Original / (Naive, CoRA 또는 Random) 쌍으로 배열한다. 복원 품질에 따라 이미지를 고르지 않았다. 8비트에서 일부 객체·배경 형태가 보이고 4비트에서는 구조가 크게 손상된다는 시각적 서술을 사용했다.

수치가 뒷받침하는 결론은 `8→4bit에서 CoRA PSNR·LPIPS 악화, SSIM은 비단조, naive 대비 이득은 비트·지표에 따라 다름`이다. “몇 비트까지 반드시 복원된다”는 일반 경계로 확대하지 않는다. 성공률이나 임계값을 다시 도입할 필요는 없다.

## 문서 확인 및 남은 편집 작업

- 새 초안에 성공률/확률 주장이 없는지, 수치 표가 원본 JSON과 일치하는지, LaTeX 환경·교차참조·citation 및 그림 파일 경로가 맞는지 정적으로 점검했다. 모두 통과했다. `test.tex`의 SHA256도 유지됐다.
- 참고문헌은 기존에 확인한 DLG, CPA, LPIPS의 3개 항목을 파일 내부에 넣었다. 최신 관련 연구 전체 검토를 마쳤다는 뜻은 아니다.
- `spconf.sty`가 있으면 conference 스타일을 사용하고, 없으면 일반 2단 article 레이아웃으로 처리한다. 기존 LPIPS 곡선은 최신 paired panels로 대체했다.
- 후속 요청에 따라 portable Tectonic 0.17.0을 `.tools/tectonic-0.17.0`에 준비해 실제 컴파일했다. 최초 출력은 5페이지였으며, 이후 사용자의 본문 높이 조정 요청을 반영한 최신 [PDF](test_cora.pdf)는 6페이지다. 수식 넘침과 미해결 reference 경고는 없다. 일부 underfull 조판 경고와 시스템 Fontconfig 설정 경고는 남지만 PDF 생성과 실제 글꼴 출력은 확인했다.
- 현재 출력은 `spconf.sty`가 없는 일반 2단 article 형식이다. 제출 전 conference 템플릿에서 레이아웃·페이지 수를 다시 확인해야 한다.
- [재컴파일 스크립트](compile_cora.ps1)는 `powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\compile_cora.ps1`로 실행한다. `-OpenPdf`를 덧붙이면 생성 후 기본 PDF 뷰어로 연다. 실행 정책은 해당 프로세스에만 적용하며 시스템 설정을 변경하지 않는다. LaTeX package cache는 `.tools/tectonic-cache`를 사용한다.
- 최신 [첫 페이지 미리보기](paper_preview/combined_results/test_cora_page_1.png)를 포함한 페이지 PNG는 `paper_preview/combined_results`에 있다. 이전 `cifar100_reserved`, `height_229mm`, `final` 폴더는 과거 편집 단계 출력이다. 문서 렌더링용 pypdfium2는 `.tools/pdf-preview`에만 설치했다.
- 모든 비트의 전체 그림은 각 결과 폴더의 `figures_with_baseline_20260918`에 보존되어 있다. 최신 PDF에서는 전체 그림 부록을 빼고 두 개의 본문 비교 그림으로 대체했다.
- 독립 batch/seed 반복과 계산량을 맞춘 대조군은 주장의 일반화 범위를 넓히는 후속 항목이며, 이번 문서 작업에서 실행하지 않았다.

작성 직전 `test.tex` SHA256: `D2F75EAE86714394718B076A7691ECD13087315E60E5975E073A671A87FEEA74`.

## 본문 높이 조정 후 페이지 내용 확인 (통합 표 편집 이전 기록)

사용자 요청에 따라 `test_cora.tex`의 geometry에서 본문 높이만 229mm로 설정했다. 글자 크기, 가로 폭, 원고 내용은 유지했다. 컴파일 로그의 `textheight=651.5681pt`로 설정 적용을 확인했다.

| 페이지 | 현재 내용 |
|---|---|
| 1–4 | 초록·서론·관측 모델·CoRA 방법·실험 결과·논의 |
| 5 | 결론과 참고문헌 |
| 6 | 전체 복원 이미지 부록 |

페이지 내용만 보면 결론이 5페이지로 넘어가 있고 복원 그림이 6페이지에 있어, 기술 내용을 4페이지 안에 두는 제출 구성이 아직 충족되지 않는다. 이번 요청에서는 높이만 변경하고 페이지 내용을 점검했으며, 본문 축약이나 그림 재배치는 수행하지 않았다.

## CIFAR-100 결과 자리 추가 (통합 표 편집 이전 기록)

후속 사용자 요청에 따라 ImageNet 비트별 결과인 4.2절 바로 뒤에 **4.3절 `CIFAR-100: reconstruction across bit widths`**를 추가했다. 설정 설명, 8/7/6/5/4비트 PSNR·SSIM·LPIPS 표, 복원 예시와 해석을 넣을 공간이다. 표의 측정값은 모두 `---`이며 미완료 결과용 자리임을 명시했다. 실험을 실행하거나 성능·결론을 채우지 않았다.

최신 PDF에서 예약 공간은 **4페이지 왼쪽 열**에 있다. 전체는 6페이지이며, 5페이지에는 논의의 나머지·결론·참고문헌, 6페이지에는 기존 복원 그림 부록이 있다. 본문 높이 229mm는 유지했다. 컴파일과 실제 페이지 배치 확인을 마쳤다.

LaTeX 주석에는 후속 결과를 넣을 때 모델·attack backend·decoding·예산·matching을 실제 수행 조건으로 기재하도록 남겼다. 현재 CIFAR-100 실행 인계는 original CPA/FC2/center decoding이므로 자동으로 CoRA 결과라고 표시하지 않는다.

Portable compiler 출처: [Tectonic 공식 릴리스](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0). Windows x86_64 MSVC ZIP의 SHA256 `f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`를 release metadata와 대조했다. 패키지 다운로드에 논문 내용이나 연구 데이터를 전송하지 않았다.

## 통합 표와 naive/random 비교 그림 (데이터셋 행 전환 이전 기록)

사용자의 후속 요청에 따라 기존 별도 CIFAR-100 예약 표를 통합했다. [Table 1 소스](paper_assets/combined_results.tex)는 ImageNet/CIFAR-100 각각에 PSNR·SSIM·LPIPS 열을 두고, 양자화 8/7/6/5/4비트와 희소화 유지율 50/30/20/10%마다 naive와 randomized 행을 배치한다. CIFAR-100의 모든 값은 `---`로 남겨뒀다. 희소화의 matched random 결과는 10% 유지율에서만 확인되어 다른 유지율의 random 셀도 비워뒀다.

희소화 수치는 `quantization_aware_cpa/results/cpa_fia_topk30_20_10_same80_B16/comparison_metrics.csv`와 `quantization_aware_cpa/results/cpa_fia_random_topk10_same80_R20_S5k_K3_F25k/comparison_metrics.csv`에서 읽었다. 양자화의 modified CPA/feature-score/pixel-MSE matching과 달리 original CPA/update-replay selector/source-cosine matching을 사용한다. 원고의 4.3절과 표 캡션에 이 차이를 명시했다. 희소화 random을 양자화 CoRA와 동일한 프로토콜로 표시하지 않는다.

- [Figure 1: 모든 비트의 naive/CoRA 비교](paper_assets/quantization_naive_cora.png)
- [Figure 2: 모든 유지율의 naive/random 비교](paper_assets/sparsification_naive_random.png)
- [문서 자산 생성 스크립트](paper_assets/build_comparison_panels.py)
- [수치·이미지 출처 및 SHA256 manifest](paper_assets/comparison_manifest.json)

그림은 저장된 복원 tensor와 저장된 assignment만 사용한다. 지표 계산, 후보 재선택, Hungarian 재매칭, attack 실행은 수행하지 않았다. 원본은 저장된 정규화를 되돌려 표시하고, 복원 RGB는 보정·강화·크롭 없이 표시한다. 양자화는 모든 비트에 같은 원본 두 개를 사용하고, 희소화도 모든 유지율에 동일한 두 등록 인덱스를 사용한다. 희소화 미실험 random 셀은 `Not run`으로 표시했다. 두 그림 모두 CIFAR-100 패널이 들어갈 위치를 표시했다. 입력 파일 해시를 생성 전후 대조해 기존 결과 불변을 확인했다.

본문 높이 229mm를 유지한 최신 PDF는 **6페이지**다. 4페이지에 통합 표와 양자화 그림, 5페이지에 희소화 그림과 실험·논의, 6페이지에 논의의 나머지·결론·참고문헌이 배치된다. 컴파일 성공, overfull 및 미해결 교차참조 없음, 실제 PDF 페이지 렌더링을 확인했다. 기술 내용 4페이지 제한에 맞춘 축약은 아직 수행하지 않았다. 편집 직전 원고는 `paper_assets/test_cora_before_combined_layout.tex`에 보관했다.

## 최신: 예시 형태의 Table 1 및 test.tex 포맷 적용

- `Dataset`은 CIFAR-100 / ImageNet의 두 행 그룹, `Method`는 Naive / CoRA(양자화), Naive / Random fill(희소화)로 둔다.
- 같은 Table 1에서 (a) 8/7/6/5/4비트, (b) 유지율 50/30/20/10%를 각각 가로 열 그룹으로 두고, 각 그룹 아래 PSNR↑·SSIM↑·LPIPS↓를 반복한다. 글자 크기는 9pt이며 표를 이미지로 축소하지 않는다.
- 표의 PSNR은 예시처럼 소수점 둘째 자리, SSIM·LPIPS는 넷째 자리로 표시한다. 원본 측정값과 본문의 세부 값은 그대로다. 미실험 셀은 `---`로 유지한다.
- [표 생성기](paper_assets/build_results_table.py)는 기존 JSON/CSV를 읽어 [통합 표](paper_assets/combined_results.tex)만 출력한다. 그림 생성기도 이 함수를 사용해 이전 레이아웃으로 덮어쓰지 않도록 했다. 기존 복원 그림은 그대로 유지했다.
- `test.tex`와 동일한 `article + spconf`를 사용한다. [공식 ICASSP 2027 spconf](https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/spconf.sty)를 루트에 저장했다. SHA256: `dc5d632639040cb183f2ab62780f314845aa021be73048c8a9ec9c2072d64a86`. 스타일 원본은 수정하지 않았다. 본문 영역 178×229mm를 사용한다.
- Tectonic/XeTeX에서 Times가 다른 글꼴로 대체되지 않도록 `fontenc`의 T1 인코딩을 명시했다. 실제 Times 계열 폰트와 bold/italic 적용, overfull 0건, font-shape fallback 0건, 미해결 참조 없음, 페이지 렌더링을 확인했다. 패키지 주석 인코딩 및 일부 underfull 경고는 남는다.
- 최신 PDF는 **5페이지**다. 4페이지에 Table 1과 양자화 그림, 5페이지에 희소화 그림·논의·결론·참고문헌이 있다. 기술 내용이 여전히 5페이지에 있으므로 이번 편집을 4페이지 제한 충족으로 표현하지 않는다.
- 편집 전 백업: `paper_assets/test_cora_before_dataset_rows.tex`, `paper_assets/combined_results_before_dataset_rows.tex`. `test.tex`는 이 편집에서 변경하지 않았으며 SHA256은 `E43273F70736AAB00769CF66B854880212DC0305933B8B8038207D1DD105928D`이다.

### 표 크기 축소

사용자의 표 크기 피드백을 반영해 패널 제목과 조건 머리글을 한 행으로 합치고, 지표 방향 화살표는 캡션에 한 번만 표시했다. 캡션을 줄이고 `arraystretch`를 1.1에서 1.0으로, 패널 사이 간격을 `medskip`에서 `smallskip`으로 변경했다. 데이터셋 행 그룹·모든 측정값·9pt 글꼴·`test.tex`의 spconf 포맷은 유지했다. 최신 표 미리보기는 `paper_preview/compact_table/table_page.png`다. 재컴파일과 표 렌더링을 확인했으며 overfull은 없고 전체는 여전히 5페이지다.

### 최신: Table 1을 한 단 폭으로 변경

사용자가 지적한 크기는 세로 높이가 아니라 두 단에 걸치는 가로 폭이다. Table 1을 `table*`에서 `table`로 바꾸고 표 너비를 `textwidth`에서 `columnwidth`로 변경했다. 데이터셋·방법 행 그룹과 비트/유지율 열을 유지하면서 PSNR·SSIM·LPIPS를 Metric 열의 세로 행으로 배치했다. 기존 수치와 9pt 글꼴을 유지하며 한 단 폭에 맞는다. 최신 PDF의 **3페이지 오른쪽 단**에 있고, [미리보기](paper_preview/single_column_table/table_page.png)를 렌더링해 넘침 없이 한 단 안에 들어감을 확인했다. 전체 5페이지, overfull 0건이다. 이전 두 단 표 소스는 `paper_assets/combined_results_before_single_column.tex`에 보관했다.

### Related Work 복원

사용자가 제공한 Related Work를 현재 원고에 맞춰 결론 앞 **6절**로 추가했다. 결론은 7절, 참고문헌은 8절이다. DLG/IG의 gradient matching, CPA의 ICA, Zhang의 survey, Yue의 obfuscation 분석, Li의 generative prior, 압축 gradient 대상 공격을 다루고 CoRA의 randomized decoding/public selection과 연결했다.

원문에서 누적 침해확률, calibrated sampling, 공격 backend를 전혀 수정하지 않는다는 주장, 현재 3.3절이 hard compression constraint의 성능 저하를 보여준다는 주장은 복원하지 않았다. 현재 quantization은 modified CPA이며 uniform sampler와 feature-based selector를 사용한다. `yu2024compressed`는 로컬 bibliography가 없고 실제 문헌을 특정하지 못해, 출판사에서 확인한 Ding et al.의 DLCG(2024)로 교체했다. DLCG는 sparse synthetic gradients를 다루므로 quantization-aware loss로 설명하지 않았다.

확인한 1차 출처:

- [Geiping et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)
- [Zhang et al., IJCAI 2022](https://www.ijcai.org/proceedings/2022/791)
- [Yue et al., USENIX Security 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/yue)
- [Li et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Auditing_Privacy_Defenses_in_Federated_Learning_via_Generative_Gradient_Leakage_CVPR_2022_paper.html)
- [Ding et al., Neurocomputing 2024](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011202), DOI `10.1016/j.neucom.2024.128349`.

서론에도 IG 인용을 연결하고 참고문헌 8개를 첫 인용 순서로 배치했다. 현재 `test_cora.pdf`는 **6페이지**이며 Related Work는 5페이지, 참고문헌은 5–6페이지다. 컴파일 성공, 미정의 인용 0건, overfull 0건, 5페이지 렌더링을 확인했다. 원고의 spconf 포맷과 표·그림은 유지했다. 편집 전 원고는 `paper_assets/test_cora_before_related_work.tex`에 보관했다. 과거 페이지 수 기록보다 이 결과가 최신이다.

### `Not run` 결과 재검색 및 정정

후속 폴더 검색에서 50%와 30% 유지율의 과거 randomized CoRA 결과와 실제 선택된 복원 파일을 확인했다. 기존 `Not run`은 프로젝트 전체의 실험 이력을 부정하는 표현으로 읽힐 수 있어 부정확했다. 과거 결과는 B24/T1/R8 및 별도 checkpoint/sampler를 사용하며 현재 B16/T10/R20 same80 비교와 다르다. 20%는 naive 결과를 확인했고 randomized 결과는 찾지 못했다. [상세 근거와 그림 링크](PAPER_SPARSIFICATION_RESULT_AUDIT_20260921.md)에 기록했다. 현재 그림의 빈 셀은 `N/A`로 바꾸고 캡션에 **해당 matched protocol의 결과 미확인**이라는 뜻을 명시했다. 서로 다른 실험의 수치·복원 이미지를 현재 표에 혼합하지 않았다. PNG/PDF 및 논문 PDF 재생성을 완료했다.
