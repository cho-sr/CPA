# no-CPA control 결과 철회 (2026-09-09)

사용자 확인: **“no cpa control의 결과는 변인 통제에 실패한 실험이라 버려야하는 실험임.”**

`m7_no_cpa_control`의 FP32 및 4-bit 결과는 **무효**이며 논문, 그래프,
대조군 비교와 후속 분석의 근거에서 제외한다. 구체적으로 어떤 변인이
통제되지 않았는지는 이 지시에서 지정되지 않았으므로 추가로 단정하지 않는다.
실행이 완료되었다는 기록과 실험의 과학적 유효성은 별개다.

철회 범위:

- no-CPA 평균·분포를 유효한 대조군 또는 LPIPS 복원 바닥으로 사용하는 해석.
- no-CPA 대비 효과 크기, 두 대조군을 결합한 IUT `p=0.03125`, CPA 기여도 입증 주장.
- no-CPA 기준 `F-L`, `(F-L)/(F-P)` 정규화 전체와 기존 68.2878% / 76.6870% 주장.
- no-CPA로 보정한 기존 rank 성공 판정과 공통 `LPIPS < 0.5704289049` 판정,
  모든 파생 성공 개수·임계선·성공한 이미지가 다수라는 해석.
- shuffled 결과의 `shuffled_minus_no_cpa_4bit` 열과 no-CPA 대비 해석.
- 해당 내용을 포함한 이전 요약 PNG, publication 그림 2, 그림 3의 기존 합본,
  그림 5 및 이전 5종 배포 ZIP. 그림 3(a)의 독립 순열 비교 자체는 무효화하지 않는다.

원본 CSV·텐서·실행 로그·프로토콜 및 lock 파일은 추적용으로 보존한다.
기존 산출물에 `complete`나 성공 판정이 남아 있어도 현재 유효한 결과로
읽으면 안 된다. 폴더별 `INVALIDATED.md`와
[`result_validity.json`](../../quantization_aware_cpa/results/result_validity.json)이
원본의 과학적 사용 가능 여부를 덮어쓴다. 폐기 데이터로 보고서와 그림을
다시 생성하는 진입점은 차단한다.

현재 요청한 비교는 다음 두 원본만 사용한다.

| 조건 | 원본 CSV | 이미지 수 |
|---|---|---:|
| FP32 client update, Cocktail Party Attack + feature inversion attack | `m6_uia_fp32_B16_fia/per_image_metrics.csv` | 80 |
| 4-bit client update observation model, Cocktail Party Attack + feature inversion attack | `m7_uia_4bit_B16/per_image_metrics.csv` | 80 |

동일 80장 비교의 LPIPS·PSNR·SSIM과 직접 측정한 임베딩 `|cosine| > 0.9`
개수(79/80, 53/80)는 no-CPA를 사용하지 않아 이번 철회의 영향을 받지 않는다.
이는 폐기한 LPIPS 임계값 기반 성공 개수와 다른 지표다.
현재 [PNG](../../quantization_aware_cpa/results/client_update_fp32_vs_4bit/client_update_fp32_vs_4bit.png)와
[배포 ZIP](../../quantization_aware_cpa/results/client_update_fp32_vs_4bit.zip)은 유지한다.

독립 CPA/UIA 원시 지표, prior-only 및 순열 비교의 독립 원시 지표,
진행 중인 direct upstream FedAvg naive CPA 실험은 이번 지시의 폐기 대상이 아니다.
새 실험이나 대체 임계값을 이 정정으로 승인하거나 생성하지 않는다.
