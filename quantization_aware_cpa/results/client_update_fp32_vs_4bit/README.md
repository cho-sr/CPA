# Client-update inversion: FP32 and 4-bit observations

방금 완료한 **4-bit 클라이언트 업데이트 복원 실험**과 **동일 조건의 FP32 기준선**을 비교한 그래프입니다. VGG-16 / ImageNet, 클라이언트당 16장, 로컬 10 step, 독립 업데이트 5개, 두 조건 모두 동일한 전체 80장을 사용합니다.

[그래프 PNG](client_update_fp32_vs_4bit.png) · [벡터 PDF](client_update_fp32_vs_4bit.pdf) · [편집용 SVG](client_update_fp32_vs_4bit.svg) · [영문 캡션](caption.md)

4개 패널은 LPIPS, PSNR, SSIM, 실제 임베딩의 source cosine 분포입니다. 옅은 점은 이미지별 값, 빈 원은 배치별 평균, 검은 마름모는 전체 평균이며 연결선은 동일 배치끼리 연결합니다.

| 지표 | FP32 | 4-bit observation model |
|---|---:|---:|
| LPIPS | 0.454597 | 0.558584 |
| PSNR (dB) | 22.406 | 21.926 |
| SSIM | 0.342655 | 0.297555 |
| 실제 임베딩 복구, absolute cosine > 0.9 | 79/80 | 53/80 |

방법은 두 조건 모두 Cocktail Party Attack (CPA) + feature inversion attack (FIA)입니다. 4-bit 표기는 per-row subtractive-dither 관측 모델을 가리킵니다. `per_image_metrics.csv`와 `metric_summary.csv`에 이 두 조건의 수치만 담았습니다. PNG는 300 dpi이며, 원본 CSV와 대응을 검증했습니다.
