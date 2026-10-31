# INT4 per-tensor / per-channel FC1 비교 (B=2, SGD 3 steps)

세 arm 모두 VGG16 IMAGENET1K_V1에서 Conv를 고정하고 FC 6개 텐서를 SGD로 3 step 학습했습니다. 공식 CPA 25,000회 뒤 abs(S_hat)를 Direct FIA에 넣어 25,000회 실행했습니다.

| Arm | FC1 update 상대 L2 | update cosine | PSNR (dB) | SSIM | VGG LPIPS | 임베딩 cosine |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 0 | 1 | 12.84057 | 0.288133 | 0.456719 | 0.999980 |
| int4_per_tensor | 0.631835 | 0.803866 | 12.10495 | 0.245963 | 0.760441 | 0.437659 |
| int4_per_channel | 0.296708 | 0.957303 | 12.31299 | 0.258675 | 0.590131 | 0.964400 |

[FACT] Per-tensor scale은 1개(4 B), per-channel scale은 4,096개(16,384 B)입니다.
[FACT] Per-channel에서 2,387/4,096행의 absmax가 0이어서 지시된 convention대로 s_r=1.0을 사용했습니다. 따라서 전체 행 scale/ratio 분위수의 중앙값 이상은 이 zero-row convention 영향을 받습니다.
[FACT] 그림은 원본 이미지별 pixel-MSE Hungarian 대응을 사용합니다. 공격 최적화에는 원본 이미지·label을 전달하지 않았습니다.
[LIMIT] 이미지 2장과 seed 1개에 대한 서술 비교입니다. 유의성, 성공 기준, 일반화 또는 방법 전체의 우위를 주장하지 않습니다.
[LIMIT] INT4 arm은 FC1 weight update만 양자화했습니다. 나머지 다섯 FC update는 CPA 입력으로 사용하지 않았습니다.

비교 수치와 전체 매칭은 `comparison.json`, 그림은 `comparison.png`에 있습니다.

[주의] 공식 CPA의 복소 고유값→실수형 cast와 FIA CUDA adaptive_avg_pool 역전파의 비결정성 경고가 발생했습니다. 동일한 원본 코드와 warn-only 설정을 유지했으며, GPU 결과의 strict bitwise 재현성은 보장되지 않습니다.
