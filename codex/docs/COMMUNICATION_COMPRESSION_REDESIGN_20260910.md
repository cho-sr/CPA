# 통신 압축만 바꾸는 inversion 실험 재설계 — 2026-09-10

상태: **코드·기존 결과 감사와 실험 설계 완료. 새 코덱 구현 및 공격 실험은 아직 실행하지 않음.**
기계 판독용 설계: [m10_compression_only_B16.protocol.json](../../quantization_aware_cpa/configs/m10_compression_only_B16.protocol.json).
이 JSON은 실행용 runner config가 아니다. 기존 실행·결과·프로토콜 lock은 보존한다.

## 1. 검증할 질문

**동일한 클라이언트 업데이트를 실제로 더 적은 바이트로 전송해도, 기존 inversion attack으로 원본의 정보를 복원할 수 있는가?**

독립변수는 업데이트의 통신 표현(FP32 / packed 4-bit)이다. 최신 M8 CPA+FIA를 주 공격으로 고정하고, 최근 M9 raw-row+FIA도 별도 행에서 그대로 유지한다. 공격 간 차이와 압축 전후 차이를 구분한다. 압축 후 성공은 가설이며, 결과가 나오기 전에 확정하지 않는다.

## 2. 최근 실제 실행에서 확인한 문제

완료 순서는 M6 FP32 → M7 dither4 → M8 직접 upstream CPA → M9 raw-row FIA이다. M8 완료 시각은 2026-09-09 16:40 UTC, M9는 2026-09-10 05:26 UTC이다. 네 실험 모두 80행 CSV의 저장 해시와 평균 지표가 일치했고, 이미지 index·label·path도 동일한 80장으로 확인됐다.

| 기존 실행 | 관측 / 공격 | LPIPS | source absolute cosine > 0.9 |
|---|---|---:|---:|
| M6 | FP32 / 포팅 CPA+FIA | 0.454597 | 79/80 |
| M7 | dither4 / 포팅 CPA+FIA | 0.558584 | 53/80 |
| M8 | 동일 dither4 / upstream CPA+FIA | 0.556121 | 53/80 |
| M9 | 동일 dither4 / 고정 raw rows → abs → FIA | 0.706844 | 1/80 |

출처: 각 `results/m6_uia_fp32_B16_fia`, `m7_uia_4bit_B16`, `m8_fedavg_naive_cpa_dither4_B16`, `m9_fedavg_rawrow_dither4_B16`의 `summary.json`, `per_image_metrics.csv`, 완료 기록.

### 2.1 현재 stochastic은 정수 코덱이 아니다

`qacpa/quantizer.py:52–75`의 실제 동작은 다음과 같다.

```text
r[row] = max(abs(Y[row]))
delta[row] = 2*r[row]/15
Y_hat = Y + Uniform(-delta/2, delta/2)
Y_hat = 원본 Y를 사용하는 셀 안쪽 투영(Y_hat)
```

정수 코드, 4-bit packing, 독립 decoder가 없다. `run_m7_uia_4bit.py:172–184`가 만든 `quantized.pt`에는 FP32 `Y_hat`와 FP32 `delta`가 저장되고, `run_m8_fedavg_naive.py:183–188`는 그 `Y_hat`를 공격한다. 실제 batch00 파일의 `Y_hat`는 `(4096,25088)`, `torch.float32`였다.

CPU 확인용 `linspace(-1,1,4097)` 한 행을 bits=4, seed=0으로 처리했을 때 stochastic 출력은 **서로 다른 값 4097개**였다. 따라서 현재 결과는 해당 잡음 모델 아래의 복원 결과이며, 실제 4-bit 전송량 감소의 검증 결과로 사용할 수 없다.

`rounding='deterministic'`으로 옵션만 바꾸는 것도 충분하지 않다. 공통 `_project_observation_into_cell`(`quantizer.py:109–126`)이 원본 Y로 복호값을 이동시킨다. 같은 CPU 확인에서 deterministic 출력도 17개 값이었고, 정수 코드 단위 편차 `abs(Y_hat/delta-round(Y_hat/delta))`의 최댓값이 약 `7.49e-5`였다. 기존 함수를 실제 decoder로 재사용하지 않는다.

### 2.2 기존 FP32 지표를 최신 공격의 대조군으로 바로 쓰면 안 된다

M6/M7은 포팅된 fp64 covariance `/d` + `eigh`와 **FIA 전** Hungarian 순열을 사용한다. M8은 upstream fp32 covariance `/(d-1)` + `eig`와 **FIA 후 평가에서만** Hungarian 순열을 사용한다. FIA 초기 잡음과 source의 대응도 달라진다.

따라서 `M6 FP32 vs 새 M8 4-bit`를 주 비교로 삼으면 공격 구현까지 바뀐다. **M8 경로를 그대로 사용하는 새 FP32 arm이 필요하다.** 상세 근거는 [기존 M8 프로토콜](FEDAVG_NAIVE_CPA_PROTOCOL.md)에 이미 기록돼 있다.

## 3. 채택할 압축 방식

**행별 16레벨 uniform affine quantization + 실제 stochastic rounding + 4-bit packing**을 사용한다. 식과 코드 식별자는 `uniform_affine_uint4_stochastic_v1`로 고정한다.

기존 의도였던 행별 max-abs 범위, 확률적 반올림, `delta=2r/15`를 유지하기 위한 선택이다. 결정적 `[-7,7]` 양자화도 유효하지만, 이번에는 반올림 규칙과 간격 `r/7`을 함께 바꾸지 않는다. 결과를 보고 둘 중 유리한 코덱을 선택하지 않는다.

행 i에서 r_i>0인 경우:

```text
r_i = float32(max_j(abs(Y_ij)))
delta_i = 2*r_i/15
t_ij = clip((Y_ij/r_i + 1)*7.5, 0, 15)
l_ij = floor(t_ij)
p_ij = t_ij-l_ij
q_ij = l_ij + 1[U_ij < p_ij]          # 정수 0..15
packed[k] = q[2*k] | (q[2*k+1] << 4)
Y_hat_ij = float32(r_i*(2*q_ij/15-1)) # payload로만 계산
```

- FP32 radius r_i를 행마다 하나씩 보낸다. delta와 offset=-r은 수신자가 계산한다. 별도 zero-point는 없다.
- 반올림과 위 식의 중간 계산은 CPU float64, 복호된 공격 입력은 FP32로 고정한다. 양 끝에서 같은 전송 radius와 연산 규칙을 쓴다.
- r_i=0인 행은 q=0으로 인코딩하고 정확히 0으로 복호한다. 비영 행의 16개 격자에는 0이 정확히 포함되지 않음을 명시한다.
- encoder 전용 CPU `torch.Generator` seed는 `0+1000003*batch_index`. 행 순서대로 64행 chunk에서 float64 난수를 생성한다. 영 행을 포함한 모든 원소에 난수를 생성한 뒤 영 행의 코드를 0으로 덮어써서 RNG 소비 순서를 고정한다. codec seed·버전·chunk·torch 버전을 고정하고 기록한다. 공격 RNG와 공유하지 않는다.
- decoder는 seed가 필요 없다. 원본 Y, 잡음 배열, FP32 잔차, 원본 이미지/embedding을 받지 않는다.
- `q`를 int8/uint8 배열에 담는 것만으로는 4-bit 전송이 아니다. **두 코드를 실제 한 바이트로 packing**해야 한다.
- NaN/Inf, 음수 radius, 길이·shape 불일치, noncanonical padding은 거부한다. 모든 영 행도 생략하지 않고 동일한 dense 형식으로 저장한다.
- 기존 원본 의존 셀 투영, clipping percentile 조정, error feedback, 추가 재학습은 이 비교에 넣지 않는다.

이것은 QSGD 원 구현의 재현이 아니다. 실제 이산 확률적 양자화와 별도의 인코딩이 통신 압축을 구성한다는 구분은 [QSGD 원 논문](https://papers.nips.cc/paper/6768-qsgd-communication-efficient-sgd-via-gradient-quantization-and-encoding.pdf)의 방식과 일치한다. 여기서는 위의 행별 max-abs와 고정 길이 4-bit 표현을 별도로 정의한다.

**오차 모델도 정정한다.** 이상적 산술에서 실제 stochastic rounding의 조건부 평균 오차는 0, 분산은 `delta²*p*(1-p)`, 최대 절대오차는 delta 이하다. 기존의 독립 균등 오차, `delta²/12`, `±delta/2` 셀은 성립하지 않는다. FP32 반올림 허용오차를 별도로 기록한다. M8/M9는 debias·MP·Stage2를 쓰지 않으므로 공격을 변경할 필요가 없다. 과거 MP 예측 수치는 새 코덱에 전용하지 않는다.

## 4. 기존 조건 고정 및 비교표

| 공격 경로 | FP32 packet | 동일 업데이트의 packed 4-bit packet |
|---|---|---|
| M8 upstream CPA → abs → FIA (주 비교) | 새 실행 | 새 실행 |
| M9 고정 raw rows → abs → FIA (기존 보조 비교 유지) | 새 실행 | 새 실행 |

**각 행 안에서 압축만 비교한다.** 기존 M8/M9 dither 수치는 별도 역사적 참고표에 두며 새 4-bit 열을 채우는 데 재사용하지 않는다. 동일 배치·압축 조건의 packet은 두 공격이 공유한다. M9에 새 CPA나 normalization을 추가하지 않는다.

고정 사항:

- ImageNet val, pretrained VGG16 IMAGENET1K_V1, round0, conv 동결, classifier eval/dropout off.
- victim seed0, distinct-label n=16, batch0..4, 동일 80장. 원본 선택을 반복하지 않고 저장 metadata를 쓴다.
- FC-only SGD lr=0.001, momentum=0, weight decay=0, T=10. 각 update는 round0에서 독립적으로 시작한 기존 업데이트다. 서버 학습 5라운드가 아니다.
- `results/m6_uia_fp32_B16_fia/batchXX/observation.pt`의 **실제** `observation.Y_delta_over_lr`를 인코딩한다. 이상적인 `Y_factor`나 `B_cumulative@Z`로 대체하지 않는다. 새 학습은 필요 없다.
- 관측은 기존 정의 `Y=(W_initial-W_final)/lr`를 유지한다. 통신 프로토콜상 정규화된 업데이트를 보내고 서버는 알려진 lr로 `DeltaW=-lr*decoded_Y`를 해석한다. 원래 파라미터 차이와의 FP32 왕복이 bitwise 같다는 주장은 하지 않는다.
- 개별 client의 한 전송을 관측하는 기존 위협 모델, 공격 대상 `classifier.0.weight` `(4096,25088)`를 유지한다.
- CPA는 `qacpa.upstream_naive.run_upstream_cpa`, full16, identity init, seed0+batch, Adam 25000회/lr0.001, scheduler none, decor5.3/T7.7/ne1/nv0.13/l1=5/tv0.1. upstream의 실제 ne/TV 처리까지 그대로 둔다.
- M9는 기존에 저장한 행 ID를 재사용한다. CPU seed0+batch의 `randperm(4096)[:16]`와 일치하는지 확인하고, 행 순서·영 행도 유지한다.
- FIA는 두 경로 모두 `qacpa.upstream_naive.run_upstream_fia`, 원래 Direct, 16장씩 25000회, Adam lr0.1, TV0.1/FI1, cosine, 매 batch seed42. 추정 source 또는 선택한 raw row의 abs만 입력한다.
- signed source/raw row의 absolute-cosine Hungarian 대응은 FIA 완료 후 평가에서만 적용한다. 모든 80장을 평가하고 보충·선별하지 않는다.
- LS-A, debias, MP 진단/절단, Stage2, refinement는 기존 M8/M9처럼 비활성이다. 모델·공격·FIA·metric 코드의 해시를 arm 공통으로 고정한다.

## 5. 전송량의 정확한 범위

새 packet은 32-byte little-endian header와 데이터로 정의한다. header layout은 `<8sBBBBIIQI`: magic(8), version(1), codec_id(1), bits(1), flags(1), rows(4), cols(4), elements(8), radius_count(4). shape와 전송 dtype을 이 header로 복원한다.

- FP32 packet: header + row-major little-endian FP32 Y.
- 4-bit packet: header + rows개의 little-endian FP32 radius + packed codes. 홀수 개 원소이면 마지막 high nibble은 0이고 elements로 제거한다.
- RNG seed와 실험 provenance는 연구 기록이며 decoder 입력이 아니다. 추가 metadata를 실제 전송하게 되면 그 바이트도 합산한다.
- `torch.save`의 전체 checkpoint 크기나 메모리의 dequantized FP32 크기를 통신 packet 크기로 대신하지 않는다. 실제 파일 길이/전송 buffer 길이를 측정한다.

공격 대상 행렬 한 개의 **설계상 예상값**:

| 항목 | 바이트 |
|---|---:|
| FP32 데이터 | 411,041,792 |
| packed 4-bit 코드 | 51,380,224 |
| 행별 FP32 radius | 16,384 |
| 각 packet header | 32 |
| FP32 packet 합계 | 411,041,824 |
| 4-bit packet 합계 | 51,396,640 |

예상 감소는 약 **7.9974배 / 87.496%**이며, 코덱 구현 후 실제 packet으로 확인해야 한다.

이 수치는 **첫 FC weight 전송분**이다. 저장 관측은 FC 전체 업데이트를 담고 있지 않으므로 전체 client upload 8배 감소로 보고하지 않는다. 가령 나머지 학습 대상 FC 파라미터를 모두 FP32로 전송한다고 가정하면 데이터+radius 기준 전체 FC payload는 `494,571,424 → 134,926,240 bytes`, 약 3.6655배가 된다. 이는 실제 측정값이 아닌 구성 기반 예시이며 다른 tensor와 metadata의 실측을 대체하지 않는다. 전체 FL 수렴·정확도나 다라운드 누적 통신량은 이번 단일 관측 실험의 결론 범위가 아니다.

## 6. 구현 순서와 검증 항목

1. 새 `qacpa/communication_codec.py`를 추가한다. 기존 `quantizer.py`와 역사적 결과는 수정하지 않는다. `encode(Y)->bytes`, `decode(bytes)->Y_hat`를 분리한다.
2. integer code/packing 왕복, 영 행·양음 극값·격자점·중간점·홀수 길이·손상 packet을 확인한다. decoder에 원본을 제공하지 않은 별도 호출에서 같은 결과가 나와야 한다. FP32 arm은 bitwise round trip을 확인한다. stochastic 평균/분산 확인은 toy 입력과 고정 사전 반복 수로만 수행한다.
3. 기존 입력과 M8/M9 코드·설정·weights·M9 row ID의 해시를 검증하고 새 실행 manifest를 만든다. 새 codec ID와 payload hash가 cache key에 들어가야 한다. 역사적 `quantized.pt`를 새 입력으로 읽으면 실패 처리한다.
4. 저장된 다섯 Y로 FP32/4-bit packet을 한 번씩 생성한다. 실제 바이트, codec 시간, error mean/MSE/relative L2/cosine, 영 행 수, endpoint-code 점유율과 실제 범위 초과 clipping을 구분해 기록한다. 진단용 원본 접근은 encoder/평가에만 둔다.
5. 수신 packet을 복호해 고정 공격을 실행한다. 주 비교 CPA의 FP32/4-bit 전체80장을 먼저 완료하고, 같은 packet으로 M9 FP32/4-bit를 실행한다. 단일 GPU에서 순차 실행한다. 결과가 나빠도 seed·iteration·rank·TV를 바꾸거나 INT8로 주 결과를 교체하지 않는다.
6. 모든 source-order 복원과 post-FIA 평가 결과, batch별·image별 지표, packet hash와 runtime을 저장한다. 공통 코드가 바뀌면 새 experiment version으로 모든 관련 arm의 정합성을 다시 확인한다.

새 runner는 `qacpa.run_compression_only`로 별도 추가하는 설계다. M8/M9 공격 함수와 평가 함수를 재사용하며, 이번 문서 작성 시점에는 아직 존재하지 않는 실행 명령을 제공하지 않는다.

## 7. 판정과 산출물

- 1차: 실제 전송 바이트와 FP32 대비 감소율. 수신 packet만으로 공격이 실행됐는지 확인한다.
- 복원: 전체80 LPIPS/PSNR/SSIM, 배치별 평균, 이미지별 paired 차이, 전체 source cosine 분포와 기존 `|cos|>0.9` 개수. 이 개수는 embedding 복원 지표이며 이미지 복원 성공률로 바꿔 부르지 않는다.
- 그림: 모든80장의 원본/FP32/4-bit를 동일한 순서와 크기로 제시한다. NumPy seed42로 미리 고정한 동일 random12도 함께 제시한다.
- CPA와 raw-row의 비교는 두 입력 구성 방법의 기술적 비교다. 압축만의 효과는 각각의 FP32/4-bit 쌍에서 해석한다.
- 기존 무효 no-CPA control, LPIPS cutoff `0.5704289049`, 정규화 retention 및 파생 성공 횟수는 사용하지 않는다. 새 결과를 보고 임계값을 만들어 성공을 선언하지 않는다.
- “통신량은 줄었지만 복원이 무너졌다”도 그대로 보고한다. 결과가 없는 현재 상태에서는 “실제 압축에서도 공격이 성공했다”라고 쓰지 않는다.

필수 산출물은 `input_manifest.json`, `communication.csv`, `quantization_diagnostics.csv`, `per_image_metrics.csv`, `per_batch_metrics.csv`, `paired_compression_effect.csv`, 모든80장 및 random12 그림이다. FP32 대비 성능 변화와 압축 후 실제 복원 정도를 함께 제시하고, 별도 성공률 기준이나 학습 효용 검증은 이 설계에 임의로 추가하지 않는다.
