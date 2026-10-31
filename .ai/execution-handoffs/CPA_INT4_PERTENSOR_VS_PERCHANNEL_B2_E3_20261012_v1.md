# INT4 FC1 weight scale: per-tensor vs per-channel, B=2, SGD epoch 3

TASK_ID: CPA_INT4_PERTENSOR_VS_PERCHANNEL_B2_E3_20261012
PLAN_VERSION: 1
STATUS: COMPLETED
AUTHOR: CLAUDE_REVIEWER (사용자 직접 지시 "2장으로 하고 지시서 작성", 2026-10-12)
ASSIGNED_ROLE: CODEX_EXECUTOR
CLAIMED_BY: current explicitly assigned CODEX_EXECUTOR session (runtime session ID unavailable; actual model/effort reported separately)
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_REQUEST_2026-10-12 -- 이미지 2장으로 local update를 만들고,
  FC1 weight update의 4비트 양자화 scale을 per-tensor와 per-channel 두 단위로 측정하여
  original(FP32 update, 원본 이미지)과의 차이를 비교한다. SGD epoch 3.
DIRECT_EXECUTION_OVERRIDE: EXPLICIT_USER_REQUEST_2026-10-12 -- 사용자가 이 세션에 작업지시서 범위의 구현·검증·실행을 직접 승인하고 CODEX_EXECUTOR 역할을 지정했다. 초안의 Planner 전달 게이트만 이 작업에 한해 대체하며 과학적 조건은 바꾸지 않는다.
EXECUTION_RECORD: quantization_aware_cpa/results_private/int4_pt_vs_pc_B2_E3_20261012/run_manifest.json

## 목적

scale은 이미지가 아니라 weight update 텐서 값에서 측정한다.
- per-tensor: s = FP32(max|ΔW1| / 7), scale 1개
- per-channel: 출력 채널(행) r마다 s_r = FP32(max_j |ΔW1[r,j]| / 7), scale 4,096개
두 양자화 update가 (a) FP32 update와 얼마나 다른지, (b) CPA→FIA 복원 이미지가 원본과 얼마나 다른지 측정한다.

## 고정 조건

- 모델: VGG16 IMAGENET1K_V1, checkpoint SHA256 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0
  (`signed_baseline.verify_checkpoint` 사용). Conv 고정, FC 6개 전체 학습, dropout 0, model.eval().
- 이미지: `quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/observation/evaluation_truth.pt`
  의 x[0], x[1]과 해당 labels. 인덱스로 고정한 선택이며 품질 기준 선택이 아니다.
- 특징: z = flatten(avgpool(features(x))), no_grad. 저장된 z[0:2]와의 최대 차이를 기록한다.
- 학습: SGD lr=0.001, momentum 0, weight decay 0, foreach=False, mean cross-entropy,
  full batch B=2, epoch 3 = 3 step (batch 하나이므로). 기존 10 step 대신 3 step으로 바꾼 것이 이번 사용자 지정이다.
- update 정의: θ_after3 − θ_start. ordinary SGD 결과와 `fc_replay.fc_sgd_updates(..., steps=3,
  create_graph=False, checkpoint_forward=False)` 결과를 6개 텐서 모두 torch.equal로 대조한다.
- 결정론: `signed_baseline.configure()` (TF32 off, deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8), seed 42.

## 양자화 (FC1 `classifier.0.weight` [4096, 25088])

- 공통: symmetric signed INT4, code ∈ [-7, 7], q = round(G/s).clamp(-7,7) (round-half-to-even), 복원값 q·s, CPU FP32.
- per-tensor: `quantized_update_cpa.cell_utils.encode` 그대로 사용.
- per-channel: s_r = absmax_r / 7, absmax_r = 0인 행은 s_r = 1 (q=0). 그 외 규칙은 per-tensor와 동일.
- CPA는 FC1만 사용하므로 나머지 FC 텐서는 양자화하지 않아도 공격 입력이 같다. 통신량 보고용으로만
  scale 개수/바이트를 기록한다 (per-tensor 4 B, per-channel 16,384 B).

## Arm

1. FP32 (기준, 양자화 없음)
2. INT4 per-tensor
3. INT4 per-channel

세 arm 모두 같은 CPA/FIA 설정·seed·초기값을 사용한다.

- CPA: `upstream_naive.run_upstream_cpa(Y_hat=ΔW1, n_components=2, config=signed_baseline.CPA_CONFIG, seed=42)`
  입력은 raw ΔW1 그대로 (부호/lr 변환 없음). 25,000 iteration.
- FIA: `upstream_naive.run_upstream_fia(target_z=abs(S_hat), config=signed_baseline.FIA_CONFIG, seed=42)`, 25,000 iteration.
- 원본 `official_cpa/src` 소스는 수정하지 않는다. 실행 전후 `upstream_naive.source_hashes()` 일치를 확인한다.
- 중간 선택·조기 종료·하이퍼파라미터 조정 금지. 최종 iteration 결과만 사용한다.

## 측정

update 수준 (복원 전, FC1):
- per-channel scale 분포: 분위수(0, 1, 5, 25, 50, 75, 95, 99, 100%), s_r / s_tensor 비율 분포
- scale이 0인 행 개수
- 0이 아닌 code 비율 (per-tensor / per-channel)
- 상대 L2 오차 ‖G_q − G‖/‖G‖, 코사인 유사도 cos(G_q, G), 최대 절대 오차
- 참고: 행렬 rank 진단으로 상위 3개 singular value (FP32/PT/PC)

이미지 수준 (세 arm 모두 끝난 뒤에만 원본 열람):
- pixel-MSE Hungarian 매칭(2×2) 한 번으로 PSNR, SSIM, LPIPS(VGG, torchmetrics, `report_full_update_iterations._make_lpips_metric`) 계산
- embedding cosine: abs(S_hat) vs 원본 z, Hungarian
- 원본 2장 + 세 arm 복원 이미지 비교 그림 1장

판정 기준은 따로 두지 않는다. 표본이 이미지 2장, seed 1개이므로 차이는 서술 통계로만 보고하고 일반화하지 않는다.

## 출력

`quantization_aware_cpa/results_private/int4_pt_vs_pc_B2_E3_20261012/` (새 폴더, 기존 결과 덮어쓰기 금지)
- run_manifest.json (입력·checkpoint·소스 hash, 설정, runtime)
- update_stats.json (위 update 수준 측정)
- arm별 폴더: sources.pt, candidate.pt, cpa_result.json, fia_result.json, result.json
- comparison.json, comparison.png, summary_ko.md (간결한 한국어 요약과 한계)

## 예산

GPU 1개에서 arm을 순차 실행한다. B=2라 FIA 비용이 작아 arm당 수 분~수십 분으로 예상되지만 실측값이 아니다.
첫 arm 진행 속도로 전체 시간을 추정해 기록한다. 실패나 비유한 값이 나오면 해당 단계에서 멈추고 원인을 보고한다.
자동 재시작이나 추가 arm/seed는 하지 않는다.

## 전달

Planner가 검토 후 READY로 바꾸고 `executor_bridge.py`로 기존 Executor에 전달한다.
현재 `.ai/EXECUTION_HANDOFF.md`(완료된 CoRA 작업)는 이 초안이 덮어쓰지 않았다.

## 실행 결과 (2026-10-12)

- STATUS: COMPLETED. 사용자가 이 세션을 CODEX_EXECUTOR로 지정하고 직접 실행을 승인해, 초안 전달 게이트만 이 작업에 한해 대체했다. 실험 조건은 변경하지 않았다.
- 결과 경로: `quantization_aware_cpa/results_private/int4_pt_vs_pc_B2_E3_20261012/`
- 3-step ordinary SGD와 `fc_replay`의 6개 FC update가 모두 `torch.equal`로 일치했다. source `z[0:2]`와 재계산한 VGG feature의 최대 절대 차이도 `training_validation.json`에 기록했다.
- FP32, INT4 per-tensor, INT4 per-channel 각 arm이 CPA 25,000회와 FIA 25,000회를 완료했다. GPU arm 순차 총 실행시간은 3,276.1초, pipeline 완료까지 3,283.2초다.
- FC1 update: per-tensor 상대 L2 오차 0.631835, cosine 0.803866; per-channel 상대 L2 0.296708, cosine 0.957303. Per-channel zero-absmax 행은 2,387/4,096이며 scale=1 convention을 적용했다.
- 평균 복원 지표 (PSNR dB / SSIM / VGG LPIPS): FP32 12.84057 / 0.288133 / 0.456719; per-tensor 12.10495 / 0.245963 / 0.760441; per-channel 12.31299 / 0.258675 / 0.590131. 임계값이나 일반화 결론은 두지 않았다.
- 모든 candidate/source hash와 25,000회 최종 log iteration을 검증했고, original `official_cpa` source hashes는 실행 전후 일치했다. 기존 결과를 덮어쓰지 않았다.
- 환경 경고 2건을 run manifest와 summary에 기록했다: upstream CPA의 complex-to-real cast, FIA CUDA adaptive average-pooling backward의 비결정성 경고(warn-only 설정).
- 최종 검증: 세 arm 산출물·shape·finite·[0,1] pixel range·Hungarian one-to-one 매칭·비교 JSON/그림/요약 파일 모두 통과.
