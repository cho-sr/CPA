# FedAvg round 1, client 0: INT4 FC1 scale per-tensor vs per-channel, B=16, SGD epoch 3

TASK_ID: CPA_FEDAVG_R001_C0_INT4_PT_VS_PC_B16_E3_20261012
PLAN_VERSION: 1
STATUS: COMPLETED
AUTHOR: CLAUDE_REVIEWER (사용자 직접 지시, 2026-10-12)
ASSIGNED_ROLE: CODEX_EXECUTOR
CLAIMED_BY: CODEX_EXECUTOR (GPT-6-based Codex; user-authorized 2026-10-12)
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_REQUEST_2026-10-12 -- "이미지 16장 가지고 클라이언트 3개 두고
  첫번째 클라이언트 업데이트를 보고 테스트". 사용자 선택: 기존 3-client FedAvg checkpoint, round 1,
  FC-only SGD epoch 3, scale 관찰 + 복원(CPA→FIA).

## 목적

3-client ImageNet FedAvg의 round 1 global model에서 client 0이 자기 16장으로 local update를 만들 때,
FC1 weight update의 4비트 scale을 per-tensor와 per-channel(출력 행)로 측정하고,
각 양자화 update로 복원한 이미지를 원본과 비교한다.

client 0의 update는 자기 데이터와 global model만으로 정해진다. 다른 두 client는 이전 round 집계를
통해 global model에만 반영되어 있다. 이번 작업에서 client 1/2의 update는 계산하지 않는다.

## 입력 (모두 기존 파일, 덮어쓰기 금지)

BASE = `C:/Users/조상제/Desktop/fedavg_cora_transfer_20261006/fedavg_cora_transfer_20261006`

- global checkpoint: `BASE/imagenet-fedavg/runs/imagenet_pretrained_full_reset_fc_E5_R20/rounds/round_001.pt`
  SHA256 0a1e344b0835884dd450a48bb7db3439262b6c3f1ffa2d4e387a07143a3defad.
  `BASE/imagenet-fedavg/cora_common.py:load_global`과 같은 검사(round==1, trainable_scope=all,
  client_counts 3개)를 통과해야 한다. torchvision `vgg16(weights=None)` + strict load.
- client 0 이미지: `quantization_aware_cpa/results/m6_uia_fp32_B16_fia/batch00/inputs.pt`
  SHA256 24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141 (CoRA registration의
  client 0 / input_batch 0과 동일 파일). x [16,3,224,224] ImageNet 정규화, labels = metadata[i]['label'].
- 대조용: `BASE/imagenet-fedavg/runs/cora_full_r1357_c3_b16_bits84_c100_full25k/private/round001/client00/truth.pt`
  의 x가 위 입력 x와 같은지 확인만 한다.

## local 학습 (sender, 원본 이미지·label 사용은 여기서만)

- round_001 모델, model.eval() (dropout 비활성), ReLU inplace=False, Conv(features) 고정, classifier 6개 텐서 학습.
- z = flatten(avgpool(features(x))) no_grad, round_001 conv 가중치로 계산.
- SGD lr=0.001, momentum 0, weight decay 0, foreach=False, mean cross-entropy, full batch B=16,
  epoch 3 = 3 step. (CoRA probe의 10 step과 다르며 사용자 지정이다.)
- update = θ_after3 − θ_start. 여섯 텐서를 `qacpa/fc_replay.fc_sgd_updates(..., steps=3,
  create_graph=False, checkpoint_forward=False)` 결과와 torch.equal로 대조한다.
  CoRA의 `(W_before − W_after)/lr`과는 부호·배율만 다르며, CPA whitening과 abs(S_hat) adapter,
  absmax/7 scale 비율에는 영향이 없다. 이 사실을 보고서에 적는다.
- 결정론: `qacpa.signed_baseline.configure()`(TF32 off, deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8), seed 42.

## 양자화 (FC1 `classifier.0.weight` [4096, 25088])

직전 작업 `CPA_INT4_PERTENSOR_VS_PERCHANNEL_B2_E3_20261012`와 동일 규칙.
- symmetric signed INT4, code [-7,7], q = round(G/s).clamp(-7,7) (round-half-to-even), 복원 q·s, CPU FP32.
- per-tensor: `quantized_update_cpa.cell_utils.encode`.
- per-channel: s_r = absmax_r/7. absmax_r = 0인 행은 q=0이며 저장 규약상 s_r=1이다.
  **scale 통계에서는 이 행들을 제외하고 별도 개수로만 보고한다.** (직전 작업에서 이 행들이 분위수를 오염시켰다.)

## scale 관찰 (복원 전)

- per-tensor scale 값.
- update가 전부 0인 행 수, 활성 행 수.
- 활성 행의 s_r 분위수(0,1,5,10,25,50,75,90,95,99,100%)와 s_r/s_tensor 분위수, 최대/최소 비.
- 활성 행 중 s_r < s_tensor/14 인 행 수(per-tensor에서 행 전체가 0이 되는 행)와 그 행들의 에너지 비율.
- 행별 상대 L2 오차·0이 아닌 code 비율 분위수 (per-tensor / per-channel), FP32 행별 0이 아닌 값 비율.
- 전체: 상대 L2 오차, cosine, 0이 아닌 code 비율, 상위 17개 singular value (FP32/PT/PC).
- 그림 `scale_only.png`: (a) log10 scale 히스토그램 + per-tensor scale 선 + s/14 선,
  (b) 정렬한 행 scale 곡선, (c) 4,096행 분류 막대. 축 이름과 범례는 영어로 둔다.

## 복원 arm

1. FP32 (양자화 없음)  2. INT4 per-tensor  3. INT4 per-channel

- CPA: `qacpa.upstream_naive.run_upstream_cpa(Y_hat=FC1 update, n_components=16,
  config=qacpa.signed_baseline.CPA_CONFIG, seed=42)`, 25,000 iteration. 원본 `official_cpa/src` 수정 금지.
- FIA: target = abs(S_hat), `signed_baseline.FIA_CONFIG`, seed 42, 25,000 iteration, 원본 `feature_inversion.Direct`.
  **feature extractor는 IMAGENET1K_V1이 아니라 round_001 conv 가중치**여야 한다
  (`cora_common.RoundFeatureModel`과 같은 features/avgpool). `upstream_naive.run_upstream_fia`는
  IMAGENET1K_V1 adapter를 고정 사용하므로, 원본 Direct 호출 규약은 그대로 두고 모델만 round_001 adapter로
  넘기는 별도 wrapper를 새 파일에 둔다. 기존 파일은 수정하지 않는다.
- 세 arm은 같은 seed, 같은 FIA 초기값(hash로 확인), 같은 설정. 중간 선택·조기 종료·튜닝 금지.

## 평가 (세 arm 모두 끝난 뒤에만 원본 열람)

- pixel-MSE Hungarian 16×16 한 번 → 같은 대응으로 PSNR, SSIM, LPIPS
  (`qacpa/full_update_evaluation.evaluate_candidate`, `report_full_update_iterations._make_lpips_metric`).
- embedding cosine: abs(S_hat) vs round_001 feature z, Hungarian (`signed_baseline_evaluation.embedding_evaluation`).
- 원본 16장 + 세 arm 복원 비교 그림 `comparison.png` (16행 × 4열).
- 판정 cutoff 없음. 이미지 16장·seed 1개의 서술 비교이며 일반화하지 않는다.

## 출력

`quantization_aware_cpa/results_private/fedavg_r001_c0_int4_pt_vs_pc_B16_E3_20261012/` (새 폴더)
- run_manifest.json, training_validation.json, update_stats.json, scale_only.png
- arm별 sources.pt, candidate.pt, cpa_result.json, fia_result.json, result.json
- comparison.json, comparison.png, summary_ko.md
raw update 텐서는 저장하지 않아도 되지만, 저장한다면 private 폴더 안에만 둔다.

## 예산·중단

GPU 1개, arm 순차. 직전 B=2 작업은 세 arm 합계 3,276초였고 B=16은 FIA가 더 무거우므로 더 길 수 있다.
첫 arm 진행 속도로 전체 시간을 추정해 기록한다. 실패·비유한 값·hash 불일치는 해당 단계에서 멈추고
원인을 보고한다. 자동 재시작, 추가 round/client/seed/arm 없음.


## EXECUTION RESULT (CODEX_EXECUTOR, 2026-10-12)

- Status: FAILED during the registered local-update validation stage.
- Inputs passed SHA256 and round_001 checks; batch00 matched round001/client00 truth exactly.
- The three-step client-0 SGD calculation ran, but fc_sgd_updates validation stopped because classifier.1 ReLU remained in-place. The handoff requires ReLU inplace=False.
- No CPA or FIA arm started. No candidate, comparison, or evaluation metrics were produced.
- Official official_cpa/src source hashes were rechecked unchanged (8 tracked upstream/wrapper files); no original source or prior result was modified.
- Per the handoff stop-on-failure rule, no further run was started.
- Private execution artifacts and failure details: quantization_aware_cpa/results_private/fedavg_r001_c0_int4_pt_vs_pc_B16_E3_20261012/.


## RERUN AUTHORIZATION (CODEX_EXECUTOR, 2026-10-12)

The user explicitly authorized attempt 02. Attempt01 failure.json, prelaunch_validation_failure.json, and run_manifest.json are preserved under attempt01_failed/. The stale .run.lock was removed only after confirming its recorded PID was not alive and no Python process for this driver was running. The only model behavior fix is setting every nn.ReLU.inplace=False in compute_training_updates; all experiment parameters and arms remain unchanged.


## ATTEMPT02 RESULT (CODEX_EXECUTOR, 2026-10-12)

- Status: COMPLETED. The only model-behavior change was setting all 15 nn.ReLU modules to inplace=False; six local FC updates matched fc_sgd_updates exactly.
- round_001/client00 batch00 input SHA256, checkpoint SHA256, and image equality against the registered truth passed. All three arms completed CPA 25,000 and FIA 25,000 iterations.
- FIA initialization SHA256 was identical across arms. FIA used the round_001 convolutional feature extractor; its state hash was unchanged.
- FC1 per-tensor scale: 9.462029083806556e-06. Per-channel scales were summarized over 4,088 active rows; 8 zero rows were excluded. 151 active rows had scale below s_tensor/14; their update-energy fraction was 0.0010798613.
- FC1 update relative-L2 error / cosine: per-tensor 0.6902655 / 0.7582636; per-channel 0.3398092 / 0.9445395.
- Mean image metrics (PSNR dB / SSIM / VGG LPIPS / embedding cosine):
  - FP32: 12.291501 / 0.343760 / 0.498426 / 0.999318
  - INT4 per-tensor: 11.602848 / 0.292153 / 0.816701 / 0.609303
  - INT4 per-channel: 11.687858 / 0.295502 / 0.686431 / 0.819255
- Image metrics used one 16x16 pixel-MSE Hungarian match per arm; PSNR, SSIM and LPIPS use the same pairs. No success cutoff was applied.
- Official upstream source hashes remained unchanged. Attempt01 failure.json, prelaunch_validation_failure.json and run_manifest.json remain preserved under attempt01_failed/.
- Final artifacts are in quantization_aware_cpa/results_private/fedavg_r001_c0_int4_pt_vs_pc_B16_E3_20261012/.
