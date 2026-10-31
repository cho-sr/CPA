# FedAvg round 1, client 0: per-channel INT4, top-10% scale rows random cell sampling × 100

TASK_ID: CPA_FEDAVG_R001_C0_PC_TOP10_RANDOM100_20261013
PLAN_VERSION: 1
STATUS: DRAFT_AWAITING_USER_DIRECT_DELIVERY
AUTHOR: CLAUDE_REVIEWER (사용자 직접 지시, 2026-10-13)
ASSIGNED_ROLE: CODEX_EXECUTOR
CLAIMED_BY: none
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_REQUEST_2026-10-13 -- "A에 대해서 상위 10%에 대해서 진행해줘",
  범위 선택 "Full CPA+FIA for 100". per-channel INT4에서 scale 상위 10% 행만 cell 내부 random sampling,
  나머지 행은 naive(q×s), candidate 100개 모두 CPA 25k + FIA 25k.

## 배경

선행 작업 `CPA_FEDAVG_R001_C0_INT4_PT_VS_PC_B16_E3_20261012` (결과:
`quantization_aware_cpa/results_private/fedavg_r001_c0_int4_pt_vs_pc_B16_E3_20261012/`)과
행 선택 분석 `.../hybrid_row_selection/row_selection.json`을 잇는다. 기준 A(상위 10%)는 활성 4,088행 중
409행을 고르며, 이 행들이 예상 양자화 오차(s_r² 기준)의 32.8%를 차지한다.

## 고정 조건 (선행 작업과 동일, 변경 금지)

- global model `round_001.pt` (SHA256 0a1e344b0835884dd450a48bb7db3439262b6c3f1ffa2d4e387a07143a3defad),
  load_global과 같은 검사, 모든 nn.ReLU inplace=False, model.eval(), Conv 고정.
- client 0 이미지 `quantization_aware_cpa/results/m6_uia_fp32_B16_fia/batch00/inputs.pt`
  (SHA256 24995487511a3e9d8d4665fdcd704f7a0f792249a9ba79880638d13bc4740141), labels = metadata label.
- SGD lr=0.001, momentum 0, wd 0, foreach=False, mean CE, B=16 full batch, 3 step. update = θ_after3 − θ_start.
- 재현 확인: FC1 update의 FP64 L2 norm == 0.014023203019876779, FP32(absmax/7) == 9.462029083806556e-06.
  불일치하면 멈춘다.
- CPA: `upstream_naive.run_upstream_cpa(n_components=16, config=signed_baseline.CPA_CONFIG, seed=42)`, 25,000회.
- FIA: abs(S_hat), `signed_baseline.FIA_CONFIG`, seed 42, 25,000회, 원본 Direct,
  **round_001 conv feature adapter** (선행 작업의 wrapper 재사용 가능, 기존 파일 수정 금지).
- 모든 candidate가 같은 CPA seed, 같은 FIA seed, 같은 FIA 초기값을 쓴다. 바뀌는 것은 FC1 입력 행렬뿐이다.

## 양자화와 행 선택 (공개 정보만 사용)

- FC1 `classifier.0.weight` [4096, 25088], per-channel(출력 행) symmet