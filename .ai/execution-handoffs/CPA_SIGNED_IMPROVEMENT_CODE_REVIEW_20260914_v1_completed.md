# Signed INT4 개선 후보 코드 검토 인계

TASK_ID: CPA_SIGNED_IMPROVEMENT_CODE_REVIEW_20260914
PLAN_VERSION: 1
STATUS: COMPLETED
TASK_KIND: READ_ONLY_CODE_AND_MATH_REVIEW
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: gpt-5.6-luna
REQUIRED_REASONING_EFFORT: xhigh
CLAIMED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_CODE_REVIEW_ONLY_ON_EXISTING_LUNA_XHIGH
CONSENSUS_REFERENCE: USER_REQUEST_IN_CURRENT_PLANNER_CONVERSATION_20260914
DELIVERY_STATUS: REVIEW_REPORT_WRITTEN
RESULT_REPORT_PATH: .ai/execution-results/CPA_SIGNED_IMPROVEMENT_CODE_REVIEW_20260914/report.md
COMPLETED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
PLANNER_SESSION_ROLE: CODEX_PLANNER
MODEL_DECISION: User explicitly approved existing Luna/xhigh CLI for this read-only review only

## [FACT] 요청과 전달 상태

사용자 요청: "검토하는 코드 codex 에게 넘겨".
직전 요청은 기존 결과에서 개선 여지가 있는 부분을 찾는 것이었다. 따라서 이번 인계는
이미 정리한 개선 후보의 코드·수학 검토에 한정한다. 새 연구 구현/테스트/실험은 승인하지 않았다.

기존 지정 Codex thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039, PID18272에 대해
2026-09-14T12:38:44Z executor_bridge --status를 실행했다. 현재 runtime 근거는
logs_2.sqlite row5741830의 gpt-5.6-luna / xhigh이며, Astra/high 요구와 불일치하여
bridge 검사에서 거절돼 당시 전송 명령/queue는 실행하지 않았다.

이후 Planner가 "현재 Luna / xhigh로 설정된 기존 Codex 창에 코드 검토만 넘길까요?"라고
질문했고 사용자가 "넘겨"로 명시적으로 승인했다. 따라서 이 한 건의 read-only 검토는
동일한 기존 thread/PID의 Luna/xhigh 설정으로 전달한다. 모델이나 전역 역할 지침은 바꾸지
않으며, 연구 구현·실험을 Luna에 허용한 것으로 확대하지 않는다. 현재 runtime과 저장된
thread 설정의 일치, 정확한 PID/시작 시각/cwd 및 queue 가능 상태를 재확인했다.
기존 bridge는 Astra/high를 고정 검사하므로, 이 명시적 예외 작업은 같은 codex queue를
사용하는 1회 전달로 기록한다. 중복 방지·원문 snapshot·전송 영수증은 그대로 보존한다.
이전 완료된 baseline 지시서는 completed archive로 보존한 뒤 active handoff에 이 검토를 배치한다.

## 유지 조건과 허용 범위

- 원본 CPA 코드·whitening·loss·optimizer와 기존 baseline/output을 변경하지 않는다.
- 현재 signed INT4는 deterministic nearest-even [-7,7], FC tensor별 FP32 max_abs/7 scalar다.
  과거 affine per-row 실험과 구분한다. codec/scope/batch/client/training/threat model은 유지한다.
- Conv frozen, FC 전체 학습, single client, B16/T10, dropout0, SGD .001, momentum/decay0.
- 실제 attacker observation은 q,s. GT G/x/z/y는 후보 생성·선택·ranking에 사용하지 않는다.
- midpoint attraction 및 scale optimization loss 금지. scale은 별도 metric이다.
- MP threshold, iid Uniform quantization error, dither 가정은 사용하지 않는다.
- 허용: 기존 코드·문서·완료된 JSON/log 읽기, 수학적 검토, 검토 보고서 작성.
- 금지: 연구 코드 수정, 테스트 실행, 새 metric/행렬·tensor 진단 계산, 관측 생성,
  학습·CPA/FIA·SGD replay·benchmark, 새로운 worker/감시 루프/실험 예약.
- active handoff에 기록된 read-only 검토 과제다. 기존 완료 실험을 재개하지 않는다.

## 읽을 자료

주 제안: .ai/proposals/CPA_SIGNED_IMPROVEMENT_REVIEW_20260914.md

관련 구현:
- quantization_aware_cpa/qacpa/signed_baseline.py:33, 127, 220, 261
- quantization_aware_cpa/qacpa/signed_baseline_observation.py:22
- quantization_aware_cpa/qacpa/upstream_naive.py:204, 267
- quantization_aware_cpa/qacpa/uia.py:183, 237
- quantization_aware_cpa/qacpa/fc_replay.py
- quantization_aware_cpa/qacpa/run_deterministic_hybrid.py:132, 180, 235
- quantized_update_cpa/cell_utils.py
- quantized_update_cpa/quant_aware_ica.py (legacy reference only; canonical과 혼동 금지)
- official_cpa/src/gradient_inversion.py:113, 163, 218
- official_cpa/src/attack.py:118
- official_cpa/src/feature_inversion.py:Direct

기존 결과:
- .ai/execution-results/CPA_SIGNED_BASELINE_B16_T10_20260914/report.md 및 results.json
- quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive/raw_metrics.json
- quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/reference/raw_metrics.json
- quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/evaluation/comparison.json

원 논문: https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.pdf
FC의 선형 혼합·ReLU sign/sparsity prior 근거와 새 제안의 성능 보장을 구분한다.

## 검토 질문

1. FIA target adapter: 기존 abs(S)와 행별 부호 선택 후 ReLU의 차이를 검토한다.
   부호 선택은 관측으로 얻은 S만 사용한다. 기존 abs도 원 코드의 방식이므로 버그로 단정하지 않는다.
   반대 부호가 항상 noise인지, 부호 에너지 tie/zero 처리, source norm/순열 보존,
   정보 손실과 target diversity, GT 없이 적용 가능한지 검토한다. CPA 원본 외부의
   별도 FIA-input 분기로 구현 가능할지와 필요한 최소 후속 검증을 제안한다.

2. 첫 FC 공동 구조: 정확한 산술에서 DeltaW1=A Z, Deltab1=A*1_16, Z>=0가
   현재 고정 Conv/동일 batch/SGD10에 성립하는지 확인한다. A는 signed이며 shape[4096,16],
   Z는 [16,25088]이다. 실제 FP32 잔차 때문에 exact rank16 artifact라고 주장하지 않는다.
   bias 관측을 사용하면 label 없이 추가 제약이 되는지, 남는 식별 불가능성을 설명한다.

3. 외부 cell 전처리: min ||V-AZ||_F^2 subject to V in observed weight cells, Z>=0
   (A는 부호 제약 없음)의 실제 gradient/no-op/zero-factor stationary point/zero-loss
   다중해 문제를 검토한다. 최종 V와 AZ의 rank·cell 조건을 구분한다. bias cell 안의 v_b와
   v_b≈A1 관계는 별도 ablation 후보이며, tensor별 서로 다른 scale을 하나로 합치지 않는다.
   midpoint/scale loss는 넣지 않는다. fixed observed-scale q 일치와 scale 재계산 일치도 구분한다.

4. CPA가 이미 centered top16 whitening을 수행하므로, 단순 rank16 SVD 절단(+원 mean 복원)이
   CPA 출력에 본질적 변화를 주지 않을 수 있다는 반례를 검토한다. retained subspace 또는
   whitened input의 변화를 보지 않고 개선을 주장하지 않는다. full-rank ICA 역변환 no-op도 제외한다.

5. 향후 signed FIA update-consistency: 비공개 label 추정 규칙이 없는 현재 상태의 한계,
   이전 affine soft/hard 결과와의 구분, 전체 B16 공동 replay, all6 code/bin/scale metric의
   조건을 검토한다. GT label로 빈칸을 채우거나 이미지별로 독립 순위를 매겨 섞지 않는다.
   임의 weighted score 또는 oracle 선택 정책을 만들지 않는다.

6. legacy quant_aware_ica.py의 별도 eigh/재whitening/변경 목적함수를 새 canonical
   CPA처럼 재사용하면 원본 보존 조건과 충돌하는지 확인한다. 기존 결과는 보존한다.

## 보고와 종료

검토 결과만 .ai/execution-results/CPA_SIGNED_IMPROVEMENT_CODE_REVIEW_20260914/
report.md 및 필요시 review.json에 새로 저장한다. 연구 코드와 기존 결과를 수정하지 않는다.
각 후보를 ACCEPT / REVISE / REJECT / NEEDS_TEST로 판정하고, 실제 파일·줄 근거,
Fact / Inference / Proposal / Unknown을 구분한다. 후보별 기대효과·실패 가능성·미확정 조건,
후속 실험의 최소 설계와 예상 비용을 제안하되 실행하지 않는다.

본 검토가 완료되면 일반 final 답변으로 보고하고 종료한다. 다른 AI 호출·반복 감시·후속 연구를
시작하지 않는다. Planner는 전달 상태를 한 번 확인하고 턴을 종료하며 검토 완료를 기다리지 않는다.
