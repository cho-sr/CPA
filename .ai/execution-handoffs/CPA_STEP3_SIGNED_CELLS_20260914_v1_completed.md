# Astra high 실행 작업지시서 — signed codec utility

TASK_ID: CPA_STEP3_SIGNED_CELLS_20260914
PLAN_VERSION: 1
STATUS: COMPLETED
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: Astra
REQUIRED_REASONING_EFFORT: high
CLAIMED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
USER_EXECUTION_AUTHORIZATION: EXPLICIT_STEP2_STEP3_WITH_LATEST_SCOPE_DECISIONS
CONSENSUS_REFERENCE: USER_DECISIONS_IN_CURRENT_CODEX_PLANNER_CONVERSATION_20260914
STEP2_STATUS: ARTIFACT_MISSING_NO_BASELINE_EXECUTION
STEP3_STATUS: COMPLETED
DELIVERY_STATUS: RECEIVED_USER_DIRECT_AND_CLAIMED
PLANNER_SESSION_ROLE: CODEX_PLANNER

## [FACT] 사용자 승인과 최신 범위

사용자는 이 대화에서 먼저 다음과 같이 지시했다.
> 아래 결정사항에 따라 Step 2~3까지만 진행하라.
> 아직 전체 quantization-aware CPA를 구현하지 마라.

후속으로 두 사항을 직접 확정했다.
> 기존 Astra high 실행 창에 인계
> symmetric 목표 유지: 기존 signed codec utility만 검증하고 Step 2는 artifact missing 처리

따라서 이번 READY 작업은 **기존 signed codec utility 구현·CPU unit test·raw metric
인터페이스 준비**다. Step 2를 위해 관측을 새로 생성하거나 FP32 Reference/CPA/FIA를
실행하지 않는다. 동일 범위의 실행 승인을 다시 요구하지 않는다.
이 작업에 관한 새 Claude 합의가 있었다고 주장하지 않는다. 최신 사용자 결정이 기준이다.

## [FACT] Step 2 및 실제 artifact 조사 결과

현재 조건의 공개 관측:
`quantization_aware_cpa/results/deterministic_fc_hybrid_pilot_B16_T10/observation/manifest.json`

- Conv 고정, FC 전체 학습, single client, B16/T10, SGD LR 0.001, momentum/weight_decay 0,
  dropout 0, 시작 checkpoint와 학습 규칙 공개, labels 비공개.
- 실제 codec은 `uniform_affine_uint4_nearest_even_v1`. q 범위 0..15,
  FP32 row radius r, decode = r*(2*q/15-1). signed [-7,7]의 s*q와 다르다.
- classifier.0.weight q shape (4096,25088), 해제 dtype uint8,
  radius shape (4096,), 실제 packet 내부 float32. 별도 s는 저장하지 않는다.
- weight는 per-row, bias는 단일 row다. 관측·metadata를 signed codec으로 재해석하지 않는다.
- private sender_validation.json에는 raw_updates_saved:false가 기록되어 있다.
  현재 수집물의 전체 FC raw FP32 update는 저장되지 않았다.
- signed 관측 후보는 존재하나 Conv도 학습한 B8/T10 또는 B2/T1 등이다.
  mixed_bit의 obsclient0 NPZ도 다른 client codes를 포함하므로 단일 client 공개 입력으로
  전체 파일을 넘겨서는 안 된다. 이번 작업은 그 자료도 공격에 사용하지 않는다.
- 현재 fixed 조건을 모두 만족하는 signed artifact 및 기존 생성 CLI는 확인되지 않았다.
  Step 2: artifact missing. 생성 command: UNAVAILABLE. 없는 CLI option을 만들지 않는다.

향후 baseline canonical은 `official_cpa/src/gradient_inversion.py`의 eig/whitening 경로다.
기존 CPU eigh는 legacy reference로 보존한다. FP32는 `FP32 Reference`로만 구분한다.
이번에는 해당 baseline을 실행하거나 소스·cache·기존 결과를 수정하지 않는다.

## [FACT / PROPOSAL] 이번 utility 검증 reference

검증 reference는 기존
`quantized_update_cpa/quantization.py:quantize_tensor_symmetric`
함수의 `bits=4, rounding="nearest"`, CPU FP32 입력 경로로 명시한다.
이는 기존 signed utility를 검증하기 위한 구현상 기준이며, **artifact가 없는 미래
실험의 scale scope나 codec을 새로 결정하는 것이 아니다.**

확인된 reference 동작:
- parameter tensor 하나의 absmax로 scalar scale을 계산한다.
- 비영 tensor: FP32 s = max_abs/7, FP32 나눗셈, torch.round nearest-even,
  clamp [-7,7], q dtype int8, FP32 decode q*s.
- 영 tensor: 원 구현대로 q=0, s=1. 다른 collector의 s=0 규칙을 혼합하지 않는다.
- reference 자체는 수정하지 않는다. 실행 전후 source hash를 기록한다.
- common.py를 import하면 현재 없는 root/src를 요구한다. utility 테스트는
  quantization.py를 namespace/path import하여 그 의존을 피한다.
- reference의 부동소수점 연산 순서, dtype, 양의 작은 scale을 유지한다.
  finfo.eps clamp, FP64 나눗셈으로 변경하거나 임의 codec 보정을 넣지 않는다.

## [PROPOSAL] 허용 파일과 구현 계약

새 파일만 추가한다. 동일 경로에 다른 작업 파일이 있으면 먼저 확인하고 덮어쓰지 않는다.

1. `quantized_update_cpa/cell_utils.py`
2. `quantized_update_cpa/tests/test_cell_utils.py`
3. 필요시 `quantized_update_cpa/trial_metrics.py` — 아래 raw metric 인터페이스만
4. `.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/` — 실행 근거·결과
5. 이 작업지시서의 Executor 접수/상태/결과 필드

기존 quantizer, CPA/FIA, collector, optimizer, 설정, 관측, 결과, STATE/RELAY는 수정하지 않는다.
일반적인 구현 수정 및 실패 원인 확인·집중 재검증은 위 범위 안에서 진행할 수 있다.

필수 함수:

```text
encode(G) -> q,s
dequantize(q,s) -> s*q
get_cells(q,s,codec_metadata) -> L,U
project_to_cells(G,L,U)
check_code_consistency(G,q,s)
scale_consistency(G,s)
```

- encode는 위 reference와 torch.equal 수준으로 대조한다. 여기서 G는 작은 synthetic
  unit fixture다. 사용자 private G/x/y/z를 읽거나 학습하여 생성하지 않는다.
- metadata에는 검증 codec의 출처, scale scope, 입력/scale dtype, rounding 및 zero 처리를
  명시한다. affine packet이나 정체불명의 metadata를 signed로 추측하여 받아들이지 않는다.
- get_cells는 사용자 bounded edge 의도를 유지하되 실제 FP32 codec의 반올림을 반영해야 한다.
  이상적인 half-step closure와 실제 code를 소유하는 표현가능 경계를 구분·문서화한다.
- 단순 닫힌 (q±0.5)*s clamp는 nearest-even tie에서 C를 실패할 수 있다.
  실제 encoder로 끝점과 이웃을 검사하여 exact-code가 유지되는 표현가능 구간을 사용한다.
  arbitrary epsilon으로 테스트를 느슨하게 하거나 잘못된 code를 허용하지 않는다.
- FP32(7*s)가 원본 absmax와 정확히 같다고 가정하지 않는다. finite edge cap은 실제
  max_abs→scale 연산의 반올림을 고려해야 한다. 원본 max_abs를 추가 공격 입력으로 쓰지 않는다.
- 원 reference 자체의 퇴화 입력(예: 양의 absmax/7이 0으로 underflow)은 원래 결과 및
  지원 범위/실패 이유를 명시한다. reference 변경, 조용한 dtype 변경이나 보정은 금지한다.
- project는 위 구간에 대한 projection만 한다. 임의 anchor를 ±7s에 고정하지 않는다.
- check_code_consistency는 **관측 s 고정** 후 동일 round/divide/clamp를 사용한다.
  encode(candidate)처럼 candidate 자체에서 새 scale을 계산하여 대체하지 않는다.
- scale_consistency는 candidate에서 reference와 같은 방법으로 계산한 scale과 관측 scale의
  오차·일치 여부를 별도 metric으로 반환한다. optimization loss나 hard anchor가 아니다.

Raw metric 인터페이스는 CPA objective, MI, negentropy, sparsity, sign consistency,
source diversity, quantization consistency, scale consistency를 구분해 저장할 수 있게 한다.
실행하지 않은 CPA/source metric은 null과 NOT_COMPUTED 또는 UNKNOWN 사유를 저장한다.
기존 W decor를 실제 MI라고 부르거나 가짜 0으로 채우지 않는다.
weighted score, ranking, selected index 산출 로직은 구현하지 않는다.
evaluation metric은 별도 영역에서 embedding-oracle cosine과 image-oracle LPIPS/PSNR/SSIM을
구분한다. 이번에는 oracle 평가와 정답 loading을 수행하지 않는다.

## [PROPOSAL] 필수 unit tests 및 실행 순서

1. 역할·실제 runtime 설정 확인 후 이 TASK_ID/PLAN_VERSION만 claim한다.
   이미 RUNNING/CLAIMED인 같은 작업을 중복 실행하지 않는다.
2. reference source hash, 사용 Python/torch/numpy/pytest 버전, CPU/dtype를 기록한다.
3. 기존 signed reference를 직접 호출하는 작은 deterministic synthetic fixture로 구현을 검증한다.
   test seed는 고정하고 기록한다. 실제 client 관측이나 이미지 dataset을 생성하지 않는다.
4. 사용자 A–F를 각각 독립적으로 확인한다.
   A. self-encoded 원본 FP32 G가 반환된 L,U 안에 있다.
   B. projection 후 모든 좌표가 L,U 안에 있다.
   C. projection 결과를 **관측 s로** 다시 양자화하면 원래 q와 정확히 같다.
   D. 원 구현의 Yq=s*q가 L,U 안에 있다.
   E. q=+7,-7,0, rounding tie와 그 양쪽 표현가능 이웃을 검사한다.
   F. zero tensor의 q=0,s=1 및 dequantization이 reference와 정확히 같다.
5. reference 대조에는 clipping, 양의 작은 scale, FP32 scale 반올림이 드러나는 입력을 포함한다.
   A–F의 실패를 xfail/skip/tolerance 확대로 숨기지 않는다. reference의 지원 불가능한 입력은
   별도 limitation/counterexample로 구분하고 정상 지원 범위와 혼합하지 않는다.
6. 성공 또는 범위 내 해결 불가능한 문제를 확인한 뒤 결과와 source 보존 여부를 기록한다.

작업 폴더: `C:/Users/조상제/Desktop/CPA`
아래는 **위 테스트 파일 구현 후 Executor가 실행할 명령**이다. Planner는 실행하지 않았다.

```powershell
& C:/miniconda/envs/DLG/python.exe -B -m pytest -q quantized_update_cpa/tests/test_cell_utils.py -o cache_dir=.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/pytest-cache --junitxml=.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/pytest.xml
```

## 범위·자원·종료 조건

- CPU 소형 unit tests와 필요한 재검증만. VGG model loading/학습/CPA/FIA/GPU 실행은 0회다.
- 데이터/관측 생성, 기존 private 자료 loading, benchmark, 새로운 seed/arm sweep,
  예약·자동 재시작, 새로운 AI/Executor 대체 세션 생성은 허용하지 않는다.
- quantization-aware CPA, ICA→inverse 반복, whitening 변경, MP threshold,
  midpoint loss, scale loss, label replay, update-consistency reranking은 구현하지 않는다.
- 사용자 지정 시간·RAM 숫자 예산은 UNKNOWN/NOT_SPECIFIED다. 이를 이유로 승인된
  소형 CPU unit test 실행을 다시 승인받지 않는다. 실측 시간과 가능하면 process peak RAM을
  보고하고 GPU peak VRAM은 GPU 미사용으로 구분한다. 대규모 성능실험으로 확장하지 않는다.
- Step 3 A–F 및 reference 동일성 검증 후 멈춘다. 성공은 utility 검증 성공이며
  reconstruction 개선 또는 Step 2 baseline 재현 성공이 아니다.
- 종료 상태는 완료했을 때만 COMPLETED. 해결 불가능한 명세/codec 문제는 NEEDS_REVIEW로
  원인과 반례를 보고한다. Step 2 상태는 완료 후에도 ARTIFACT_MISSING으로 유지한다.

## 결과 파일 및 보고

출력 폴더: `.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/`
- `report.md`: [FACT]/[INFERENCE]/[PROPOSAL]/[UNKNOWN], 수정 파일·수학적 의미·테스트·문제
- `results.json`: Step 2 missing, codec provenance/shape/dtype, A–F와 reference 대조 결과,
  source hashes, runtime/resource, 지원 범위·limitation. NaN 대신 null+사유.
- `pytest.xml` 및 테스트 실행 로그: 실제 실행 명령·exit status·passed/failed 수 포함

FP32/INT4 reconstruction 수치, embedding-oracle, image-oracle은 이번 미실행으로 명시한다.
사용자에게 재승인을 요청하지 않고 이 승인 범위를 완료한 뒤 결과 경로를 Planner에게 보고한다.

## Executor 접수/결과 — Executor가 작성

- 실제 CLI 모델/추론 설정 확인 근거: session turn_context: model=gpt-6-astra; effort=high; cli_version=0.153.4
- 접수한 TASK_ID/PLAN_VERSION 및 세션 ID: CPA_STEP3_SIGNED_CELLS_20260914 / v1 / 01a09e4d-51f3-7121-84e2-03a7a9ac3039
- 시작/종료 시각: 2026-09-14T06:23:40.345735+00:00 / 2026-09-14T06:30:00.354545+00:00
- 실제 수행 범위 및 결과: signed cell_utils 및 raw trial_metrics 구현; CPU synthetic tests 79 passed, 0 failed/skipped; reference/STATE/RELAY hash 보존
- Step 2: ARTIFACT_MISSING_NO_BASELINE_EXECUTION
- Step 3: COMPLETED
- 산출물·검증 근거: .ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/report.md, results.json, pytest.xml, pytest.log, runtime.json; GPU/CPA/FIA/학습/관측 생성/정답 loading 미실행

## 전달 상태

이 파일은 READY 인계 문서다. 기존 Astra high CLI가 실제 수신/claim했다는 뜻은 아니다.
현재 프로젝트에는 Executor 대상으로 연결된 자동 전달 경로가 없고 기존 visible_bridge는
CODEX Planner/CLAUDE만 지원한다. 사용자가 이미 연 Astra high 실행 창에 이 문서의 경로와
TASK_ID/PLAN_VERSION을 전달하여 접수한다. Planner는 다른 CLI를 대신 시작하지 않는다.
