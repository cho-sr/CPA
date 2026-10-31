# FP32 Reference / signed INT4 naive — B16 T10 비교 실행

TASK_ID: CPA_SIGNED_BASELINE_B16_T10_20260914
PLAN_VERSION: 1
STATUS: COMPLETED
ASSIGNED_ROLE: CODEX_EXECUTOR
REQUIRED_MODEL: Astra
REQUIRED_REASONING_EFFORT: high
CLAIMED_BY: 01a09e4d-51f3-7121-84e2-03a7a9ac3039
USER_EXECUTION_AUTHORIZATION: EXPLICIT_USER_RUN_PAIRED_BASELINE_AFTER_STEP3
CONSENSUS_REFERENCE: USER_REQUEST_IN_CURRENT_PLANNER_CONVERSATION_20260914
DELIVERY_STATUS: RECEIVED_AND_CLAIMED_EXISTING_EXECUTOR
PLANNER_SESSION_ROLE: CODEX_PLANNER

## [FACT] 승인 근거 및 범위

사용자는 이전에 기존 Astra high 창으로 인계하고 Planner가 자동 전달하도록 지정했다.
Step3 signed utility는 해당 실행 창에서 79개 CPU 테스트를 통과하고 완료했다.
Planner가 다음 작업을 '동일 조건의 symmetric 관측 생성 → FP32 Reference와 INT4 baseline
비교 실험'으로 설명한 직후 사용자가 **'실험 해저ㅜ'**라고 직접 실행을 요청했다.
이 최신 요청은 이번 새 관측 생성과 두 baseline 비교를 승인한다. 과거 Step2의
artifact missing/생성 금지는 이전 utility-only 작업의 종료 상태로 보존한다.
동일 범위의 재승인을 요구하지 말고 구현·검증·실행·최종 평가까지 완료한다.
이번 새 계획에 Claude가 동의했다고 주장하지 않는다. 사용자 결정과 명시 요청이 기준이다.

완료 기준: 동일한 1개 client/batch00의 raw update를 한 번 수집하여 FP32 Reference와
signed INT4 naive를 각각 official CPA→Direct FIA로 복원하고, 전체16개를 종료 후 평가한다.
새 quantization-aware CPA/ICA 최적화, random ensemble, MP, FIA update-consistency loss,
ranking, 추가 seed/batch/arm은 실행하지 않는다.

## [FACT / PROPOSAL] 고정 입력과 새 관측

workspace: `C:/Users/조상제/Desktop/CPA`
Python: `C:/miniconda/envs/DLG/python.exe`
source truth: `quantization_aware_cpa/results_private/deterministic_fc_hybrid_pilot_B16_T10/evaluation_truth.pt`
expected SHA256: `9430257f1142c81f12a6610bb13f6458c50c26221538019e4140246367c26eb4`
checkpoint: `C:/Users/조상제/.cache/torch/hub/checkpoints/vgg16-397923af.pth`
expected SHA256: `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`

시작 시 실제 파일 hash를 확인한다. 원본 데이터 대체나 재샘플링을 하지 않는다.
기존의 ImageNet 정규화 FP32 [16,3,224,224] 이미지와 int64 labels [16]을 sender에서만 읽는다.
- VGG16 Conv 고정, FC 전체 6개 parameter 학습; dropout 0; model.eval().
- 동일 full batch의 순서 0..15를 10회 반복한다. SGD lr=.001, momentum=0,
  weight_decay=0, foreach=False, cross_entropy mean. seed42.
- 새 글로벌 round나 여러 client를 시뮬레이션하지 않는다. 이번 관측은 하나의 local update다.
  원본에서 숫자 round/client id가 기록되지 않은 것은 UNKNOWN/NOT_RECORDED로 표시한다.
- raw update 정의는 `Delta = theta_after10 - theta_start`.
- `deterministic_hybrid_setup.py:collect`의 기존 모델 로딩/Conv freeze/ordinary FC SGD와
  fc_replay의 raw 6개 delta torch.equal 검증 구조를 재사용하되 affine exporter를 호출하지 않는다.
- sender의 원본 labels 사용은 실제 local training 생성에 한정한다. 공격이나 image update
  consistency replay에 GT labels를 넘기지 않는다.

새 signed 관측은 **검증 완료한 기존 signed per-parameter-tensor codec**으로 생성한다:
`quantized_update_cpa/cell_utils.py` 및 원 reference
`quantized_update_cpa/quantization.py:quantize_tensor_symmetric(bits=4,rounding="nearest")`.
CPU FP32 tensor별 scalar s=maxabs/7, FP32 G/s, nearest-even, q int8 [-7,7], decode FP32 q*s.
zero tensor는 기존대로 q0/s1. 임의 epsilon clamp/FP64 codec 연산으로 바꾸지 않는다.

이는 이번에 승인된 **별도의 signed 관측**이다. 기존 STATE에 기록된 affine per-row radius
packet과 같은 codec/scale scope라고 주장하거나 기존 packet을 재해석하지 않는다.
새 manifest에 per_parameter_tensor_absmax_scalar 및 codec provenance를 명시한다.
원래 affine packet·실험·fixed-condition 기록을 변경하지 않는다.

공개 bundle에는 q,s,shape,public training/checkpoint metadata만 저장한다. 별도 max_abs,
private error 통계, x/y/z/rawG를 포함하지 않는다. 실제 4-bit payload를 저장한다면
signed code q를 q+7인 nibble 0..14로 저장하고 15를 reserved로 검증하는 새 wire ID를 사용한다.
이 저장 mapping은 수치 quantizer 변경이 아니다. 기존 affine wire ID를 재사용하지 않는다.
4bit code bytes, scalar-scale bytes, header/manifest bytes와 저장파일 크기를 구분해 보고한다.

새 private collector 출력에는 raw6FC Delta, oracle features, x/labels 및 sender 검증을 저장한다.
FP32 Reference용 rawG와 evaluation-only 데이터를 공개 attacker 입력과 분리한다.

## [PROPOSAL] 두 arm의 동일 실행 계약

1. **FP32 Reference**: private reference FC1 raw Delta → official CPA → abs(S_hat) → Direct FIA.
   compression 이전의 정보가 많은 reference이며 quantized attacker 성능으로 부르지 않는다.
2. **INT4 naive**: 공개 q,s에서 Yq=s*q → official CPA → abs(S_hat) → Direct FIA.
   이 프로세스의 입력/API는 private truth/raw update 파일을 받지 않는다.

attack layer는 classifier.0.weight, shape [4096,25088], source count16.
사용자의 Step2 수식과 legacy 01/02 raw-update 기준에 따라 두 arm 모두 **raw Delta/raw sq를
그대로** CPA에 넣는다. 부호 반전, -Delta/lr, T/B 나눗셈, 추가 정규화를 하지 않는다.
과거 affine/M6의 -Delta/lr 관측과 다르므로 과거 결과의 수치 재현으로 주장하지 않는다.

CPA는 `qacpa.upstream_naive.run_upstream_cpa`와 기존 official class를 그대로 사용한다.
- n_components16; iterations25000; seed42; Adam lr.001; scheduler none.
- decor5.3, T7.7, nv.13, L1=5, tv.1, ne1; log_every5000.
- 기존 official 코드의 NE=0 덮어쓰기 동작을 수정하지 않는다.
- whitening: cov/(d-1), torch.linalg.eig, top abs eigenvalues. CPU eigh/MP로 교체하지 않는다.
- arm별 별도 input fingerprint/whitening cache를 사용한다. FP32 whitening을 INT4에 재사용하지 않는다.
- source adapter abs(S_hat), 복원 순서 그대로. true-z permutation/scale 보정을 FIA 전에 하지 않는다.

FIA는 `qacpa.upstream_naive.run_upstream_fia`의 기존 Direct 구현을 재사용한다.
- 전체16개, iterations25000; seed42; Adam lr.1; cosine scheduler; FI1, TV.1; log_every5000.
- loss=1-cos(f(x),z_hat)+.1*TV(x); gm0; labels 불필요.
- 시작 checkpoint의 frozen feature와 adapter의 IMAGENET1K_V1 checkpoint hash가 같은지 확인한다.
- 양쪽 image 초기 seed/수치 runtime 설정을 일치시킨다. adapter가 실제로 설정하는 cudnn
  benchmark/deterministic 등을 실행 시점에 기록한다. 설정되지 않은 결정성을 주장하지 않는다.
- 최종 iteration만 평가한다. GT·LPIPS·source cosine으로 checkpoint나 candidate를 고르지 않는다.

## 평가와 산출물

공개 attacker 작업/최종 후보를 저장·hash한 뒤 **별도 offline evaluation 프로세스**에서
private 자료를 읽는다. FP32 Reference 프로세스에서 나온 oracle metric을 naive arm에 전달하지 않는다.
- embedding-oracle: abs source와 oracle feature의 cosine, 최종 Hungarian matching만 사용.
- image-oracle: 최종16개 이미지에 pixel-MSE Hungarian을 한 번 적용하여 같은 대응으로
  PSNR/SSIM/VGG LPIPS를 계산한다. image matching을 embedding matching과 혼동하지 않는다.
- `full_update_evaluation.py`의 image matching 방식과 기존 LPIPS evaluator를 재사용한다.
  labels를 요구하는 기존 wrapper 때문에 GT labels/dummy label accuracy를 공격에 추가하지 않는다.
  필요한 이미지 평가 helper를 분리하여 labels 없는 후보를 평가할 수 있게 한다.
- 최종 모든16개의 metric 및 평균, paired naive-minus-reference 차이를 저장한다.
- FP32 원본 update error는 evaluation-only로 계산한다.
- raw CPA objective/NE/L1/NV, 가능한 source diversity, q/bin/scale fit을 기록하고,
  MI 등 구현되지 않은 metric은 null+이유로 둔다. 새 weighted score/ranking을 만들지 않는다.
- source ordering, sign adapter, 모든 configuration/seed/source/input/output hashes,
  q/s shape·dtype·scale scope, runtime 및 CUDA peak allocated/reserved를 기록한다.
- 1 batch/seed 결과다. 성공 threshold/유의성/일반화는 임의로 정하지 않는다.
- 원본·FP32 Reference·INT4 naive를 최종 evaluation matching으로 표시한 전체16개 PNG와
  간단한 비교 report를 저장한다. 비교 그림은 평가 산출물이며 공격 입력이 아니다.

public/code output: `quantization_aware_cpa/results/signed_int4_baseline_B16_T10/`
private/reference/evaluation: `quantization_aware_cpa/results_private/signed_int4_baseline_B16_T10/`
job logs/results: `.ai/execution-results/CPA_SIGNED_BASELINE_B16_T10_20260914/`
새 폴더만 사용한다. 기존 파일 덮어쓰기/삭제를 하지 않는다.

## 구현과 검증의 허용 범위

필요한 최소 신규 모듈과 tests/CLI를 작성한다. 예시 위치:
- `quantization_aware_cpa/qacpa/signed_baseline_observation.py`: signed public/private exporter/reader.
- `quantization_aware_cpa/qacpa/signed_baseline.py`: collector/reference/naive/evaluate 각 독립 subcommand.
- `quantization_aware_cpa/tests/test_signed_baseline.py`: codec wire roundtrip/metadata rejection,
  public-private 경계, 두 arm 설정 동등성, truth 없는 FIA 연결 및 평가 분리 검증.
- 필요한 선언된 raw metric 인터페이스 및 위 새 출력 폴더.

위 이름은 구현 파일명 계획이며 현재 존재하는 CLI인 것처럼 보고하지 않는다.
원 quantizer, cell_utils의 수치 contract, official CPA/FIA, 기존 affine collector를 변경하지 않는다.
공유 helper 수정이 꼭 필요하면 기본 동작을 보존하는 최소 optional 경로로 제한하고 변경 근거를 기록한다.
구현상 수정/실패 진단 및 필요한 집중 테스트는 이번 승인 범위에 포함된다.

실행 순서: 현재 runtime/승인/새 job 미접수 확인 → claim → 입력·소스 hash와
설정 manifest 사전 저장 → 작은 관련 tests → 새 sender 수집·검증·분리저장 →
naive 공개 arm → 별도 FP32 Reference arm → 후보 고정 후 offline 평가 → 결과 보고.
각 실행 명령과 exit code를 job 로그에 보존한다. 수집은 ordinarySGD/replay 검증 각1회,
CPA는 두arm 각25000, FIA는 두arm 각25000이다. 실패/중단 시 같은 설정으로만 복구하고
완료된 arm은 재실행하지 않는다. 임의 iteration 축소/추가 sweep/추가 batch는 하지 않는다.

resource: 현재 사용 가능한 CUDA GPU 한 개에서 순차 실행. model/data 조건은 고정한다.
FIA 2x25000은 시간이 걸릴 수 있다. 초기 실제 처리속도로 남은 시간을 기록한다.
사용자 지정 walltime/VRAM 숫자 상한은 UNKNOWN. 필요 memory optimization은 동일
수학·batch·iteration을 유지하는 경우에만 적용하고 양쪽에 동일하게 반영한다.
NaN/수치 실패/OOM은 기존 설정을 조용히 바꾸지 말고 근거와 함께 NEEDS_REVIEW로 보고한다.
새 목적함수/NE 활성화/MP/반복수 sweep/무관한 모델 학습은 금지한다.

## Executor claim / result

실제 설정 근거: current turn_context gpt-6-astra/high; thread 01a09e4d-51f3-7121-84e2-03a7a9ac3039; codex.exe PID18272; snapshot SHA256 verified
시작/종료: 2026-09-14T08:09:03.763936+00:00 / 2026-09-14T09:33:54.548840+00:00
진행 로그: .ai/execution-results/CPA_SIGNED_BASELINE_B16_T10_20260914/
수집/naive/reference/evaluation 상태: COMPLETED / COMPLETED / COMPLETED / COMPLETED
결과: COMPLETED; .ai/execution-results/CPA_SIGNED_BASELINE_B16_T10_20260914/report.md 및 results.json; all16 offline 평가 완료; 최종 Planner queue 전달 FAILED(no rollout found), 1회 시도 후 재전송 없음

READY를 확인하고 CLAIMED_BY에 정확한 세션 ID를 기록하여 RUNNING으로 전환한다.
현재 최신 .ai/EXECUTION_HANDOFF.md가 이 job/v1인지 확인하고 계획 본문은 유지한다.
완료한 경우만 COMPLETED로 표시하고 report.md/results.json/실제 수치·한계를 보고한다.
원래 Step3 완료 지시서는 .ai/execution-handoffs/에 별도 보존되어 있다.
