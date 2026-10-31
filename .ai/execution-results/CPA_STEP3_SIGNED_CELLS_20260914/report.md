# CPA_STEP3_SIGNED_CELLS_20260914 v1 — Executor 결과

[FACT] 승인된 Step 3 signed codec utility 구현·CPU synthetic unit test·raw metric
인터페이스 준비를 완료했다. Step 3: **COMPLETED**. Step 2:
**ARTIFACT_MISSING / NO_BASELINE_EXECUTION**. 이번 완료는 utility 검증이며
FP32 Reference 재현이나 이미지 복원 개선 결과가 아니다.

## 역할·승인·범위

[FACT] 사용자 직접 지시로 지정된 CODEX_EXECUTOR가 v1 READY 작업을 접수했다.
세션 `01a09e4d-51f3-7121-84e2-03a7a9ac3039`의 JSONL `turn_context`에서
`model=gpt-6-astra`, `effort=high`를 확인했다. CLI 버전은 0.153.4다.
접수 시각은 `2026-09-14T06:23:40.345735+00:00`이며 결과 완료 시각은
`results.json`의 `completed_at_utc`에 기록했다. 다른 Executor/AI 세션을 만들지 않았다.

[FACT] 추가한 연구 소스는 다음 세 파일이다.

- `quantized_update_cpa/cell_utils.py`: encode, dequantize, get_cells,
  project_to_cells, check_code_consistency, scale_consistency.
- `quantized_update_cpa/tests/test_cell_utils.py`: 독립 A–F 및 reference/입력 계약 테스트.
- `quantized_update_cpa/trial_metrics.py`: raw metric과 oracle 평가 영역을 분리한 JSON 인터페이스.

[FACT] 그 외 쓰기는 작업별 execution-results 디렉터리와 인계서의 Executor
접수·상태·결과 필드로 한정했다. 기존 quantizer/CPA/FIA/collector/optimizer/설정/관측/결과를
수정하지 않았다. `.ai/STATE.md`와 `.ai/RELAY.md`도 수정하지 않았다.

## Codec 및 구간의 의미

[FACT] reference는 기존 `quantized_update_cpa/quantization.py`의
`quantize_tensor_symmetric(bits=4, rounding="nearest")`를 path import하여 직접 호출했다.
common.py, VGG, 데이터셋을 import/load하지 않았다. CPU FP32 입력에 대해
tensor별 scalar `s=FP32(absmax(G)/7)`, `q=int8(clamp(round(FP32(G/s)), -7, 7))`,
`Yq=FP32(q*s)`를 동일 연산 순서로 사용한다. 영 tensor는 기존대로 `q=0,s=1`이다.
코드/입력/scale dtype은 각각 int8/float32/float32이며 shape별 기록은 results.json에 있다.

[FACT] metadata에 출처, bits/range, tensor별 absmax scalar scale, dtype, CPU,
nearest-even, 영 tensor 처리 및 finite edge 정책을 명시했다. 정확히 일치하지 않는
metadata는 거부한다. affine uint4 packet이나 per-row metadata를 signed로 재해석하지 않는다.
이 metadata는 기존 utility의 검증 범위를 나타내며 미래 실험의 codec/scale scope 결정이 아니다.

[FACT] 이상적인 실수 half-step 경계 `(q±0.5)*s`와 실제 FP32 code 소유 경계를 구분했다.
get_cells는 양의 유한 FP32 값의 정수 bit 순서에서 실제 divide/round/clamp로 이진 탐색한다.
그 결과를 음수 쪽으로 대칭시켜 양 끝이 포함되는 표현가능 구간을 반환한다. 반환할 때마다
모든 15개 code의 양 끝점과 바로 인접한 표현가능 값의 code를 실제로 검사한다.
임의 epsilon, tolerance, FP64 codec 나눗셈, finfo.eps scale clamp는 사용하지 않는다.

[FACT] 유한 cap은 관측 s만으로 구한
`max{m: m은 유한 비음수 FP32, FP32(m/7)=s}`다. 따라서 기존 `FP32(7*s)`를
cap으로 삼을 때 생길 수 있는 원본 배제를 피한다. 실제 반례:

| 항목 | 값 |
|---|---:|
| 원본 absmax, bits `0x3fe00006` | 1.7500007152557373 |
| 관측 s | 0.25000008940696716 |
| 단순 FP32(7*s) | 1.7500005960464478 |
| 반환한 finite cap | 1.7500007152557373 |

[INFERENCE] 양의 FP32 순서에서 고정-s quantizer가 단조이므로 각 code의 최초/최후
표현가능 값을 찾는 탐색은 그 사이 모든 표현가능 값의 동일 code를 보장한다.
finite cap은 가능한 absmax의 상한만 부여한다. projection은 좌표별 clamp이고,
특정 좌표를 ±7s에 고정하거나 후보의 scale 일치를 강제하지 않는다.

[FACT] code consistency는 관측 s를 고정한다. 별도 scale consistency는 후보의
reference scale, 관측 scale, 절대/상대 오차, exact equality와 지원 여부를 기록한다.
후보를 다시 encode하여 code consistency를 대체하지 않는다. scale 오차는 diagnostic이며
optimization loss, anchor, weighted score, ranking, selected index 로직은 없다.

## 테스트와 실행 근거

[FACT] 최종 **79 passed, 0 failed, 0 errors, 0 skipped**. xfail/tolerance 완화는 없다.
고정 synthetic seed는 **20260914**다. fixture는 normal (4×19), zero (2×5),
tie vector, tiny normal, 양의 subnormal scale, scale-rounding vector, FP32 최대 유한값,
scalar를 포함한다. 실제 client 관측이나 private G/x/y/z는 사용하지 않았다.

| 필수 검증 | 결과 |
|---|---|
| A: self-encoded 원본 G가 L,U 내부 | PASS |
| B: projection 결과가 L,U 내부 | PASS |
| C: projection을 관측 s로 양자화한 q가 정확히 일치 | PASS |
| D: reference Yq=s*q가 L,U 내부 | PASS |
| E: ±7/0 포함 15개 code 끝점·이웃, 14개 tie와 양쪽 이웃 | PASS |
| F: zero q=0,s=1 및 dequantization 일치 | PASS |
| reference q/s/dequantization torch.equal | PASS |
| clipping, 별도 scale 진단, metadata/입력 거부, raw schema | PASS |

[FACT] 첫 실행은 78 passed/1 failed였다. 유틸리티 계약 검사는 통과했지만,
단순 7*s cap의 실패를 입증하려던 네 입력에 해당 현상이 없어 witness assertion이 실패했다.
1.0 인근 32개 FP32 값에서는 반례가 없었고, 1.75 인근 32개 값에서 위 반례를 확인했다.
테스트 입력만 구체적 반례로 수정하고 assertion/허용오차는 유지했다.
추가로 전 code의 dequantized center containment 검사도 명시했다.
첫 실행 결과는 `pytest-first.xml`, `pytest-first.log`로 보존했다.

[FACT] 최종 실행 명령:

```powershell
& C:/miniconda/envs/DLG/python.exe -B -m pytest -q quantized_update_cpa/tests/test_cell_utils.py -o cache_dir=.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/pytest-cache --junitxml=.ai/execution-results/CPA_STEP3_SIGNED_CELLS_20260914/pytest.xml
```

[FACT] `run_validation.py`가 위 argv로 subprocess를 실행하고 exit status 0, 실제 stdout,
JUnit, runtime, Windows process memory, fixture evidence를 수집했다. 최종 테스트
wall time **1.935초**, pytest JUnit 시간 **1.362초**다. pytest 자식 process의
lifetime peak working set은 **547,037,184 bytes (약 522 MiB)**였다.
측정은 Windows GetProcessMemoryInfo의 PeakWorkingSetSize이며 GPU VRAM과 구분한다.
GPU 실행 0회, GPU peak VRAM은 null / NOT_USED_CPU_ONLY다.
사용자 지정 시간·RAM 숫자 예산은 NOT_SPECIFIED로 유지했다.

[FACT] Python 3.11.14, torch 2.13.0+cu132, numpy 2.4.4, pytest 9.1.1,
CPU/default float32에서 수행했다. CUDA 제공 torch 빌드지만 GPU 연산은 하지 않았다.
원래 flush-denormal 설정을 바꾸지 않았고 양의 subnormal 동작은 테스트로 확인했다.

## 지원 범위와 제한

[FACT] 지원 입력은 nonempty finite CPU FP32 tensor이며, 비영 tensor는
FP32(absmax/7)>0이어야 한다. 영 tensor도 지원한다. signed int8 codes [-7,7]와
유효한 reference 출처의 양의 scalar FP32 scale을 요구한다. dtype/device를 조용히 변환하지 않는다.

[FACT] reference의 양의 absmax bit pattern 1,2,3은 `/7`에서 s=0으로 underflow한다.
`G=[-tiny,0,+tiny]`에 대한 기록된 CPU reference 결과는 q=[-7,0,7], dequantization은
모두 0이다. 중간 나눗셈에는 ±inf와 NaN이 생긴다. utility encode/get_cells는 이 입력을
명시적인 UnsupportedCodecInput으로 거부한다. reference를 고치거나 작은 scale을
epsilon으로 대체하지 않았다. 이 퇴화 사례는 정상 지원 범위의 A–F와 분리한
limitation 테스트 3개 및 results.json의 원래 결과로 기록했다.

[UNKNOWN] 다른 device/dtype/denormal 정책은 검증하지 않았다. FP32 전체 입력 공간에 대한
실험적 전수검사는 하지 않았으며 결과는 기록된 CPU runtime과 명시된 계약의 synthetic 검증이다.

## Raw metric 및 미실행 영역

[FACT] CPA objective, MI, negentropy, sparsity, sign consistency, source diversity,
quantization consistency, scale consistency를 독립 raw 필드로 둔다. 미실행 metric은
null + NOT_COMPUTED/UNKNOWN + 사유로 저장한다. 예제에는 합성 원본의 code/scale 진단만
COMPUTED로 넣었고 CPA/source metric은 계산하지 않았다. W decor를 MI라고 부르지 않는다.
NaN/Inf JSON 값은 거부하며 null과 사유를 요구한다.

[FACT] evaluation 영역은 embedding-oracle cosine과 image-oracle LPIPS/PSNR/SSIM을
분리한다. 모두 null / NOT_COMPUTED다. FP32/INT4 reconstruction 수치, CPA/FIA,
VGG loading/학습, 데이터/관측 생성, private 자료 loading, oracle 평가는 미실행이다.

[FACT] Step 2의 ARTIFACT_MISSING 판정은 승인된 인계서의 조사 결과를 유지한 것이다.
현재 affine 공개 관측을 signed artifact로 재해석하지 않았으며 signed artifact 생성 CLI는
UNAVAILABLE로 기록했다. 새로운 baseline/관측 수집은 수행하지 않았다.

## 보존과 Planner 인계

[FACT] 실행 전후 SHA256이 일치한다.

- quantization.py: `ae0f812c7de3de07cce3796c701475d6fc3e0a822f7d6dd8ab951d34201e3ceb`
- STATE.md: `7d28f690b6364b791e14c4ab8164ec708cc88201485e956081594a5278c6e446`
- RELAY.md: `003e78b6a78a5b967644e709b8dfedc1cba8a97a76bb289baef18428612426e2`

[FACT] 원시 결과·신규 소스 hash·조건은 `results.json`, 실제 실행 로그는 `pytest.log`,
JUnit은 `pytest.xml`, runtime은 `runtime.json`, raw schema 예제는
`raw_metrics_example.json`, 접수 근거는 `claim.json`에 있다.

[PROPOSAL] 추가 구현·실험 제안은 이번 Executor 범위에서 만들지 않았다.
Planner는 이 결과를 독립 검토할 수 있다. Step 2 artifact 부재와 복원 성능은 해결됐다고
보고하지 않으며, 승인된 utility 검증 완료 후 실행을 종료한다.
