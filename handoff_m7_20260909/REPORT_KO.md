**M7 UIA 4bit 업데이트 매칭 대조군용 전달 보고서 — 2026-09-09**

프로젝트는 `C:\Users\조상제\Desktop\CPA\quantization_aware_cpa`, 선택한 결과는 이 프로젝트의 `results/m7_uia_4bit_B16`이다. M7 설정이 참조하는 FP32 원본은 `results/m6_uia_fp32_B16_fia`이다. 이번 작업은 기존 파일 탐색·읽기·SHA-256 검증·텐서 구조 확인·압축으로 한정했다. 실험 코드를 수정하거나 qacpa를 import하지 않았고, 학습·양자화·CPA·FIA·지표 계산을 실행하지 않았다. 전달용 보고서와 패키징 도구만 프로젝트 밖 `handoff_m7_20260909`에 새로 작성했다.

요청된 필수 파일 누락은 없다. 원본 파일 283개, 총 7,089,521,009 bytes를 수집 대상으로 확정했다. 이후 생성한 보고서와 manifest는 별도로 추가된다. 저장 기록과 현재 원본을 비교한 해시 검증 222건이 모두 일치했다. 압축 후 각 수집 파일을 ZIP 내부에서 다시 읽어 manifest의 SHA-256과 대조하며, 최종 결과와 압축 파일 크기는 `archives.json`, 압축 자체의 해시는 `SHA256SUMS.txt`에 기록한다.

**선택 근거와 탐색 범위**

작업 폴더에서 요청된 세 실행 파일 및 stage1/fia/uia/quantizer를 모두 가진 루트를 찾았다. Desktop, Documents, Downloads, PycharmProjects에서 동일한 `run_m7_uia_4bit.py`와 모델 이름을 추가 검색했으며 다른 M7 코드 사본은 검색되지 않았다. 모델은 실행 메타데이터에 기록된 사용자 torch cache에서 찾았다. Windows 전체 디스크의 모든 접근 가능 경로를 완전 탐색한 것은 아니다.

| 검증 항목 | 확인 결과 |
|---|---|
| M7/M6 최상위 summary 상태 | 둘 다 `complete` |
| M7 설정과 summary에 내장된 설정 | 전체 JSON 값 일치 |
| M7 조건 | batch size 16, local steps 10, batch00~04, 총 80장, 4bit, per_row |
| 최상위 CSV / input manifest | 각각 80행 / 80개 이미지 메타데이터 |
| 원본 이미지 중복 | 80개 경로 모두 고유 |
| 각 배치 | CSV 16행, 서로 다른 label 16개, n=16 / T=10 / bits=4 |
| 실행 완료 시각 | 2026-09-09 10:48:03.962458 UTC = 한국시간 19:48:03.962458 |
| 새 업데이트 수집 여부 | 5개 배치 모두 `update_collection_executed=false` |
| 코드 / 입력 / 체크포인트 / 결과 해시 | 등록 정보와 대조한 222건 모두 일치 |

검색된 다른 M7 이름의 결과는 `m7_consistency_audit`, `m7_followup_review`, `m7_no_cpa_control`, `m7_normalized_metrics`, `m7_perimage_distribution`, `m7_recompute_controls`, `m7_shuffle_control`이다. 이들은 후속 분석 또는 별도 대조군이며, 요청된 UIA4bit B16/T10/5batch 원실행이 아니다. 후보별 저장 상태와 `batch*/quantized.pt` 개수는 `audit.json`의 `candidates`에 보존했다. 필드가 없는 상태나 이미지 수는 미확인으로 기록했다.

**포함 파일과 압축 구조**

7개 ZIP을 모두 같은 빈 폴더에 풀면 다음 상대 구조가 합쳐진다. ZIP은 독립 파일이며 분할 ZIP 볼륨 형식이 아니다.

```text
전달받은_폴더/
  quantization_aware_cpa/
    qacpa/                     전체 소스 패키지
    configs/                   M7/M6 JSON, lock JSON 및 나머지 기존 설정
    scripts/                   기존 관련 실행·후처리 스크립트 전체
    requirements.txt
    pyproject.toml
    README.md
    results/
      m7_uia_4bit_B16/          최상위 기록과 batch00~04 전체 파일
      m6_uia_fp32_B16_fia/      최상위 기록과 batch00~04 전체 파일
      v7_m3_fallback_5batch/per_image_metrics_aggregated.csv
  codex/docs/
    M7_FOLLOWUP_ANALYSIS_PROTOCOL.json
    UIA_PUBLISHED_PROTOCOL.md
  torch_home/hub/checkpoints/vgg16-397923af.pth
  handoff_docs/                이 보고서, manifest, audit, tensor inventory, 환경 기록
```

| 압축 파일 | 내용 |
|---|---|
| `01_code_config_reports.zip` | qacpa 전체 소스·configs·scripts·환경 명세·프로토콜, M7/M6 최상위 결과/로그/그림, 보고서/manifest |
| `02_batch00_m7_m6.zip` ~ `06_batch04_m7_m6.zip` | 해당 배치의 M7와 M6 폴더 전체 |
| `07_model_weights.zip` | 실제 초기 VGG16 체크포인트 |

각 M7 배치에는 요청한 `quantized.pt`, `quantized.meta.json`, `inputs.pt`, `inputs.meta.json`, `summary.json`, `rec.pt`, `source_cosines.csv`가 모두 있다. 원래 완료 해시와 FIA 경과도 검토할 수 있도록 `fia.pt`, `fia.meta.json`, `fia_log.json`, `stage1.pt`, `stage1.meta.json`, `stage1_diagnostics.json`, `mp_prediction.pt`, `mp_prediction.meta.json`, `per_image_metrics.csv`, 그림도 함께 포함했다. M6도 `observation.pt`와 메타데이터를 비롯한 배치 파일 전체를 포함했다.

`manifest.json`과 `manifest.csv`는 각 수집 파일의 원본 절대경로, 압축 내부 상대경로, byte 크기, SHA-256, 소속 ZIP을 제공한다. 새로 만든 문서는 원본 경로가 이번 전달 폴더이며 `generated=true`로 구분된다. manifest 자체는 자기 자신을 해시할 수 없어 파일 항목에서 제외하고 ZIP 해시로 보호한다. `source_manifest.json`은 생성 문서 추가 전의 원본 283개 목록이다.

**초기 모델과 학습 조건**

실제 초기 가중치는 `C:\Users\조상제\.cache\torch\hub\checkpoints\vgg16-397923af.pth`이며 크기는 553,433,881 bytes이다. SHA-256은 `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`로, M7/M6 실행 identity의 `weights_sha256`과 일치한다. 코드의 `vgg16(weights=VGG16_Weights.IMAGENET1K_V1)`와도 대응한다. 별도 초기 classifier 체크포인트의 사용은 미확인이고 사용 근거가 없다. 이번에 확보한 전체 VGG16 state_dict에 초기 classifier 가중치와 bias가 포함된다.

| 항목 | 코드·설정에 근거한 조건 |
|---|---|
| backbone | VGG16 convolutional features와 avgpool은 고정. 이미 저장한 Z를 로드하므로 local SGD 중 이미지/backbone forward를 반복하지 않음 |
| 학습 파라미터 | classifier의 모든 파라미터. `classifier.0`, `.3`, `.6` Linear의 weight와 bias, 총 6개 텐서 |
| 관측 대상 | `classifier.0.weight`, shape `[4096,25088]` |
| 초기화 | 배치마다 IMAGENET1K_V1 classifier를 새로 로드. 5개의 독립 client update, global training round 0 |
| optimizer | `torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0)` |
| 학습률 / scheduler | 10 step 동안 lr=0.001 고정, local SGD scheduler 없음 |
| loss | `F.cross_entropy(logits, labels, reduction="none")` 뒤 `.mean()`; 매 step 16장 평균 |
| train/eval | classifier는 `eval()`이지만 파라미터 `requires_grad=True`, `torch.enable_grad()`로 학습 |
| dropout | classifier `eval()`로 비활성화 |
| ReLU | `disable_inplace_relu`로 inplace 비활성화 |
| seed | 데이터 선택 0, local 학습 호출 전 `seed_all(0+batch_index)` |

근거 함수: `qacpa/uia.py::run_fc_only_fedavg_on_features`, `run_fc_only_fedavg_checkpoints`, `make_uia_observation`, `load_vgg16_classifier`; `qacpa/run_m6_uia_fp32_fia.py::run_batch`; M7/M6 설정의 `model`, `data`, `uia`.

배치 구성은 ImageNet val에서 seed 0으로 label 목록을 shuffle한 뒤 batch index에 해당하는 16개 label을 고른다. 각 label 내 이미지 선택 위치는 `(seed + 9973*batch_index + offset) % len(candidates)`이다. 저장 metadata에 실제 dataset_index, label, image_path, source_index가 있다. 학습 순서는 저장된 Z와 label 순서를 그대로 유지한다. 각 배치에서 동일한 16장 전체를 매 step 사용하고, zero_grad → 첫 Linear → 나머지 classifier → cross entropy 평균 → backward → SGD step을 10회 반복한다. nominal minibatch는 32이지만 실제 배치가 16장이므로 유효 minibatch는 16이고 분할 minibatch나 step별 shuffle은 없다. 5개 배치의 업데이트를 순차적으로 한 모델에 누적하지 않는다.

이미지는 `VGG16_Weights.IMAGENET1K_V1.transforms()`로 전처리되었고 `inputs.pt`의 x는 `[16,3,224,224]` float32이다. ImageNet mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` 정규화 좌표를 사용한다. 이번 전달에서 원본 JPEG로 전처리를 다시 계산하지 않았다.

**관측값 정의와 양자화**

`Y_delta_over_lr = (W_initial - W_after_10_local_steps) / 0.001`이다. 일반적인 모델 변화량 `ΔW = W_after - W_initial`에 대해 `Y = -ΔW/lr`이므로 부호가 반대다. 따라서 목표와 비교할 때 모델 변화량을 그대로 사용하면 부호와 크기가 모두 다르다. SGD의 평균 loss에 대한 10회의 gradient 합에 해당하며, **10으로 추가로 나누지 않는다**. 각 step 내부에 이미 배치 평균 1/16이 들어 있다.

`observation.pt["observation"]["Y_delta_over_lr"]`가 M7 양자화의 실제 입력이다. `Y_factor = B_cumulative @ Z`는 진단용 이상적 분해이고 실제 업데이트를 대신하지 않는다. 저장 fp32 가중치 차분의 수치 오차 때문에 두 텐서는 완전 동일하다고 가정할 수 없다. 코드의 상대 L2 진단 gate는 0.001이다. 이 정의와 용도는 `qacpa/run_m6_uia_fp32_fia.py::actual_observation` 및 저장된 observation 설명 필드에 명시되어 있다.

4bit 설정은 `qacpa/quantizer.py::quantize`의 `bits=4`, `rounding="stochastic"`, `scale_granularity="per_row"`이다. 각 행 i의 반경 `r_i=max_j |Y_ij|`, scale/step `delta_i=2*r_i/(2^4-1)=2*r_i/15`이며 단위는 lr로 정규화된 Y와 같다. `delta`는 4096개 출력 뉴런별 한 값이고 열 전체에 broadcast된다.

실제 stochastic 분기는 `Y_hat = Y + (torch.rand(Y.shape)-0.5)*delta`를 만든다. 그 다음 half-width보다 `min((delta/2)*1e-3,1e-5)`만큼 안쪽인 경계로 필요 원소를 투영한다. scale이 0인 행은 0이다. 명칭은 subtractive-dither observation이며, 저장물은 정수 packed 4bit 코드가 아니라 float32 Y_hat/delta다. 대조군은 기존 `quantized.pt`의 Y_hat과 delta를 그대로 사용해야 동일한 관측을 보장할 수 있다.

양자화 RNG는 CPU `torch.Generator(device="cpu")`, seed는 batch00~04 순서로 `0, 1000003, 2000006, 3000009, 4000012`이다. 이 값들은 각 배치 summary에서도 확인했다. 이 작업에서 seed를 사용해 관측을 새로 생성하지 않았다.

**FIA 조건과 기존 CPA 기준**

기존 FIA 함수는 `qacpa/fia.py::run_direct_fia`이다. frozen/eval VGG16의 features → avgpool → flatten 출력 Z에 대해 이미지만 최적화한다. Gaussian `torch.randn([16,3,224,224])`로 x_hat을 초기화하며 매 FIA 호출 seed 42로 재설정한다. Adam lr=0.1, 25,000 iterations, CosineAnnealingLR(T_max=25000), TV weight=0.1, feature inversion weight=1.0, 로그 간격 5000, 배치/chunk 16이다. Adam의 betas/eps 등은 명시 override 없이 설치된 torch 기본값을 사용한다.

목적함수는 `mean(1-cosine_similarity(target_z,z_hat,eps=1e-10)) + 0.1*TV(x_hat)`이고 TV는 가로·세로 인접 픽셀 절댓값 차이 평균의 합이다. 각 iteration 시작 및 종료 시 x_hat을 ImageNet 정규화 좌표에서 RGB [0,1] 범위에 대응하는 경계로 clamp한다. rec.pt는 역정규화한 RGB [0,1] tensor다. PSNR/SSIM/LPIPS는 기존 torchmetrics 코드로 평가하며 LPIPS는 VGG와 RGB [-1,1] 입력 규약이다. 이번에는 재평가하지 않았다.

기존 CPA는 전체 16 components, debias=false, MP truncation=false, 25,000 iterations, Adam lr=0.001이다. decor=5.3, T=7.7, ne=1.0, nv=0.13, l1=5.0, tv=0.1이다. MP c_mp=3 예측은 진단용이고 추정 source 수를 줄이지 않는다. FIA 입력 adapter는 추정 source의 abs를 취하고, 실제 source와의 절대 cosine Hungarian 대응을 이용해 permutation만 맞춘다. 참 source로 채우거나 scale을 대입하지 않는다. 새 업데이트 매칭 대조군 구현은 이번 범위 밖이며 작성하지 않았다.

재현성 설정은 `seed_all`에서 cudnn benchmark=false/deterministic=true와 deterministic algorithms warn_only=true를 요청한다. FIA 내부는 cudnn enabled=true/benchmark=true로 바꾸므로 전체 실행의 bitwise 결정성을 보장한다고 해석하면 안 된다. 다른 PC의 GPU/드라이버/CUDA 및 전체 실행 환경 일치 여부는 미확인이다.

**텐서 인벤토리와 다음 PC에서 사용할 입력**

56개 `.pt`/`.pth` 모두 `torch.load(weights_only=True,map_location="meta")`로 구조를 읽었다. 텐서 값을 출력하지 않았으며 `tensor_inventory.json`에 key, shape, dtype와 비텐서 필드의 자료형만 요약했다.

| 파일 / key | 주요 tensor shape | dtype / 역할 |
|---|---|---|
| M7 `quantized.pt/Y_hat` | `[4096,25088]` | float32 / 대조군에 사용할 고정 관측 |
| M7 `quantized.pt/delta` | `[4096]` | float32 / 행별 quantization step |
| M7/M6 `inputs.pt/x` | `[16,3,224,224]` | float32 / 정규화된 원본 이미지, 복원 평가용 |
| M7/M6 `inputs.pt/Z_evaluation_only` | `[16,25088]` | float32 / 평가·대응 확인용 참 임베딩 |
| M6 `observation.pt/observation/Y_delta_over_lr` | `[4096,25088]` | float32 / 기존 실제 FP32 업데이트 |
| M6 `observation.pt/observation/Y_factor` | `[4096,25088]` | float32 / 진단용 |
| M6 `observation.pt/observation/B_cumulative` | `[4096,16]` | float32 / 진단용 |
| M6 `observation.pt/observation/final_logits` | `[16,1000]` | float32 |
| `rec.pt` | `[16,3,224,224]` | float32 / 기존 복원 결과 RGB [0,1] |

labels와 이미지 순서는 inputs의 `metadata`에 있다. `Z_evaluation_only`, x, B_cumulative, Y_factor의 존재는 새 대조군의 추정 입력으로 이들을 쓰도록 허용하는 실험 정의가 아니다. 목적에 맞는 매칭 입력은 저장된 양자화 관측이고, 나머지 파일은 초기 조건 확인·평가·진단에 사용할 수 있다.

**환경과 이식 시 경로 처리**

실행 identity에는 Python 3.11.14, torch 2.13.0+cu132, numpy 2.4.4가 기록되어 있다. 현재 DLG Python도 해당 torch 버전이다. 현재 설치 환경은 `environment_snapshot.json`과 `requirements_environment_snapshot.txt`에 별도 보존했다. 현재 torchvision 0.28.0+cu132, torchmetrics 1.9.0, scipy 1.17.1, scikit-learn 1.9.0, matplotlib 3.10.8, pandas 3.0.3, pillow 12.2.0, PyYAML 6.0.3이다. 실행 당시 전체 환경은 미확인이다.

원본 requirements.txt는 버전을 pin하지 않고 torchmetrics 등 일부 import 의존성이 빠져 있다. pyproject.toml도 함께 포함했으며 qacpa 전체에는 matplotlib/pandas를 사용하는 후처리 코드가 있다. 전체 qacpa의 상대 import를 AST로 확인했을 때 누락된 로컬 모듈은 0개다. 실제 전체 모듈 import나 실험 실행 검증은 하지 않았다. requirements snapshot은 이 PC 환경의 기록이며 다른 GPU/OS에 대한 설치 성공 보장은 미확인이다.

모델을 자동 다운로드하지 않고 쓰려면 대상 PC에서 `TORCH_HOME`을 압축 해제한 `torch_home`의 절대경로로 지정하면 기존 torchvision이 `hub/checkpoints/vgg16-397923af.pth`를 찾는다. 코드·설정의 상대 경로 기준은 `quantization_aware_cpa`이고, sibling `codex/docs` 구조를 유지하면 followup protocol 상대 참조가 보존된다. Windows 스타일 source_fp32_config의 역슬래시를 포함한 원본 설정 내용도 그대로 보존했다. 다른 OS에서의 경로 처리 변경은 이번에 하지 않았다.

원본 설정의 data.root, input manifest의 image_path, inputs.pt의 feature_cache_path/victim_cache_path는 이 PC의 절대경로다. 기록의 해시를 보존하기 위해 바꾸지 않았다. 원본 실행 스크립트는 `C:\miniconda\envs\DLG\python.exe`, GPU release/dependency/identity 검사도 포함한다. 따라서 이 전달물의 기존 launcher를 다른 PC에서 그대로 실행하면 모든 경로·등록 해시 검사를 통과한다고 보장할 수 없다. 새 대조군에서는 저장된 inputs.pt와 quantized.pt/observation.pt를 직접 로드하는 별도 진입점이 필요하며, 이는 후속 코드 작업이다.

**누락·미확인·비포함 목록**

요청된 코드, M7/source config, followup protocol, 환경 설정, M7/M6 필수 배치 결과, 초기 VGG16 모델의 누락은 없다. 선택적 rec.pt/source_cosines.csv도 모든 배치에 포함했다. 텐서 구조 확인 실패도 없다. 자세한 구분은 `missing_files.json`을 따른다.

별도 초기 classifier 체크포인트의 사용 여부, 실행 당시 전체 pip/conda 환경, 다른 PC에서의 bitwise 재현성은 미확인이다. 기존 모델 로드 코드와 등록 해시는 확보한 VGG16 초기 모델 사용을 뒷받침한다.

`__pycache__`/`.pyc`와 실행 잠금 파일, 원본 ImageNet JPEG 및 legacy victim/UIA feature cache는 의도적으로 포함하지 않았다. 이 PC에는 해당 80 JPEG가 모두 존재하고, inputs에 참조된 10개 legacy cache의 존재 여부는 audit에 기록되어 있다. 정규화된 실제 입력 80장과 평가용 임베딩은 inputs.pt에 이미 들어 있다. 기존 M6 준비·훈련 경로를 재실행할 때 사용하는 캐시/원본 데이터는 이 대조군 파일 전달 범위와 구분된다.
