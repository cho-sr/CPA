**M7 업데이트 매칭 대조군용 경량 전송본**

PC에서 업로드할 파일은 `01_minimal_m7.zip` 하나다. ZIP 내부에 코드·설정·관련 프로토콜, M7 batch00~04의 quantized.pt/inputs.pt/관련 메타데이터/summary/지표/rec.pt, 원본 학습 조건 보고서, 환경 명세와 manifest가 있다. 원본 파일의 내용은 변경하지 않았다. 새 실험 코드 작성이나 실험 재실행은 하지 않았다.

초기 모델 가중치는 이 ZIP에 없다. 서버에 아래 SHA-256의 동일 파일이 있으면 재사용한다. 없으면 서버에서 공식 가중치를 직접 내려받거나, 전체 보관본의 `07_model_weights.zip`(515,045,526 bytes)을 추가 전달한다. 서버 직접 다운로드는 이번 PC 작업에서 실행하지 않았다.

- 공식 URL: https://download.pytorch.org/models/vgg16-397923af.pth
- 파일 크기: 553,433,881 bytes
- SHA-256: `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`
- URL 근거: 이 PC에 설치된 torchvision/models/vgg.py의 `VGG16_Weights.IMAGENET1K_V1.url`. 실제 초기 가중치 파일의 해시가 M7/M6 실행 identity와 일치함을 전체 보관 작업에서 확인함.

Linux 서버에서 기본 torchvision cache를 쓴다면 아래는 가중치 다운로드와 해시 검증만 수행하는 예시다. TORCH_HOME을 따로 쓰는 경우 그 디렉터리의 hub/checkpoints 경로에 저장한다.

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl --fail --location --continue-at - \
  'https://download.pytorch.org/models/vgg16-397923af.pth' \
  --output ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
printf '%s  %s\n' \
  '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0' \
  "$HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth" | sha256sum --check
```

기존 M7 관측을 그대로 쓰는 업데이트 매칭에는 M7 `quantized.pt`의 `Y_hat`, `delta`와 초기 모델, 학습 조건이 필요하다. 5배치 모두 동일한 관측 파일을 보존했다. `inputs.pt`에는 정규화된 원본 이미지 16장, label/순서 metadata, 평가용 참 임베딩이 있다. 참 임베딩은 평가용이며 추정 입력으로 사용하도록 실험을 변경한 것이 아니다.

M6의 큰 `observation.pt` 5개는 경량본에서 제외했다. 이것은 이미 저장된 M7 양자화 관측에 대한 매칭/FIA 복원에는 필요하지 않지만, 원래 FP32 업데이트 대조·진단·양자화 입력 재검증까지 하려면 전체 보관본에서 추가 전달해야 한다. M6 최상위 summary/input_manifest/지표는 근거 기록으로 포함했다.

기존 CPA stage1.pt와 중복 fia.pt, MP 진단 tensor, 전체 결과 그림·실행 로그도 제외했다. 기존 비교용 복원 rec.pt, source_cosines.csv, per_image_metrics.csv는 포함했다. 원본 JPEG와 legacy feature/victim cache는 전체 보관본과 마찬가지로 포함하지 않았다.

`minimal_handoff_docs/FULL_ARCHIVE_REPORT_KO.md`는 이전 전체 보관본의 상세 실험 조건 보고서다. 그 문서의 7개 ZIP/283개 원본/56개 tensor/전체 포함 목록은 전체 보관본에 대한 설명이다. **이번 경량 ZIP의 실제 포함 목록은 이 README와 minimal_handoff_docs/manifest.json을 기준으로 한다.**

업데이트 매칭 대조군 구현은 아직 작성되지 않았다. 전달 후 별도 구현과 Python 의존성 설치, 다른 PC에 맞는 경로 처리가 필요하다. 기존 M7 launcher를 실행하면 CPA를 수행하며, 누락된 M6 원본 및 원본 절대경로/identity 검사 때문에 경량본만으로 기존 launcher를 그대로 재실행할 수 있다고 가정하면 안 된다.

이번 전송본도 quantized.pt의 관측을 다시 생성하거나 낮은 dtype으로 바꾸지 않았다. 각 파일 SHA-256을 ZIP에서 다시 읽어 검증하고 결과를 바깥 `archives.json`에 남긴다. 전체 보관본 `handoff_m7_20260909`는 그대로 보존한다.
