# 완료된 M7 4bit FedAvg CPA의 동일 80장

ImageNet-1K val, seed=0, 배치 16장씩 5개. 원래 배치/이미지 순서와
ImageNet의 0~999 라벨을 manifest.json에 저장했습니다.

## 다른 서버에서 사용

ZIP을 풀고 이 폴더에서 PyTorch가 설치된 Python으로 실행합니다.

```bash
python load_data.py
```

자신의 실험 코드에서:

```python
from load_data import load_batch, verify

verify()
for batch_index in range(5):
    x, y = load_batch(batch_index, device="cuda")
    # x: [16, 3, 224, 224], y: [16], 원래 ImageNet 라벨
    # 이 x, y를 실험의 해당 클라이언트 입력으로 사용
```

x는 실험 당시의 전처리/정규화가 완료된 입력 텐서입니다. 추가 resize,
crop, ToTensor, Normalize를 적용하지 마세요. 이 텐서를 그대로 쓰면
서버의 JPEG 디코더나 torchvision 전처리 버전에 따른 차이도 피할 수 있습니다.

원본 JPEG 80장도 images/에 있습니다. 원본을 직접 읽을 경우 manifest.json의
순서와 label을 사용하세요. 이 부분집합에 ImageFolder를 적용하면 라벨이
0~79로 다시 매겨지므로 기존 ImageNet 라벨과 달라집니다.
원래 전처리는 VGG16_Weights.IMAGENET1K_V1.transforms()입니다.

inputs.pt는 기존 결과의 파일을 바이트 단위로 복사했습니다. 내부의 이전
Windows 경로는 출처 기록이며 로더는 그 경로를 사용하지 않습니다.
Z_evaluation_only는 평가용 실제 특징이며 공격 입력으로 사용하지 마세요.

source_config.json은 원래 설정의 참고용 사본으로, 다른 서버에서 바로 실행할
수 있는 설정은 아닙니다. 이 패키지는 동일 데이터용이며 모델 가중치나
클라이언트 업데이트, 4bit 관측값은 포함하지 않습니다. 기존 공격의 관측값까지
같이 사용하려면 원래 batch00~04/quantized.pt의 Y_hat도 별도로 옮겨야 합니다.

checksums.json은 파일 무결성 확인용 SHA256입니다.
PyTorch 없이 파일만 검증하려면 python load_data.py --verify-only를 실행하세요.
