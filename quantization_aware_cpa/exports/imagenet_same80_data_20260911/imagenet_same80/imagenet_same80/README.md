# 이번 실험과 동일한 ImageNet 80장

이 ZIP은 로컬에 있던 ILSVRC2012 validation 원본 JPEG80장을 변경 없이 복사한 데이터 묶음입니다.
INT4, Top-k50%,30%,20%,10%에 사용한 이미지 집합 및16x5 순서와 같습니다.

- `images/<synset>/<원본파일명>.JPEG`: 원본80장, 재인코딩/리사이즈 없음
- `manifest.csv`: 원래1000-class 라벨, global/batch/source 순서, 파일경로, SHA-256
- `load_same80.py`: 위 라벨과 순서로 읽는 PyTorch Dataset/DataLoader
- `protocol.json`: 전처리, 환경 버전 및 데이터 출처

다른 컴퓨터에 이 ZIP만 옮겨 압축을 풀면 됩니다. 전체 ImageNet은 필요 없습니다.
이미지 입력에 torchvision의 `VGG16_Weights.IMAGENET1K_V1.transforms()`를 적용합니다:
bilinear Resize256, CenterCrop224, RGB float tensor, ImageNet mean/std 정규화.
내보내기 시 모든80장의 전처리 결과를 저장된 실험 입력과 CPU에서 비트 단위로 대조했습니다.

PyTorch, torchvision, Pillow가 설치된 환경에서 다음을 실행하면 다섯 배치를 확인할 수 있습니다.

```bat
cd /d "압축을 푼 경로\imagenet_same80"
python load_same80.py
```

실험 코드에서는 다음처럼 불러옵니다.

```python
from load_same80 import make_loader
for batch_index, (images, labels) in enumerate(make_loader()):
    # images: (16, 3, 224, 224), labels: 원래 ImageNet1000-class 라벨
    pass
```

`ImageFolder`로 이80개 폴더만 읽으면 클래스 번호가0..79로 다시 매겨집니다.
원래 실험을 재현하려면 반드시 manifest.csv의 `label`을 사용하세요.
`dataset_index`는 원래 전체 validation 데이터셋에서의 인덱스이며, 이 부분집합의 인덱스가 아닙니다.
원래16x5 묶음을 유지하려면 shuffle=False, batch_size=16을 사용합니다.

원래 업데이트 수집은 VGG16 IMAGENET1K_V1 round0에서 매 배치 모델을 초기화하고,
Conv를 고정한 채 FC만 SGD lr0.001로10스텝 학습했습니다. Momentum, weight decay,
classifier dropout은 사용하지 않았습니다. 관측 텐서는 (W_initial-W_after10)/lr였습니다.
이 ZIP은 이미지 데이터 묶음입니다. 모델 가중치, 저장된 업데이트, 공격 결과는 포함하지 않습니다.
동일 이미지를 재학습해도 장치/라이브러리 차이에 따른 수치 차이가 생길 수 있습니다.

ImageNet 공식 데이터 안내: https://www.image-net.org/download.php
