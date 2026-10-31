"""Export the exact historical80 original JPEGs and portable labels/order."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/cpa_fia_topk50_same80_B16/image_manifest.csv"
OUTPUT = PROJECT / "exports/imagenet_same80_data_20260911"
PREFIX = "imagenet_same80"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


LOADER = '''"""Load original ImageNet labels and the frozen80-image order from manifest.csv."""
import csv
import hashlib
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VGG16_Weights


class Same80Dataset(Dataset):
    def __init__(self, root=None, verify=True):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parent
        with (self.root / "manifest.csv").open(encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if len(self.rows) != 80 or [int(r["global_index"]) for r in self.rows] != list(range(80)):
            raise ValueError("Expected original80 images in frozen order")
        for i, row in enumerate(self.rows):
            if (int(row["batch_index"]), int(row["source_index"])) != (i // 16, i % 16):
                raise ValueError("Original16x5 batch grouping changed")
            path = self.root / row["relative_path"]
            if verify and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Original JPEG changed: {path}")
        self.transform = VGG16_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row["label"])


def make_loader(root=None):
    return DataLoader(Same80Dataset(root), batch_size=16, shuffle=False, num_workers=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    for batch, (images, labels) in enumerate(make_loader()):
        print(f"batch={batch} shape={tuple(images.shape)} original_imagenet_labels={labels.tolist()}")
'''

README = '''# 이번 실험과 동일한 ImageNet 80장

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
'''


def main() -> None:
    import PIL
    from PIL import Image
    import torch
    import torchvision
    from torchvision.models import VGG16_Weights

    torch.set_num_threads(2)
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    assert len(source_rows) == 80
    assert [int(row["global_index"]) for row in source_rows] == list(range(80))
    assert len({row["image_path"] for row in source_rows}) == 80
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    rows, blobs, checked = [], {}, []
    for batch in range(5):
        cache_path = PROJECT / f"results/m6_uia_fp32_B16_fia/batch{batch:02d}/inputs.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
        for local, row in enumerate(source_rows[batch * 16:(batch + 1) * 16]):
            for key in ("global_index", "batch_index", "source_index", "dataset_index", "label", "image_path"):
                assert str(row[key]) == str(cache["metadata"][local][key]), (batch, local, key)
            path = Path(row["image_path"])
            data = path.read_bytes()
            relative = f"images/{path.parent.name}/{path.name}"
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                x = transform(image.convert("RGB"))
            if not torch.equal(x, cache["x"][local]):
                raise RuntimeError(f"Original JPEG does not reproduce cached input: {path.name}")
            rows.append({key: row[key] for key in ("global_index", "batch_index", "source_index", "dataset_index", "label")}
                        | {"synset": path.parent.name, "filename": path.name, "relative_path": relative,
                           "sha256": sha(data), "bytes": len(data), "width": width, "height": height})
            blobs[relative] = data
        checked.append({"batch_index": batch, "n_images": 16, "preprocessed_inputs_bit_exact": True})
        del cache
        print(f"Verified originals and cached input tensors: batch={batch}", flush=True)
    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(manifest_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    blobs["manifest.csv"] = manifest_buffer.getvalue().encode("utf-8")
    blobs["load_same80.py"] = LOADER.encode("utf-8")
    blobs["README.md"] = README.encode("utf-8")
    protocol = {"n_images": 80, "batch_size": 16, "batch_indices": list(range(5)),
                "dataset": "ILSVRC2012 ImageNet validation; exact historical80",
                "selection": "original global_index0..79, unchanged batches0..4",
                "labels": "original ImageNet1000-class indices; never remap to subset classes",
                "transform": repr(transform), "source_manifest_sha256": sha(SOURCE.read_bytes()),
                "source_manifest": str(SOURCE.relative_to(PROJECT)),
                "preprocessed_cache_verification": checked,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "torchvision": torchvision.__version__, "Pillow": PIL.__version__}}
    blobs["protocol.json"] = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    zip_path = OUTPUT / "imagenet_same80.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in blobs.items():
            archive.writestr(f"{PREFIX}/{name}", data)
    with zipfile.ZipFile(zip_path) as archive:
        assert len(archive.namelist()) == 84
        assert archive.testzip() is None
        for name, data in blobs.items():
            assert sha(archive.read(f"{PREFIX}/{name}")) == sha(data), name
    for name in ("manifest.csv", "README.md", "protocol.json"):
        (OUTPUT / name).write_bytes(blobs[name])
    report = {"zip": zip_path.name, "zip_bytes": zip_path.stat().st_size,
              "zip_sha256": sha(zip_path.read_bytes()), "n_original_jpegs": 80,
              "n_batches": 5, "jpeg_bytes": sum(int(row["bytes"]) for row in rows),
              "all_archive_entries_sha256_verified": True, "all80_preprocessed_inputs_bit_exact": True}
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
