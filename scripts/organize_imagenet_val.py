from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scipy.io import loadmat


def load_label_to_wnid(devkit_root: Path) -> dict[int, str]:
    meta_path = devkit_root / "data" / "meta.mat"
    meta = loadmat(meta_path, squeeze_me=True)["synsets"]
    result: dict[int, str] = {}
    for entry in meta:
        ilsvrc_id = int(entry["ILSVRC2012_ID"])
        wnid = str(entry["WNID"])
        result[ilsvrc_id] = wnid
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.imagenet_root.resolve()
    val_root = root / "val"
    raw_root = root / "val_raw"
    devkit_root = root / "ILSVRC2012_devkit_t12"
    gt_path = devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"

    label_to_wnid = load_label_to_wnid(devkit_root)
    labels = [int(line.strip()) for line in gt_path.read_text().splitlines() if line.strip()]
    images = sorted(raw_root.glob("ILSVRC2012_val_*.JPEG"))
    if len(images) != len(labels):
        raise RuntimeError(f"Expected {len(labels)} val images, found {len(images)} in {raw_root}")

    for image, label in zip(images, labels):
        wnid = label_to_wnid[label]
        class_dir = val_root / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / image.name
        if dest.exists():
            image.unlink()
        else:
            shutil.move(str(image), str(dest))

    remaining = list(raw_root.glob("*.JPEG"))
    if remaining:
        raise RuntimeError(f"{len(remaining)} validation images remain in {raw_root}")
    raw_root.rmdir()


if __name__ == "__main__":
    main()
