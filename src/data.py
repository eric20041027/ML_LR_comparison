"""Dataset registry + DataLoader factory.

Supports two datasets via a simple registry:

  - tiny_imagenet : 200 classes, 100k train / 10k val, native 64x64.
                    `val/` ships flat with annotations -> we reorganize into
                    per-class folders so ImageFolder works.
  - imagewoof     : 10 fine-grained dog breeds, ~9.5k train / 3.9k val,
                    native ~320x320 (we use the 320 build from fast-ai).
                    Already ImageFolder-ready, no reorg needed.

All datasets are resized to 224x224 so a shared ResNet-18 head + Grad-CAM
target (last conv -> 7x7) works uniformly.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive_name: str            # filename used when downloading
    extracted_root_name: str     # top-level folder inside the archive
    num_classes: int
    needs_val_reorg: bool        # True for Tiny-ImageNet, False for Imagewoof


DATASETS: dict[str, DatasetSpec] = {
    "tiny_imagenet": DatasetSpec(
        name="tiny_imagenet",
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        archive_name="tiny-imagenet-200.zip",
        extracted_root_name="tiny-imagenet-200",
        num_classes=200,
        needs_val_reorg=True,
    ),
    "imagewoof": DatasetSpec(
        name="imagewoof",
        # fast-ai's 320px build: ~340MB, plenty of margin for 224 training.
        url="https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
        archive_name="imagewoof2-320.tgz",
        extracted_root_name="imagewoof2-320",
        num_classes=10,
        needs_val_reorg=False,
    ),
}


# ---------- Download / extract ----------

def _extract(archive_path: Path, dest: Path) -> None:
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif archive_path.name.endswith((".tgz", ".tar.gz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type: {archive_path}")


def download_and_extract(dataset: str, data_dir: str | os.PathLike) -> Path:
    """Idempotently download+extract `dataset` into `data_dir`. Returns the dataset root."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir / spec.extracted_root_name
    if (root / "train").exists() and (root / "val").exists():
        return root

    archive = data_dir / spec.archive_name
    if not archive.exists():
        print(f"Downloading {spec.url} -> {archive}")
        urlretrieve(spec.url, archive)

    print(f"Extracting {archive}")
    _extract(archive, data_dir)
    return root


# ---------- Tiny-ImageNet val/ reorganization ----------

def _reorganize_tiny_imagenet_val(root: Path) -> None:
    """Flat val/images/*.JPEG -> per-class subfolders, idempotent via a sentinel."""
    val_dir = root / "val"
    sentinel = val_dir / ".reorganized"
    if sentinel.exists():
        return

    ann_file = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"
    if not ann_file.exists() or not images_dir.exists():
        sentinel.touch()
        return

    mapping: dict[str, str] = {}
    with ann_file.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    for wnid in set(mapping.values()):
        (val_dir / wnid).mkdir(exist_ok=True)
    for fname, wnid in mapping.items():
        src = images_dir / fname
        dst = val_dir / wnid / fname
        if src.exists():
            shutil.move(str(src), str(dst))
    try:
        images_dir.rmdir()
    except OSError:
        pass
    sentinel.touch()


# ---------- Transforms / loaders ----------

def build_transforms(image_size: int = 224, train: bool = True, aug: str = "basic"):
    if train:
        # Delegate to augment.py so the from-scratch 'strong' pipeline lives
        # alongside MixUp. Local import avoids a circular import at module load.
        from .augment import build_train_transform
        return build_train_transform(image_size, aug=aug)
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _resolve_data_root(dataset: str, data_root: str | os.PathLike) -> Path:
    """Allow `data_root` to be either the dataset folder directly OR its parent."""
    spec = DATASETS[dataset]
    root = Path(data_root)
    if (root / "train").exists():
        return root
    candidate = root / spec.extracted_root_name
    if (candidate / "train").exists():
        return candidate
    raise FileNotFoundError(
        f"Expected {dataset} at {root} or {candidate}. "
        f"Run: python -m scripts.download_data --dataset {dataset} --data-dir <parent>"
    )


def get_datasets(dataset: str, data_root: str | os.PathLike, image_size: int = 224,
                 aug: str = "basic"):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    root = _resolve_data_root(dataset, data_root)
    if spec.needs_val_reorg:
        _reorganize_tiny_imagenet_val(root)
    train_ds = datasets.ImageFolder(root / "train",
                                    transform=build_transforms(image_size, train=True, aug=aug))
    val_ds = datasets.ImageFolder(root / "val",
                                  transform=build_transforms(image_size, train=False))
    return train_ds, val_ds


def get_loaders(
    dataset: str,
    data_root: str | os.PathLike,
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = 224,
    aug: str = "basic",
):
    train_ds, val_ds = get_datasets(dataset, data_root, image_size=image_size, aug=aug)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_ds.classes
