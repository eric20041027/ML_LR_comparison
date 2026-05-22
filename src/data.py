"""Tiny-ImageNet download + ImageFolder-compatible DataLoader.

The official zip ships `val/` as a flat folder + annotations file; this module
reorganizes it into per-class subfolders so `torchvision.datasets.ImageFolder`
works for both `train/` and `val/`.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def download_and_extract(data_dir: str | os.PathLike) -> Path:
    """Download tiny-imagenet-200.zip into `data_dir` and extract. Idempotent."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir / "tiny-imagenet-200"
    if (root / "train").exists() and (root / "val").exists():
        return root

    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Downloading {TINY_IMAGENET_URL} -> {zip_path}")
        urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return root


def reorganize_val(root: str | os.PathLike) -> None:
    """Convert flat val/images/ into per-class folders using val_annotations.txt.

    Idempotent: a sentinel file `.reorganized` is written when done.
    """
    root = Path(root)
    val_dir = root / "val"
    sentinel = val_dir / ".reorganized"
    if sentinel.exists():
        return

    ann_file = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"
    if not ann_file.exists() or not images_dir.exists():
        # Already reorganized in some other way; just mark done.
        sentinel.touch()
        return

    mapping: dict[str, str] = {}
    with ann_file.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]  # filename -> wnid

    for wnid in set(mapping.values()):
        (val_dir / wnid).mkdir(exist_ok=True)

    for fname, wnid in mapping.items():
        src = images_dir / fname
        dst = val_dir / wnid / fname
        if src.exists():
            shutil.move(str(src), str(dst))

    # Clean up
    try:
        images_dir.rmdir()
    except OSError:
        pass
    sentinel.touch()


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_datasets(data_root: str | os.PathLike, image_size: int = 224):
    root = Path(data_root)
    if not (root / "train").exists():
        raise FileNotFoundError(
            f"Expected Tiny-ImageNet at {root}. "
            "Run scripts/download_data.py first."
        )
    reorganize_val(root)
    train_ds = datasets.ImageFolder(root / "train", transform=build_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(root / "val", transform=build_transforms(image_size, train=False))
    return train_ds, val_ds


def get_loaders(
    data_root: str | os.PathLike,
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = 224,
):
    train_ds, val_ds = get_datasets(data_root, image_size=image_size)
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
