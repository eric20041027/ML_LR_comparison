"""MixUp + train transform variants ('basic' vs 'strong' = + RandAugment).

Used by the from-scratch Imagewoof recipe (configs/imagewoof_scratch_*.yaml)
to compensate for the loss of ImageNet pretraining.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(image_size: int = 224, aug: str = "basic"):
    """Train-side augmentation pipeline.

    'basic' is the same flow used for the pretrained runs (kept for backward
    comparability). 'strong' adds torchvision's RandAugment, which is the
    de-facto recipe for from-scratch ResNet-style training on small datasets.
    """
    if aug not in {"basic", "strong"}:
        raise ValueError(f"Unknown aug {aug!r}; choose 'basic' or 'strong'")
    ops = [
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=8),
        transforms.RandomHorizontalFlip(),
    ]
    if aug == "strong":
        # num_ops=2, magnitude=9 follows the timm / RandAugment paper defaults
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops.extend([
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


class MixUp:
    """Per-batch MixUp. `alpha=0` disables; typical values 0.1 - 0.4.

    Usage in train loop:
        x, y_a, y_b, lam = mixup(x, y)
        logits = model(x)
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.alpha <= 0:
            return x, y, y, 1.0
        lam = float(np.random.beta(self.alpha, self.alpha))
        # Symmetric trick: ensure lam >= 0.5 so y_a is always the "dominant"
        # label. Stabilizes loss interpretation and matches timm's convention.
        lam = max(lam, 1.0 - lam)
        idx = torch.randperm(x.size(0), device=x.device)
        x_mixed = lam * x + (1.0 - lam) * x[idx]
        return x_mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
