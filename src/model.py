"""ResNet-18 wrapped for Tiny-ImageNet (200 classes)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 200, pretrained: bool = False) -> nn.Module:
    """Return a ResNet-18 with the final FC head resized to `num_classes`.

    Pretrained weights speed up convergence but the assignment is about
    comparing schedulers, so default is `pretrained=False` for a fair start.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv block — used as the Grad-CAM target."""
    return model.layer4[-1]
