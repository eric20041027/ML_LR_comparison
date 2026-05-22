"""Factory for the 5 learning-rate schedulers compared in this study.

All factories return an object with a `.step()` method compatible with the
PyTorch scheduler protocol. `step_granularity` tells the trainer whether to
call `.step()` per epoch (`"epoch"`) or per batch (`"batch"`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
    StepLR,
)

SCHEDULER_NAMES = ("fixed", "step", "cosine", "cosine_restart", "onecycle")


@dataclass
class SchedulerBundle:
    scheduler: Any
    step_granularity: str  # "epoch" or "batch"


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    *,
    epochs: int,
    steps_per_epoch: int,
    base_lr: float,
    max_lr: float | None = None,
) -> SchedulerBundle:
    name = name.lower()
    if name == "fixed":
        # No-op: lambda that returns 1.0 forever -> LR stays at base_lr.
        sched = LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        return SchedulerBundle(sched, step_granularity="epoch")

    if name == "step":
        step_size = max(1, epochs // 3)  # 3 drops over training
        sched = StepLR(optimizer, step_size=step_size, gamma=0.1)
        return SchedulerBundle(sched, step_granularity="epoch")

    if name == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
        return SchedulerBundle(sched, step_granularity="epoch")

    if name == "cosine_restart":
        # First cycle = ~1/4 of training, then doubling. Gives ~2-3 restarts.
        T_0 = max(1, epochs // 4)
        sched = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2,
                                            eta_min=base_lr * 0.01)
        return SchedulerBundle(sched, step_granularity="epoch")

    if name == "onecycle":
        peak = max_lr if max_lr is not None else base_lr * 10.0
        sched = OneCycleLR(
            optimizer,
            max_lr=peak,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return SchedulerBundle(sched, step_granularity="batch")

    raise ValueError(f"Unknown scheduler: {name!r}. Choose from {SCHEDULER_NAMES}.")


def current_lr(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]
