"""GPU-specific defaults applied on top of a YAML config.

Keeps the 5-scheduler comparison fair *within* a profile (all runs in one
session see the same batch / workers / precision settings), while letting
A100 sessions use bigger batches + mixed precision without editing configs.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "t4": {
        "batch_size": 128,
        "num_workers": 2,
        # T4 (Turing, sm_75) has FP16 Tensor Cores -> AMP gives ~1.5x speedup.
        # TF32 is Ampere-only so stays disabled.
        "use_amp": True,
        "tf32": False,
        "lr_scale": 1.0,
    },
    "a100": {
        # ~3x batch -> linear LR scaling (lr_scale = 3.0)
        "batch_size": 384,
        "num_workers": 8,
        "use_amp": True,
        "tf32": True,
        "lr_scale": 3.0,
    },
}


def apply_profile(cfg, name: str | None):
    """Mutate `cfg` (a TrainConfig) in place with profile defaults.

    `name=None` or `"none"` is a no-op so YAML values win unchanged.
    """
    if not name or name.lower() == "none":
        return cfg
    name = name.lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}. Choose: {sorted(PROFILES)} or 'none'.")

    overrides = dict(PROFILES[name])
    lr_scale = overrides.pop("lr_scale", 1.0)

    cfg_field_names = {f.name for f in fields(cfg)}
    for k, v in overrides.items():
        if k in cfg_field_names:
            setattr(cfg, k, v)

    cfg.base_lr = cfg.base_lr * lr_scale
    if getattr(cfg, "max_lr", None) is not None:
        cfg.max_lr = cfg.max_lr * lr_scale
    cfg.profile = name
    return cfg
