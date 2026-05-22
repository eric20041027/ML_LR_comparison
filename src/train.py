"""Training loop with TensorBoard logging + early/mid/late checkpoint capture."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import schedulers as sched_mod
from .data import get_loaders
from .model import build_resnet18
from .utils import accuracy, get_device, get_logger, set_seed


@dataclass
class TrainConfig:
    scheduler: str = "cosine"
    data_root: str = "./data/tiny-imagenet-200"
    output_dir: str = "./experiments"
    epochs: int = 20
    batch_size: int = 128
    num_workers: int = 2
    image_size: int = 224
    base_lr: float = 1e-3
    max_lr: float | None = None  # only for onecycle; defaults to base_lr * 10
    weight_decay: float = 5e-4
    num_classes: int = 200
    pretrained: bool = False
    seed: int = 42
    # Capture checkpoints at these epoch indices (1-based). If None, auto-pick
    # early / mid / late based on `epochs`.
    capture_epochs: list[int] | None = None
    run_name: str | None = None
    # GPU acceleration knobs (set by profile, not by user YAML)
    use_amp: bool = False
    tf32: bool = False
    profile: str | None = None

    def resolve_capture_epochs(self) -> list[int]:
        if self.capture_epochs:
            return sorted(set(self.capture_epochs))
        return sorted({1, max(1, self.epochs // 2), self.epochs})


def _eval(model: nn.Module, loader, device, criterion, use_amp: bool = False) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, 100.0 * total_correct / total


def train(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()

    if cfg.tf32 and device.type == "cuda":
        # Ampere+ Tensor Cores: ~2-3x matmul/conv speedup with no code changes
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    run_name = cfg.run_name or cfg.scheduler
    out_dir = Path(cfg.output_dir) / run_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"train.{run_name}", log_file=str(out_dir / "train.log"))

    # Persist config for downstream tools (Grad-CAM) to read back.
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_loader, val_loader, classes = get_loaders(
        cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
    )
    steps_per_epoch = len(train_loader)
    logger.info(f"steps_per_epoch={steps_per_epoch}, classes={len(classes)}, device={device}")

    model = build_resnet18(num_classes=cfg.num_classes, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    bundle = sched_mod.build_scheduler(
        cfg.scheduler, optimizer,
        epochs=cfg.epochs, steps_per_epoch=steps_per_epoch,
        base_lr=cfg.base_lr, max_lr=cfg.max_lr,
    )
    scheduler = bundle.scheduler
    step_granularity = bundle.step_granularity

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    capture_epochs = set(cfg.resolve_capture_epochs())
    logger.info(f"capture_epochs={sorted(capture_epochs)}")

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr_per_step": [],
    }
    global_step = 0
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"[{run_name}] epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step_granularity == "batch":
                scheduler.step()

            lr = sched_mod.current_lr(optimizer)
            writer.add_scalar("lr/step", lr, global_step)
            history["lr_per_step"].append(lr)

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += x.size(0)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")

        if step_granularity == "epoch":
            scheduler.step()

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        val_loss, val_acc = _eval(model, val_loader, device, criterion, use_amp=amp_enabled)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        logger.info(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} lr={sched_mod.current_lr(optimizer):.2e}"
        )

        if epoch in capture_epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "scheduler": cfg.scheduler,
            }, ckpt_path)
            logger.info(f"saved checkpoint: {ckpt_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "scheduler": cfg.scheduler,
            }, ckpt_dir / "best.pth")

    elapsed = time.time() - t_start
    summary = {
        "run_name": run_name,
        "scheduler": cfg.scheduler,
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "elapsed_sec": elapsed,
        "capture_epochs": sorted(capture_epochs),
    }
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    writer.close()
    logger.info(f"done in {elapsed:.1f}s. best_val_acc={best_val_acc:.2f}")
    return summary
