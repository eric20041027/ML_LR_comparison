"""Grad-CAM heatmap grid: rows = schedulers, cols = training stages.

Reads checkpoints saved by `src/train.py` and renders a comparison figure
over a fixed test image (default: first val image, but overridable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .model import build_resnet18, get_target_layer
from .utils import get_device


def load_image(path: str | Path, image_size: int = 224):
    """Return (preproc_tensor[B=1,3,H,W], rgb_float[H,W,3] in [0,1])."""
    img = Image.open(path).convert("RGB")
    rgb = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])(img)
    rgb_float = np.array(rgb).astype(np.float32) / 255.0

    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(rgb).unsqueeze(0)
    return tensor, rgb_float


def load_checkpoint(ckpt_path: Path, num_classes: int, device) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    model = build_resnet18(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def compute_cam(model, input_tensor, target_class: int | None = None) -> np.ndarray:
    target_layer = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]
    grayscale = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale  # (H, W) in [0, 1]


def pick_default_image(data_root: Path) -> Path:
    """Pick the first val image we can find as a deterministic default."""
    val = data_root / "val"
    for sub in sorted(val.iterdir()):
        if sub.is_dir():
            for img in sorted(sub.glob("*.JPEG")):
                return img
    raise FileNotFoundError(f"No val images found under {val}")


def list_runs(experiments_dir: Path) -> list[Path]:
    """Sub-folders of experiments_dir that contain a checkpoints/ folder."""
    return sorted(p for p in experiments_dir.iterdir()
                  if p.is_dir() and (p / "checkpoints").exists())


def list_stage_checkpoints(run_dir: Path) -> list[Path]:
    """Return checkpoint files for the saved early/mid/late stages, sorted by epoch."""
    return sorted((run_dir / "checkpoints").glob("epoch_*.pth"))


def build_grid(
    experiments_dir: Path,
    image_path: Path,
    out_path: Path,
    num_classes: int = 200,
    image_size: int = 224,
    target_class: int | None = None,
):
    device = get_device()
    input_tensor, rgb = load_image(image_path, image_size=image_size)
    input_tensor = input_tensor.to(device)

    runs = list_runs(experiments_dir)
    if not runs:
        raise RuntimeError(f"No runs under {experiments_dir}")

    # Use the first run to determine the set of stage epochs.
    stage_ckpts_per_run = {r.name: list_stage_checkpoints(r) for r in runs}
    n_stages = max(len(v) for v in stage_ckpts_per_run.values())
    n_runs = len(runs)

    # Grid layout: extra leftmost column shows the input image once per row.
    fig, axes = plt.subplots(n_runs, n_stages + 1,
                             figsize=(3 * (n_stages + 1), 3 * n_runs))
    if n_runs == 1:
        axes = np.array([axes])

    for r, run_dir in enumerate(runs):
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(run_dir.name, fontsize=12, rotation=90, labelpad=10)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("input", fontsize=11)

        ckpts = stage_ckpts_per_run[run_dir.name]
        for c in range(n_stages):
            ax = axes[r, c + 1]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(ckpts):
                ax.axis("off")
                continue
            ckpt_path = ckpts[c]
            model = load_checkpoint(ckpt_path, num_classes, device)
            grayscale = compute_cam(model, input_tensor, target_class=target_class)
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
            ax.imshow(overlay)
            if r == 0:
                ax.set_title(ckpt_path.stem.replace("epoch_", "ep "), fontsize=11)
            # Free GPU mem promptly between models.
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    fig.suptitle(f"Grad-CAM evolution — {image_path.name}", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default="./experiments")
    ap.add_argument("--data-root", default="./data/tiny-imagenet-200",
                    help="Used only to pick a default image if --image not given")
    ap.add_argument("--image", default=None,
                    help="Path to a specific test image. Default: first val image")
    ap.add_argument("--out", default="./experiments/grad_cam_grid.png")
    ap.add_argument("--num-classes", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--target-class", type=int, default=None,
                    help="Class index to compute CAM for; default = model's argmax")
    args = ap.parse_args()

    image_path = Path(args.image) if args.image else pick_default_image(Path(args.data_root))
    build_grid(
        experiments_dir=Path(args.experiments_dir),
        image_path=image_path,
        out_path=Path(args.out),
        num_classes=args.num_classes,
        image_size=args.image_size,
        target_class=args.target_class,
    )


if __name__ == "__main__":
    main()
