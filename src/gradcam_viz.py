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

from .data import DATASETS, IMAGENET_MEAN, IMAGENET_STD, _resolve_data_root
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


def _read_run_num_classes(run_dir: Path, fallback: int) -> int:
    """Best-effort: read num_classes from a run's saved config.json."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return fallback
    try:
        with cfg_path.open() as f:
            return int(json.load(f).get("num_classes", fallback))
    except Exception:
        return fallback


def compute_cam(model, input_tensor, target_class: int | None = None) -> np.ndarray:
    target_layer = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]
    grayscale = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale  # (H, W) in [0, 1]


def _resolve_dataset_root(data_root: Path, experiments_dir: Path) -> Path:
    """If `data_root` already contains val/, use it.
    Otherwise, infer dataset name from any run's config.json and resolve under data_root.
    """
    if (data_root / "val").exists():
        return data_root
    # Try to read dataset name from any run's config.json
    for run in experiments_dir.iterdir() if experiments_dir.exists() else []:
        cfg = run / "config.json"
        if cfg.exists():
            try:
                with cfg.open() as f:
                    ds = json.load(f).get("dataset")
                if ds and ds in DATASETS:
                    return _resolve_data_root(ds, data_root)
            except Exception:
                pass
    # Fallback: just try both
    for ds in DATASETS:
        try:
            return _resolve_data_root(ds, data_root)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not locate a dataset under {data_root}")


def pick_default_image(data_root: Path, experiments_dir: Path | None = None) -> Path:
    """Pick the first val image we can find as a deterministic default."""
    root = _resolve_dataset_root(data_root, experiments_dir or Path("./experiments"))
    val = root / "val"
    for sub in sorted(val.iterdir()):
        if sub.is_dir():
            # Imagewoof uses .JPEG; some datasets use .jpg/.png
            for ext in ("*.JPEG", "*.jpg", "*.jpeg", "*.png"):
                for img in sorted(sub.glob(ext)):
                    return img
    raise FileNotFoundError(f"No val images found under {val}")


def list_runs(experiments_dir: Path) -> list[Path]:
    """Sub-folders of experiments_dir that contain a checkpoints/ folder."""
    return sorted(p for p in experiments_dir.iterdir()
                  if p.is_dir() and (p / "checkpoints").exists())


def _canonical_capture_epochs(run_dir: Path) -> list[int] | None:
    """Look up the canonical capture_epochs for this run.

    Reads from `summary.json` (always populated) first, then `config.json`
    (only populated if resolve_capture_epochs was called). Returns None if
    neither file gives a usable list — callers should fall back to glob.
    """
    for fname in ("summary.json", "config.json"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            with p.open() as f:
                data = json.load(f)
            ce = data.get("capture_epochs")
            if ce:  # non-null, non-empty
                return sorted(int(e) for e in ce)
        except Exception:
            pass
    return None


def list_stage_checkpoints(run_dir: Path) -> list[Path]:
    """Return checkpoint files for THIS RUN's capture_epochs only, sorted by epoch.

    Filters out stale .pth files left by earlier interrupted runs (different
    epoch budgets leave behind different capture sets in the same folder).
    Falls back to globbing all if the metadata is missing.
    """
    ckpt_dir = run_dir / "checkpoints"
    canonical = _canonical_capture_epochs(run_dir)
    if canonical:
        return [ckpt_dir / f"epoch_{e:03d}.pth"
                for e in canonical
                if (ckpt_dir / f"epoch_{e:03d}.pth").exists()]
    return sorted(ckpt_dir.glob("epoch_*.pth"))


def build_grid(
    experiments_dir: Path,
    image_path: Path,
    out_path: Path,
    num_classes: int | None = None,
    image_size: int = 224,
    target_class: int | None = None,
):
    device = get_device()
    input_tensor, rgb = load_image(image_path, image_size=image_size)
    input_tensor = input_tensor.to(device)

    runs = list_runs(experiments_dir)
    if not runs:
        raise RuntimeError(f"No runs under {experiments_dir}")

    # Per-run num_classes from saved config.json (falls back to CLI value).
    # All runs in one experiments_dir are expected to share a num_classes,
    # but we read per-run so this works even with mixed-dataset directories.
    fallback = num_classes if num_classes is not None else 200
    num_classes_per_run: dict[str, int] = {
        r.name: _read_run_num_classes(r, fallback) for r in runs
    }

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
            model = load_checkpoint(ckpt_path, num_classes_per_run[run_dir.name], device)
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
    ap.add_argument("--data-root", default="./data",
                    help="Parent dir containing the dataset folder. "
                         "Used only to pick a default image if --image not given.")
    ap.add_argument("--image", default=None,
                    help="Path to a specific test image. Default: first val image")
    ap.add_argument("--out", default="./experiments/grad_cam_grid.png")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="Fallback num_classes if a run's config.json is missing; "
                         "normally auto-detected per-run")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--target-class", type=int, default=None,
                    help="Class index to compute CAM for; default = model's argmax")
    args = ap.parse_args()

    image_path = (Path(args.image) if args.image
                  else pick_default_image(Path(args.data_root), Path(args.experiments_dir)))
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
