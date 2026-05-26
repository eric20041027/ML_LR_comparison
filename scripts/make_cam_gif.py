"""Compose per-epoch Grad-CAM PNGs into an animated GIF.

Each frame shows one row: [input image | fixed | step | cosine | cosine_restart | onecycle]
at a single epoch. Frames are stacked into a looping GIF.

Pre-requisite: each scheduler's run dir must contain `cam_per_epoch/epoch_NNN.png`
(produced by train.py when `gradcam_per_epoch=true`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Display order. Matches the static grad_cam_grid.png convention.
SCHEDULERS = ["cosine", "cosine_restart", "fixed", "onecycle", "step"]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try common cross-platform fonts; fall back to bitmap default.
    for name in ("arial.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def assemble_frame(
    experiments_dir: Path,
    epoch: int,
    input_image: Image.Image,
    cell_size: int = 256,
    label_height: int = 30,
    footer_height: int = 30,
) -> Image.Image:
    """One frame: input + 5 scheduler CAMs in a row, with column labels and epoch footer."""
    n_cols = len(SCHEDULERS) + 1
    W = cell_size * n_cols
    H = label_height + cell_size + footer_height

    frame = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(frame)
    title_font = _load_font(14)
    footer_font = _load_font(18)

    # Column 0: input
    draw.text((4, 6), "input", fill="black", font=title_font)
    frame.paste(input_image.resize((cell_size, cell_size)), (0, label_height))

    # Columns 1..5: scheduler CAMs
    for col, sched in enumerate(SCHEDULERS, start=1):
        x = col * cell_size
        draw.text((x + 4, 6), sched, fill="black", font=title_font)
        cam_path = experiments_dir / sched / "cam_per_epoch" / f"epoch_{epoch:03d}.png"
        if cam_path.exists():
            cam_img = Image.open(cam_path).convert("RGB").resize((cell_size, cell_size))
            frame.paste(cam_img, (x, label_height))
        else:
            # Missing frame (interrupted run?). Draw a placeholder.
            draw.rectangle(
                [x, label_height, x + cell_size, label_height + cell_size],
                fill="#E2E8F0",
            )
            draw.text((x + 8, label_height + cell_size // 2),
                      f"no ep{epoch}", fill="#64748B", font=title_font)

    # Footer with epoch label
    draw.text((10, label_height + cell_size + 4),
              f"epoch {epoch}", fill="black", font=footer_font)
    return frame


def _resolve_input_image(experiments_dir: Path) -> Path:
    """Find the cached input.png that train.py wrote inside cam_per_epoch/."""
    for sched in SCHEDULERS:
        candidate = experiments_dir / sched / "cam_per_epoch" / "input.png"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No cam_per_epoch/input.png under {experiments_dir}. "
        "Re-train with gradcam_per_epoch=true."
    )


def _collect_epochs(experiments_dir: Path) -> list[int]:
    """Union of epoch indices that exist in any scheduler's cam_per_epoch/."""
    epochs: set[int] = set()
    for sched in SCHEDULERS:
        d = experiments_dir / sched / "cam_per_epoch"
        if not d.exists():
            continue
        for f in d.glob("epoch_*.png"):
            try:
                epochs.add(int(f.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return sorted(epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", required=True,
                    help="Path to a single experiment group's output dir "
                         "(contains fixed/, step/, cosine/, ... subdirs)")
    ap.add_argument("--out", required=True, help="output GIF path")
    ap.add_argument("--input-image", default=None,
                    help="Override the input image (default: auto-resolve from cam_per_epoch/input.png)")
    ap.add_argument("--cell-size", type=int, default=256)
    ap.add_argument("--duration", type=int, default=200,
                    help="ms per frame (default 200 = 5 fps)")
    ap.add_argument("--target-total-ms", type=int, default=None,
                    help="If set, overrides --duration so total GIF length matches this")
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir)
    epochs = _collect_epochs(exp_dir)
    if not epochs:
        raise RuntimeError(f"No epoch_*.png found under {exp_dir}. "
                           "Re-train with gradcam_per_epoch=true.")
    print(f"Found {len(epochs)} epochs: {epochs[0]}..{epochs[-1]}")

    input_path = Path(args.input_image) if args.input_image else _resolve_input_image(exp_dir)
    input_image = Image.open(input_path).convert("RGB")
    print(f"Using input image: {input_path}")

    duration = args.duration
    if args.target_total_ms:
        duration = max(40, args.target_total_ms // len(epochs))
        print(f"target_total_ms={args.target_total_ms} -> duration={duration}ms/frame")

    print("Composing frames...")
    frames = [assemble_frame(exp_dir, ep, input_image, args.cell_size) for ep in epochs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True, append_images=frames[1:],
        duration=duration, loop=0, optimize=True,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path}  ({len(frames)} frames, {duration}ms each, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
