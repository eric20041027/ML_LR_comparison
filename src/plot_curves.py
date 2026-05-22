"""Plot train/val loss and accuracy curves across all runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default="./experiments")
    ap.add_argument("--out", default="./experiments/curves.png")
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir)
    runs = sorted(p for p in exp_dir.iterdir()
                  if p.is_dir() and (p / "history.json").exists())
    if not runs:
        raise RuntimeError(f"No runs with history.json under {exp_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("train_loss", axes[0, 0], "train loss"),
        ("val_loss", axes[0, 1], "val loss"),
        ("train_acc", axes[1, 0], "train acc (%)"),
        ("val_acc", axes[1, 1], "val acc (%)"),
    ]
    for run in runs:
        with (run / "history.json").open() as f:
            hist = json.load(f)
        for key, ax, _ in metrics:
            ax.plot(range(1, len(hist[key]) + 1), hist[key], label=run.name, linewidth=1.5)

    for key, ax, title in metrics:
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
