"""Plot LR-per-step curves for all runs under experiments/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default="./experiments")
    ap.add_argument("--out", default="./experiments/lr_curves.png")
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir)
    runs = sorted(p for p in exp_dir.iterdir()
                  if p.is_dir() and (p / "history.json").exists())
    if not runs:
        raise RuntimeError(f"No runs with history.json under {exp_dir}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        with (run / "history.json").open() as f:
            hist = json.load(f)
        ax.plot(hist["lr_per_step"], label=run.name, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("Learning-rate schedules")
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
