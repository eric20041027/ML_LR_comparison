"""Convergence-speed analysis using only existing summary.json files.

Computes, for each (experiment, scheduler):
  - epoch to reach absolute thresholds (e.g. 60% / 80%)
  - epoch to reach a fraction of the scheduler's own best (e.g. 90% of best)

Outputs a markdown table on stdout and a comparison chart PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS

EXPERIMENTS = [
    ("tiny_imagenet", "Tiny-ImageNet (pre, 20 ep)", [40, 50, 60, 65]),
    ("imagewoof", "Imagewoof (pre, 20 ep)", [70, 80, 85, 88, 90]),
    ("imagewoof_scratch", "Imagewoof (scratch, 80 ep)", [40, 60, 70, 78, 82]),
]

SCHEDULERS = ["fixed", "step", "cosine", "cosine_restart", "onecycle"]


def epochs_to_threshold(epochs: list[dict], thr: float) -> int | None:
    for e in epochs:
        if e["val_acc"] >= thr:
            return e["epoch"]
    return None


def epochs_to_fraction_of_best(epochs: list[dict], frac: float) -> int | None:
    best = max(e["val_acc"] for e in epochs)
    target = best * frac
    return epochs_to_threshold(epochs, target)


def load_runs(exp_dir: str) -> dict[str, list[dict]]:
    path = RESULTS / exp_dir / "summary.json"
    with path.open() as f:
        data = json.load(f)
    runs = {r["scheduler"]: r["epochs"] for r in data["runs"]}
    return runs


def print_header(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def main():
    # === Per-experiment absolute-threshold table ===
    for exp_key, exp_label, thresholds in EXPERIMENTS:
        runs = load_runs(exp_key)
        print_header(exp_label)
        header = "Scheduler".ljust(18) + "".join(f">={t}%".rjust(10) for t in thresholds) + "  best".rjust(10)
        print(header)
        print("-" * len(header))
        for sched in SCHEDULERS:
            epochs = runs[sched]
            cells = []
            for t in thresholds:
                ep = epochs_to_threshold(epochs, t)
                cells.append(f"{ep}" if ep else "—")
            best = max(e["val_acc"] for e in epochs)
            print(sched.ljust(18) + "".join(c.rjust(10) for c in cells) + f"{best:.2f}".rjust(10))

    # === Cross-experiment: epoch to 90% of own best ===
    print_header("epochs to reach 90% of own peak val_acc")
    head = "Scheduler".ljust(18) + "".join(name.rjust(28) for _, name, _ in EXPERIMENTS)
    print(head)
    print("-" * len(head))
    for sched in SCHEDULERS:
        cells = []
        for exp_key, _, _ in EXPERIMENTS:
            runs = load_runs(exp_key)
            ep = epochs_to_fraction_of_best(runs[sched], 0.9)
            cells.append(f"{ep}/{len(runs[sched])}" if ep else "—")
        print(sched.ljust(18) + "".join(c.rjust(28) for c in cells))

    # === Plot: val_acc trajectory for each scheduler in each experiment ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    color_map = {
        "fixed": "#94A3B8",          # gray
        "step":  "#A855F7",          # purple
        "cosine": "#3B82F6",         # blue
        "cosine_restart": "#F59E0B", # orange
        "onecycle": "#EF4444",       # red
    }
    for ax, (exp_key, exp_label, _) in zip(axes, EXPERIMENTS):
        runs = load_runs(exp_key)
        for sched in SCHEDULERS:
            epochs = runs[sched]
            xs = [e["epoch"] for e in epochs]
            ys = [e["val_acc"] for e in epochs]
            ax.plot(xs, ys, color=color_map[sched], linewidth=2, label=sched)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_acc (%)")
        ax.set_title(exp_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Convergence trajectories (val_acc vs epoch)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "convergence_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
