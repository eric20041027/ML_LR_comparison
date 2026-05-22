"""Run all 5 scheduler experiments sequentially."""
import argparse
import json
from pathlib import Path

from scripts.run_experiment import load_config
from src.train import train

CONFIGS = [
    "configs/fixed.yaml",
    "configs/step.yaml",
    "configs/cosine.yaml",
    "configs/cosine_restart.yaml",
    "configs/onecycle.yaml",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    results = []
    for cfg_path in CONFIGS:
        if not Path(cfg_path).exists():
            print(f"!! skipping (not found): {cfg_path}")
            continue
        cfg = load_config(cfg_path)
        if args.data_root:
            cfg.data_root = args.data_root
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.epochs is not None:
            cfg.epochs = args.epochs
        print(f"\n========== running: {cfg.run_name} ({cfg.scheduler}) ==========")
        summary = train(cfg)
        results.append(summary)

    out = Path(args.output_dir or "./experiments") / "all_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
