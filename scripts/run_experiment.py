"""Run one experiment from a YAML config."""
import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from src.profiles import PROFILES, apply_profile
from src.train import TrainConfig, train


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    allowed = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {unknown}")
    return TrainConfig(**data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    ap.add_argument("--data-root", default=None, help="Override data_root")
    ap.add_argument("--output-dir", default=None, help="Override output_dir")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--profile", default="none",
                    choices=["none", *sorted(PROFILES)],
                    help="GPU profile: applies batch/workers/amp/tf32 + LR scaling")
    ap.add_argument("--gradcam-per-epoch", action="store_true",
                    help="Save a Grad-CAM PNG every epoch on a fixed val image "
                         "(for GIF assembly via scripts.make_cam_gif)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.gradcam_per_epoch:
        cfg.gradcam_per_epoch = True
    apply_profile(cfg, args.profile)

    summary = train(cfg)
    print("===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
