"""Run all 5 scheduler experiments for a given dataset, sequentially."""
import argparse
import json
from pathlib import Path

from scripts.run_experiment import load_config
from src.profiles import PROFILES, apply_profile
from src.train import train

# Map config-group name -> ordered list of configs. A group is not necessarily
# a literal dataset (imagewoof_scratch points to imagewoof data but trains
# from random init with stronger aug).
CONFIGS = {
    "tiny_imagenet": [
        "configs/fixed.yaml",
        "configs/step.yaml",
        "configs/cosine.yaml",
        "configs/cosine_restart.yaml",
        "configs/onecycle.yaml",
    ],
    "imagewoof": [
        "configs/imagewoof_fixed.yaml",
        "configs/imagewoof_step.yaml",
        "configs/imagewoof_cosine.yaml",
        "configs/imagewoof_cosine_restart.yaml",
        "configs/imagewoof_onecycle.yaml",
    ],
    "imagewoof_scratch": [
        "configs/imagewoof_scratch_fixed.yaml",
        "configs/imagewoof_scratch_step.yaml",
        "configs/imagewoof_scratch_cosine.yaml",
        "configs/imagewoof_scratch_cosine_restart.yaml",
        "configs/imagewoof_scratch_onecycle.yaml",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiny_imagenet",
                    choices=sorted(CONFIGS),
                    help="Config-group name (matches keys in CONFIGS). May or may not "
                         "be a literal dataset, e.g. imagewoof_scratch reuses imagewoof "
                         "data but trains from scratch with stronger aug.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--profile", default="none",
                    choices=["none", *sorted(PROFILES)],
                    help="GPU profile: applies batch/workers/amp/tf32 + LR scaling")
    ap.add_argument("--gradcam-per-epoch", action="store_true",
                    help="Save a Grad-CAM PNG every epoch on a fixed val image "
                         "(for GIF assembly via scripts.make_cam_gif)")
    args = ap.parse_args()

    configs = CONFIGS[args.dataset]
    results = []
    for cfg_path in configs:
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
        if args.gradcam_per_epoch:
            cfg.gradcam_per_epoch = True
        apply_profile(cfg, args.profile)
        print(f"\n========== running: {cfg.run_name} ({cfg.scheduler}) "
              f"[dataset={cfg.dataset}, profile={args.profile}, bs={cfg.batch_size}, "
              f"amp={cfg.use_amp}, tf32={cfg.tf32}] ==========")
        summary = train(cfg)
        results.append(summary)

    out = Path(args.output_dir or "./experiments") / "all_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
