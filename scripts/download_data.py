"""Download + extract Tiny-ImageNet into the given directory."""
import argparse
from pathlib import Path

from src.data import download_and_extract, reorganize_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data",
                    help="Parent directory; will contain tiny-imagenet-200/")
    args = ap.parse_args()

    root = download_and_extract(args.data_dir)
    reorganize_val(root)
    print(f"Tiny-ImageNet ready at: {root}")


if __name__ == "__main__":
    main()
