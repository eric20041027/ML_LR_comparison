"""Download + extract a dataset into the given directory."""
import argparse

from src.data import DATASETS, download_and_extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiny_imagenet",
                    choices=sorted(DATASETS),
                    help="Which dataset to download")
    ap.add_argument("--data-dir", default="./data",
                    help="Parent directory; the dataset folder is extracted under it")
    args = ap.parse_args()

    root = download_and_extract(args.dataset, args.data_dir)
    print(f"{args.dataset} ready at: {root}")


if __name__ == "__main__":
    main()
