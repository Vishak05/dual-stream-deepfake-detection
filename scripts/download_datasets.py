"""Run this inside Google Colab (real datasets never touch the local machine —
see README.md 'Colab workflow'). Requires a kaggle.json API token uploaded
to the Colab runtime and `pip install kaggle`."""
import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download FFHQ/CelebA/GAN-fake datasets via the Kaggle API."
    )
    parser.add_argument(
        "--kaggle-dataset",
        action="append",
        required=True,
        help="Kaggle dataset slug, e.g. 'xhlulu/140k-real-and-fake-faces'. Repeatable.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination directory, e.g. /content/data")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle
    except ImportError as exc:
        raise SystemExit("kaggle package not installed. Run: pip install kaggle") from exc

    for slug in args.kaggle_dataset:
        print(f"Downloading {slug} -> {args.out_dir}")
        kaggle.api.dataset_download_files(slug, path=str(args.out_dir), unzip=True)


if __name__ == "__main__":
    main()
