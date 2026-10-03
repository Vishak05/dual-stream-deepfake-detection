"""Builds original/augmented/test manifest CSVs from the 140k real-and-fake
faces layout (<root>/{train,test}/{real,fake}/*.jpg) plus a folder of
StyleGAN3-generated images. Labels come from the folder names, not from the
Kaggle CSVs (which use 1 = real, the opposite of dsfnet's 1 = fake)."""
import argparse
import random
from pathlib import Path

from dsfnet.data.manifest import (
    build_original_manifest,
    build_stylegan3_augmented_manifest,
    scan_source_folder,
    write_manifest_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build original/augmented/test manifest CSVs.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Folder containing train/ and test/, each with real/ and fake/.")
    parser.add_argument("--stylegan3-dir", type=Path, required=True, help="StyleGAN3 images added to the augmented training manifest.")
    parser.add_argument("--stylegan3-test-dir", type=Path, default=None, help="Optional held-out StyleGAN3 images added to the test manifest.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--paper-degraded-dir", type=Path, default=None, help="Degraded StyleGAN3 copies; also writes paper_augmented_manifest.csv = original + the first --paper-clean-count StyleGAN3 images + these copies (the base paper's Fig. 3 composition).")
    parser.add_argument("--paper-clean-count", type=int, default=4000)
    parser.add_argument("--max-train-per-class", type=int, default=None, help="Randomly subsample each training class to at most N images.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _subsample(entries: list, n: int | None, rng: random.Random) -> list:
    if n is None or len(entries) <= n:
        return entries
    return sorted(rng.sample(entries, n), key=lambda e: e.path)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_root = args.dataset_root / "train"
    original = build_original_manifest(
        real_sources={"FFHQ": train_root / "real"},
        fake_sources={"StyleGAN": train_root / "fake"},
    )
    real = _subsample([e for e in original if e.label == 0], args.max_train_per_class, rng)
    fake = _subsample([e for e in original if e.label == 1], args.max_train_per_class, rng)
    original = real + fake
    augmented = build_stylegan3_augmented_manifest(original, args.stylegan3_dir)

    test_root = args.dataset_root / "test"
    test = scan_source_folder(test_root / "real", label=0, source="FFHQ") + scan_source_folder(test_root / "fake", label=1, source="StyleGAN")
    if args.stylegan3_test_dir is not None:
        test += scan_source_folder(args.stylegan3_test_dir, label=1, source="StyleGAN3")

    manifests = {"original_manifest.csv": original, "augmented_manifest.csv": augmented, "test_manifest.csv": test}
    if args.paper_degraded_dir is not None:
        clean = scan_source_folder(args.stylegan3_dir, label=1, source="StyleGAN3")[: args.paper_clean_count]
        degraded = scan_source_folder(args.paper_degraded_dir, label=1, source="StyleGAN3")
        manifests["paper_augmented_manifest.csv"] = original + clean + degraded
    for name, entries in manifests.items():
        write_manifest_csv(entries, args.out_dir / name)
        n_fake = sum(e.label for e in entries)
        print(f"{name}: {len(entries)} images ({len(entries) - n_fake} real, {n_fake} fake)")


if __name__ == "__main__":
    main()
