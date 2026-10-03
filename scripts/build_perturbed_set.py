"""Writes a degraded copy of every image in a manifest, plus a new manifest.

--mode test applies the base paper's test-set protocol (Sec. III-A3, one random
operation per image) and also writes ops.csv (path -> operation) for a
per-operation breakdown. --mode train-degrade applies the Sec. III-B training
augmentations, used to build the paper's 1,000 augmented StyleGAN3 copies.
Each image's randomness is seeded by (--seed, its index), so reruns match."""
import argparse
import csv
import random
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from dsfnet.data.manifest import ManifestEntry, read_manifest_csv, write_manifest_csv
from dsfnet.data.perturb import base_paper_perturbation, training_degradation

SAVE_QUALITY = 95   # matches the 140k dataset's JPEG quality


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a perturbed copy of a manifest's images.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder for the perturbed JPEGs.")
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["test", "train-degrade"], default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def _process(job: tuple) -> tuple[ManifestEntry, str]:
    index, entry, out_dir, mode, seed = job
    rng = random.Random(f"{seed}-{index}")
    image = Image.open(entry.path).convert("RGB")
    if mode == "test":
        image, op, quality = base_paper_perturbation(image, rng)
    else:
        image, op, quality = training_degradation(image, rng), "train-degrade", None
    out_path = Path(out_dir) / f"{index:06d}_{entry.source}_{Path(entry.path).stem}.jpg"
    image.save(out_path, quality=quality or SAVE_QUALITY)
    return ManifestEntry(path=str(out_path), label=entry.label, source=entry.source), op


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = read_manifest_csv(args.manifest)
    jobs = [(i, e, str(args.out_dir), args.mode, args.seed) for i, e in enumerate(entries)]

    if args.workers <= 1:
        # No fork: forking after PyTorch has started its thread pools can deadlock.
        results = [_process(job) for job in jobs]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(_process, jobs, chunksize=64)

    write_manifest_csv([entry for entry, _ in results], args.out_manifest)
    if args.mode == "test":
        ops_path = args.out_manifest.with_name(args.out_manifest.stem + "_ops.csv")
        with open(ops_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "op"])
            writer.writerows((entry.path, op) for entry, op in results)
        print(f"wrote {ops_path}")
    print(f"wrote {len(results)} images to {args.out_dir} and {args.out_manifest}")


if __name__ == "__main__":
    main()
