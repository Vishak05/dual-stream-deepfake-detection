"""Materialize perturbed copies of a folder (base paper Sec. III-A-3 operations).

Used for the symmetric-augmentation control: the base paper perturbs 1,000
StyleGAN3 fakes offline and leaves the real subset unchanged, so a detector can
learn "degraded => fake". Generating a matched set of perturbed *reals* removes
that shortcut, and `manifest.build_symmetric_augmented_manifest` folds them in.

Example:
    python scripts/perturb_images.py \
        --in-dir data/standardized/FFHQ --out-dir data/perturbed_real \
        --limit 1000 --seed 42
"""
import argparse
import random
from pathlib import Path

from PIL import Image

from dsfnet.data.perturbations import apply_perturbation, sample_perturbation

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write perturbed copies of every image in a folder.")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Perturb at most this many images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-mode", choices=("blur", "noise"), default="blur",
                        help="'blur' follows the paper's kernel-size description; "
                             "'noise' follows its 'Gaussian Noise' label.")
    parser.add_argument("--jpeg-quality", type=int, default=None,
                        help="Pin the JPEG quality factor instead of sampling from [50, 99].")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in sorted(Path(args.in_dir).rglob("*")) if p.suffix.lower() in _IMAGE_EXTENSIONS]
    if args.limit is not None:
        paths = paths[: args.limit]

    for i, path in enumerate(paths):
        # Seeded per source path, so the same image always receives the same
        # perturbation no matter how many images are processed or in what order.
        rng = random.Random(f"{args.seed}:{path.name}")
        params = sample_perturbation(rng, jpeg_quality=args.jpeg_quality)
        with Image.open(path) as image:
            out = apply_perturbation(image, params, noise_mode=args.noise_mode, rng=rng)
        out.save(args.out_dir / f"{path.stem}_perturbed.png")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    print(f"wrote {len(paths)} perturbed images -> {args.out_dir}")


if __name__ == "__main__":
    main()
