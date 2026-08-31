"""Build the fixed, perturbed test set once and write it to disk.

The base paper applies randomized post-processing (rotation, scaling, blur,
mirroring, JPEG) to all 7,000 test images. Sampling those on the fly at
evaluation time would give every model a *different* test set, so an FPR
difference between two models could be nothing but perturbation luck. Baking them
in once means every model is scored on byte-identical inputs.

Example:
    python scripts/build_test_set.py \
        --real-source FFHQ=data/standardized/FFHQ \
        --fake-source StyleGAN2=data/standardized/StyleGAN2 \
        --per-source 1000 --out-dir data/test_perturbed \
        --out-manifest data/test_manifest.csv

Sweep compression robustness by pinning quality and varying it across runs:
    --jpeg-quality 30 --out-dir data/test_q30 --out-manifest data/test_q30.csv
"""
import argparse
import random
from pathlib import Path

from PIL import Image

from dsfnet.data.manifest import ManifestEntry, class_counts, write_manifest_csv
from dsfnet.data.perturbations import apply_perturbation, sample_perturbation

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, _, path = value.partition("=")
    return name, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the fixed perturbed test set.")
    parser.add_argument("--real-source", type=parse_source, action="append", default=[],
                        help="NAME=PATH of a real-image source. Repeatable.")
    parser.add_argument("--fake-source", type=parse_source, action="append", default=[],
                        help="NAME=PATH of a fake-image source. Repeatable.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=1000,
                        help="Images to take from each source (the paper uses 1,000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-mode", choices=("blur", "noise"), default="blur")
    parser.add_argument("--jpeg-quality", type=int, default=None,
                        help="Pin the JPEG quality factor (for the robustness sweep).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.real_source and not args.fake_source:
        raise SystemExit("provide at least one --real-source or --fake-source")

    entries: list[ManifestEntry] = []
    sources = [(name, folder, 0) for name, folder in args.real_source]
    sources += [(name, folder, 1) for name, folder in args.fake_source]

    for name, folder, label in sources:
        out_folder = args.out_dir / name
        out_folder.mkdir(parents=True, exist_ok=True)
        paths = [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in _IMAGE_EXTENSIONS]
        paths = paths[: args.per_source]

        for path in paths:
            rng = random.Random(f"{args.seed}:{name}:{path.name}")
            params = sample_perturbation(rng, jpeg_quality=args.jpeg_quality)
            with Image.open(path) as image:
                out = apply_perturbation(image, params, noise_mode=args.noise_mode, rng=rng)
            out_path = out_folder / f"{path.stem}.png"
            out.save(out_path)
            entries.append(ManifestEntry(path=str(out_path), label=label, source=name))

        print(f"{name} (label {label}): {len(paths)} images -> {out_folder}", flush=True)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(entries, args.out_manifest)
    counts = class_counts(entries)
    print(f"test manifest -> {args.out_manifest}")
    print(f"  real={counts['real']} fake={counts['fake']} fake_prior={counts['fake_prior']:.3f}")


if __name__ == "__main__":
    main()
