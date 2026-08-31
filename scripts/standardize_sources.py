"""Phase 0 -- re-encode every source to a common resolution before anything else.

Why this has to run first: FFHQ is 1024x1024, CelebA aligned crops are about
178x218, and StyleGAN3 is generated at 256x256 or 1024x1024. Resizing straight to
the 224x224 network input therefore applies a *different resampling factor per
source*, and resampling rewrites an image's high-frequency content. That makes
resize factor correlate with label, and a frequency-domain detector can learn the
resize instead of the generator -- which would invalidate every spectral result.

Standardizing all sources to one resolution with one filter removes that
confound. The emitted data card records what was actually done, so the control is
auditable rather than assumed.

Example:
    python scripts/standardize_sources.py \
        --source FFHQ=data/raw/ffhq --source CelebA=data/raw/celeba \
        --source StyleGAN2=data/raw/stylegan2 \
        --out-dir data/standardized --size 256
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_RESAMPLE = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, _, path = value.partition("=")
    return name, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardize all image sources to one resolution.")
    parser.add_argument("--source", type=parse_source, action="append", required=True,
                        help="NAME=PATH of a source folder. Repeatable.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--resample", choices=sorted(_RESAMPLE), default="lanczos")
    parser.add_argument("--data-card", type=Path, default=None,
                        help="Where to write the JSON data card (default: <out-dir>/data_card.json)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    resample = _RESAMPLE[args.resample]
    card = {
        "size": args.size,
        "resample": args.resample,
        "sources": {},
    }

    for name, folder in args.source:
        out_folder = args.out_dir / name
        out_folder.mkdir(parents=True, exist_ok=True)
        resolutions: Counter = Counter()
        written = 0

        for path in sorted(Path(folder).rglob("*")):
            if path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                resolutions[f"{image.width}x{image.height}"] += 1
                resized = image.resize((args.size, args.size), resample=resample)
            resized.save(out_folder / f"{name}_{written:06d}.png")
            written += 1

        card["sources"][name] = {
            "input_folder": str(folder),
            "output_folder": str(out_folder),
            "images": written,
            "original_resolutions": dict(resolutions.most_common()),
        }
        print(f"{name}: {written} images -> {out_folder}", flush=True)

    card_path = args.data_card or (args.out_dir / "data_card.json")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2))
    print(f"data card -> {card_path}")


if __name__ == "__main__":
    main()
