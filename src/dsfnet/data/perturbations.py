"""Test-time perturbations from the base paper, Sec. III-A-3.

These are applied *once* and written to disk (see scripts/build_test_set.py)
rather than sampled on the fly. Every model must be scored on byte-identical
inputs or the FPR comparison between them is not a comparison at all.

Two places where the paper is ambiguous, both surfaced rather than silently
resolved:

* It labels the operation "Gaussian Noise" but describes it by *kernel size*
  (K in {3^2, 9^2, 12^2}), which is a blur parameter, not a noise parameter.
  ``noise_mode="blur"`` (default) follows the description; ``noise_mode="noise"``
  follows the label and adds true additive Gaussian noise instead.
* Rotation is given as R = 45 + 45U with U ~ Uniform(0, 6), i.e. multiples of 45
  degrees in [45, 315]. Non-right-angle rotations leave black corners; the paper
  does not say whether it expands the canvas, and we do not (``expand=False``),
  which is the reading that keeps the test images the same size.
"""

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageFilter

# Paper kernel sizes K in {3, 9, 12} mapped to PIL Gaussian radii (~(K-1)/2).
_BLUR_RADII = (1.0, 4.0, 5.5)
_NOISE_SIGMAS = (3.0, 9.0, 12.0)
_MIRRORS = ("none", "h", "v", "hv")
_SCALES = (0.5, 2.0)


@dataclass(frozen=True)
class PerturbationParams:
    rotation_deg: int
    scale: float
    blur_index: int
    mirror: str
    jpeg_quality: int


def sample_perturbation(rng: random.Random, jpeg_quality: int | None = None) -> PerturbationParams:
    """Draws one perturbation. Pass an explicit ``jpeg_quality`` to pin that axis
    (used by the compression-robustness sweep) and let the rest vary."""
    return PerturbationParams(
        rotation_deg=45 * rng.randint(1, 7),
        scale=rng.choice(_SCALES),
        blur_index=rng.randrange(len(_BLUR_RADII)),
        mirror=rng.choice(_MIRRORS),
        jpeg_quality=jpeg_quality if jpeg_quality is not None else rng.randint(50, 99),
    )


def apply_perturbation(
    image: Image.Image,
    params: PerturbationParams,
    noise_mode: str = "blur",
    rng: random.Random | None = None,
) -> Image.Image:
    """Applies one sampled perturbation, in the order the paper lists them."""
    if noise_mode not in ("blur", "noise"):
        raise ValueError(f"noise_mode must be 'blur' or 'noise', got {noise_mode!r}")

    out = image.convert("RGB")
    out = out.rotate(params.rotation_deg, resample=Image.BILINEAR, expand=False)

    width, height = out.size
    scaled = (max(1, int(width * params.scale)), max(1, int(height * params.scale)))
    out = out.resize(scaled, resample=Image.BILINEAR)

    if noise_mode == "blur":
        out = out.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADII[params.blur_index]))
    else:
        out = _add_gaussian_noise(out, _NOISE_SIGMAS[params.blur_index], rng or random.Random(0))

    if params.mirror in ("h", "hv"):
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if params.mirror in ("v", "hv"):
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=params.jpeg_quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _add_gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    # numpy is a hard dependency already; import locally to keep this module
    # usable from PIL-only contexts.
    import numpy as np

    seed = rng.randrange(2 ** 31)
    generator = np.random.RandomState(seed)
    arr = np.asarray(image, dtype=np.float32)
    noisy = arr + generator.normal(0.0, sigma, arr.shape)
    return Image.fromarray(noisy.clip(0, 255).astype("uint8"), "RGB")
