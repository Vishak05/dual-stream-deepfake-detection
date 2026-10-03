"""Image degradations from the base paper (Hafezi et al., IEEE Access 2026).

base_paper_perturbation reproduces the test-set post-processing of Sec. III-A3:
one randomly chosen operation per image. training_degradation reproduces the
Sec. III-B augmentations used to make the 1,000 extra StyleGAN3 training copies.
Both take a random.Random so results are reproducible per image."""
import io
import random

from PIL import Image, ImageOps
from torchvision.transforms import functional as F

TEST_OPS = ("rotation", "scaling", "gaussian_blur", "mirror", "jpeg")
BLUR_KERNELS = (3, 9, 13)   # the paper's 12 rounded up: Gaussian kernels must be odd


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def base_paper_perturbation(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str, int | None]:
    """Returns (perturbed image, op label like 'rotation:135', JPEG quality to
    save with or None). The quality is set only for the 'jpeg' op, so the
    caller can save that image without compressing it a second time."""
    op = rng.choice(TEST_OPS)
    if op == "rotation":
        angle = 45 + 45 * rng.randint(0, 6)           # Eq. (1): R = 45 + 45U
        return img.rotate(angle), f"rotation:{angle}", None
    if op == "scaling":
        factor = rng.choice((0.5, 2.0))                # reduce by 50% or magnify by 100%
        size = (round(img.width * factor), round(img.height * factor))
        return img.resize(size, Image.BILINEAR), f"scaling:{factor}", None
    if op == "gaussian_blur":
        k = rng.choice(BLUR_KERNELS)
        return F.gaussian_blur(img, kernel_size=k), f"gaussian_blur:{k}", None
    if op == "mirror":
        kind = rng.choice(("horizontal", "vertical", "both"))
        if kind in ("horizontal", "both"):
            img = ImageOps.mirror(img)
        if kind in ("vertical", "both"):
            img = ImageOps.flip(img)
        return img, f"mirror:{kind}", None
    quality = rng.randint(50, 99)
    return jpeg_roundtrip(img, quality), f"jpeg:{quality}", quality


def training_degradation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Sec. III-B: random rotation, flip, Gaussian blur, scaling, crop and
    recompression, each applied with probability 0.5 (at least one)."""
    w, h = img.size
    steps = [rng.random() < 0.5 for _ in range(6)]
    if not any(steps):
        steps[rng.randrange(6)] = True
    rotate, flip, blur, scale, crop, recompress = steps
    if rotate:
        img = img.rotate(rng.uniform(0, 360))
    if flip:
        img = ImageOps.mirror(img) if rng.random() < 0.5 else ImageOps.flip(img)
    if blur:
        img = F.gaussian_blur(img, kernel_size=rng.choice(BLUR_KERNELS))
    if scale:
        factor = rng.uniform(0.5, 1.5)
        img = img.resize((round(w * factor), round(h * factor)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if crop:
        side = rng.uniform(0.6, 1.0)
        cw, ch = round(w * side), round(h * side)
        left, upper = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((left, upper, left + cw, upper + ch)).resize((w, h), Image.BILINEAR)
    if recompress:
        img = jpeg_roundtrip(img, rng.randint(50, 99))
    return img
