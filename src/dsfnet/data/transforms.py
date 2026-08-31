import io
import random

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEG:
    """Randomly re-encodes a PIL image as JPEG (base paper Sec. III-B, 'random
    recompression'). Must run before ToTensor, since it needs a PIL image."""

    def __init__(self, quality_range: tuple[int, int] = (50, 99), p: float = 0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(quality_range={self.quality_range}, p={self.p})"


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """The base paper's online training augmentation (Sec. III-B / III-E): rotation,
    flip, scaling, cropping, Gaussian blur and JPEG recompression.

    Applied to *both* classes. The base paper perturbs 1,000 fake images offline
    while stating the real subset "remains unchanged" (Sec. III-B), yet perturbs
    every test image (Sec. III-A-3) -- which lets a detector learn "degraded =>
    fake". Symmetric augmentation is what makes that shortcut unavailable; the
    asymmetric variant is reproduced separately in manifest.build_symmetric_*
    for the controlled comparison.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        RandomJPEG(quality_range=(50, 99), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic resize+normalize used for validation/test (no augmentation).

    Test-time perturbations are not applied here: they are baked into the test
    images once by scripts/build_test_set.py, so every model is scored on
    byte-identical inputs.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Inverts the ImageNet normalization, returning approximately [0, 1] RGB.

    The frequency stream needs true luminance, so it has to undo the per-channel
    normalization first -- averaging normalized channels mixes three differently
    scaled quantities and yields an arbitrary DC term.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return x * std + mean
