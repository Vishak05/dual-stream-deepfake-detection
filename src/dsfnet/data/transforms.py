import random

from PIL import Image
from torchvision import transforms

from .perturb import jpeg_roundtrip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Matches the base paper's online training augmentation (Sec. III-E)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic resize+normalize used for validation/test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class RandomJPEG:
    """Re-encodes a PIL image as JPEG at a random quality with probability p."""

    def __init__(self, quality: tuple[int, int] = (50, 99), p: float = 0.5):
        self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return jpeg_roundtrip(img, random.randint(*self.quality))


class RandomDownUp:
    """Downscales a PIL image by `factor` and back, losing fine detail the way
    the base paper's 50% scaling does."""

    def __init__(self, factor: float = 0.5):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        small = img.resize((max(1, round(w * self.factor)), max(1, round(h * self.factor))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def build_robust_train_transform(image_size: int = 224) -> transforms.Compose:
    """Label-symmetric robustness augmentation: the base paper's degradations
    (rotation, flips, scaling/cropping, blur, JPEG recompression) applied to
    every training image, real or fake, so degradation can't signal 'fake'."""
    return transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(degrees=180)], p=0.5),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([RandomDownUp(0.5)], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=13, sigma=(0.1, 2.3))], p=0.3),
        RandomJPEG(quality=(50, 99), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
