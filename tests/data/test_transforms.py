import numpy as np
import torch
from PIL import Image

from dsfnet.data.transforms import (
    RandomJPEG,
    build_train_transform,
    build_eval_transform,
    denormalize_imagenet,
)


def test_eval_transform_output_shape_and_dtype(make_dummy_image):
    path = make_dummy_image(size=(64, 64))
    image = Image.open(path).convert("RGB")
    tensor = build_eval_transform(image_size=224)(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_output_shape_and_dtype(make_dummy_image):
    path = make_dummy_image(size=(64, 64))
    image = Image.open(path).convert("RGB")
    tensor = build_train_transform(image_size=224)(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_covers_the_papers_augmentations():
    """Sec. III-B lists rotation, flip, scaling, cropping, blur and recompression;
    scaling/cropping and recompression were previously missing."""
    names = [type(t).__name__ for t in build_train_transform().transforms]
    for expected in ("RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
                     "RandomRotation", "RandomApply", "RandomJPEG"):
        assert expected in names, f"{expected} missing from the training pipeline"


def test_random_jpeg_runs_before_tensor_conversion():
    """RandomJPEG needs a PIL image, so it must precede ToTensor."""
    names = [type(t).__name__ for t in build_train_transform().transforms]
    assert names.index("RandomJPEG") < names.index("ToTensor")


def test_random_jpeg_passes_through_when_probability_is_zero(make_dummy_image):
    image = Image.open(make_dummy_image(size=(32, 32))).convert("RGB")
    assert np.array_equal(np.asarray(RandomJPEG(p=0.0)(image)), np.asarray(image))


def test_random_jpeg_recompresses_when_always_applied(make_dummy_image):
    image = Image.open(make_dummy_image(size=(32, 32))).convert("RGB")
    out = RandomJPEG(quality_range=(50, 50), p=1.0)(image)
    assert out.size == image.size
    assert not np.array_equal(np.asarray(out), np.asarray(image))


def test_denormalize_inverts_the_training_normalization(make_dummy_image):
    """The frequency stream needs true luminance, which means undoing the
    per-channel normalization first."""
    image = Image.open(make_dummy_image(size=(224, 224))).convert("RGB")
    normalized = build_eval_transform()(image).unsqueeze(0)
    recovered = denormalize_imagenet(normalized)

    expected = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    assert torch.allclose(recovered[0], expected, atol=1e-4)
