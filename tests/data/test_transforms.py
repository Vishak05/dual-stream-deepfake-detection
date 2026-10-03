import torch
from PIL import Image
from dsfnet.data.transforms import build_train_transform, build_eval_transform


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


def test_robust_train_transform_output_shape_and_dtype(make_dummy_image):
    from dsfnet.data.transforms import build_robust_train_transform
    image = Image.open(make_dummy_image(size=(256, 256))).convert("RGB")
    for _ in range(5):
        tensor = build_robust_train_transform(image_size=224)(image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32


def test_random_jpeg_always_and_never(make_dummy_image):
    from dsfnet.data.transforms import RandomJPEG
    image = Image.open(make_dummy_image(size=(64, 64))).convert("RGB")
    assert RandomJPEG(p=0.0)(image) is image
    out = RandomJPEG(quality=(50, 50), p=1.0)(image)
    assert out.size == image.size and out is not image
