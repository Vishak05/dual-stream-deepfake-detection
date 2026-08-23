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
