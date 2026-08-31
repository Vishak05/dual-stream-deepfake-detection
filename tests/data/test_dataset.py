import torch

from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import ManifestEntry
from dsfnet.data.transforms import build_eval_transform


def _entries(make_dummy_image):
    return [
        ManifestEntry(path=str(make_dummy_image(name="real.png", seed=1)), label=0, source="FFHQ"),
        ManifestEntry(path=str(make_dummy_image(name="fake.png", seed=2)), label=1, source="StarGAN"),
    ]


def test_dataset_len_and_getitem(make_dummy_image):
    dataset = DeepfakeImageDataset(_entries(make_dummy_image), transform=build_eval_transform())

    assert len(dataset) == 2

    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 0

    tensor, label = dataset[1]
    assert label == 1


def test_dataset_can_yield_the_source_column(make_dummy_image):
    """The manifest carries `source`, but __getitem__ used to drop it, which made
    per-generator metrics impossible."""
    dataset = DeepfakeImageDataset(
        _entries(make_dummy_image), transform=build_eval_transform(), return_source=True
    )
    tensor, label, source = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert label == 0
    assert source == "FFHQ"


def test_source_is_opt_in(make_dummy_image):
    dataset = DeepfakeImageDataset(_entries(make_dummy_image), transform=build_eval_transform())
    assert len(dataset[0]) == 2
