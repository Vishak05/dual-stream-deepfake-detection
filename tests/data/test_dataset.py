from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import ManifestEntry
from dsfnet.data.transforms import build_eval_transform


def test_dataset_len_and_getitem(make_dummy_image):
    real_path = make_dummy_image(name="real.png", seed=1)
    fake_path = make_dummy_image(name="fake.png", seed=2)
    entries = [
        ManifestEntry(path=str(real_path), label=0, source="FFHQ"),
        ManifestEntry(path=str(fake_path), label=1, source="StarGAN"),
    ]
    dataset = DeepfakeImageDataset(entries, transform=build_eval_transform())

    assert len(dataset) == 2

    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 0

    tensor, label = dataset[1]
    assert label == 1
