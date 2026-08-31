import pytest
import torch

from dsfnet.models.frequency_stream import FrequencyStream


def test_frequency_stream_output_shape():
    model = FrequencyStream()
    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 256)
    assert out.shape[1] == FrequencyStream.EMBED_DIM


def test_frequency_stream_gradients_flow():
    model = FrequencyStream()
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    out.sum().backward()
    assert model.conv1.weight.grad is not None


@pytest.mark.parametrize("variant,transform", [("radial", "fft"), ("radial", "dct"), ("scalar", "fft")])
def test_every_variant_produces_a_finite_embedding(variant, transform):
    stream = FrequencyStream(variant=variant, transform=transform, n_bins=8)
    out = stream(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, FrequencyStream.EMBED_DIM)
    assert torch.isfinite(out).all()


def test_input_channels_follow_the_variant():
    """The scalar map has one channel; the radial profile has one per bin."""
    assert FrequencyStream(variant="scalar").conv1.in_channels == 1
    assert FrequencyStream(variant="radial", n_bins=12).conv1.in_channels == 12


def test_spectral_map_matches_the_configured_variant():
    x = torch.randn(2, 3, 64, 64)
    assert FrequencyStream(variant="scalar").spectral_map(x).shape == (2, 1, 2, 2)
    assert FrequencyStream(variant="radial", n_bins=8).spectral_map(x).shape == (2, 8, 2, 2)


def test_input_is_batch_normalized_before_the_first_convolution():
    """Algorithm 2's 'standardize S across the batch'. Log-power values are large
    and far from zero-mean, so without this the first conv starts badly
    conditioned."""
    stream = FrequencyStream(variant="radial", n_bins=8)
    assert isinstance(stream.input_norm, torch.nn.BatchNorm2d)
    assert stream.input_norm.num_features == 8


def test_frequency_stream_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        FrequencyStream(variant="nonsense")
