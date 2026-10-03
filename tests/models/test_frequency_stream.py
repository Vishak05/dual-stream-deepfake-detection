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


def test_frequency_stream_mean_spectrum_mode_output_shape():
    model = FrequencyStream(mode="mean_spectrum")
    x = torch.rand(2, 3, 224, 224)
    assert model.spectral_map(x).shape == (2, 1, 32, 32)
    assert model(x).shape == (2, FrequencyStream.EMBED_DIM)


def test_frequency_stream_rejects_unknown_mode():
    try:
        FrequencyStream(mode="wavelet")
        assert False, "expected ValueError"
    except ValueError:
        pass
