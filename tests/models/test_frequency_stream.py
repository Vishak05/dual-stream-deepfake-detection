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
