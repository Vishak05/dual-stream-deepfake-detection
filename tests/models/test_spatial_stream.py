import torch
from dsfnet.models.spatial_stream import SpatialStream


def test_spatial_stream_output_shape():
    model = SpatialStream(pretrained=False)
    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 512)
    assert out.shape[1] == SpatialStream.EMBED_DIM
